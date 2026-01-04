#!/usr/bin/env python3

import os
import time
import requests
import pandas as pd
import csv
import re
from io import StringIO

# ─── Configuration ──────────────────────────────────────────────────────────────
# Load API key from env.local file
def load_api_key():
    try:
        with open('env.local', 'r') as f:
            for line in f:
                if line.strip().startswith('OPEN_ROUTER_KEY'):
                    return line.split('=')[1].strip().strip('"')
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None

CLEAN_PATH         = "filtered_complaints_narratives.csv"  # Use the filtered CSV file
OUTPUT_TXT_PATH    = "Deepseek_Results_4k-10k.txt"
OPENROUTER_API_KEY = load_api_key()
MODEL              = "deepseek/deepseek-r1-0528:free"
CSV_CHUNK_SIZE     = 100      # Read CSV in chunks of 5 rows
PROCESS_CHUNK_SIZE = 10      # Process 5 rows at a time for API calls
PAUSE_SEC          = 1      # throttle between calls (adjust if you hit rate limits)

# ─── ROW RANGE CONFIGURATION ────────────────────────────────────────────────────
START_ROW = 4130      # Starting row number (1-based, excluding header)
END_ROW = 10000     # Ending row number (1-based, inclusive) - None for all rows
# ────────────────────────────────────────────────────────────────────────────────

# Global counter for credit errors
credit_error_count = 0
MAX_CREDIT_ERRORS = 3

SYSTEM_INSTRUCTIONS = """
You are a topic-modeling assistant.
For each row of data I provide, extract 5 key topics from the consumer complaint narrative.

Output format - one line per row:
Row <row_number>: <Topic1>, <Topic2>, <Topic3>, <Topic4>, <Topic5>

Use the EXACT row numbers I provide in the data.
No additional text, explanations, or formatting.
"""


def clean_narrative(text):
    """Clean narrative text for processing"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text


def send_chunk(chunk_data):
    """Send a chunk to the API"""
    global credit_error_count
    
    # Prepare data lines
    data_lines = []
    for csv_row_num, narrative in chunk_data:
        clean_text = clean_narrative(narrative)
        if not clean_text:
            continue
        data_lines.append(f"Row {csv_row_num}: {clean_text}")
    if not data_lines:
        return ""

    # Create prompt
    data_content = "\n\n".join(data_lines)
    user_content = f"Analyze these consumer complaints and provide 5 topics for each:\n\n{data_content}"

    body = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_content}
        ]
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json=body,
            timeout=60
        )
        
        # Check for specific HTTP status codes
        if resp.status_code == 402:
            # 402: Insufficient credits
            credit_error_count += 1
            error_msg = "Insufficient credits (HTTP 402)"
            try:
                error_json = resp.json()
                if "error" in error_json and "message" in error_json["error"]:
                    error_msg = error_json["error"]["message"]
            except:
                pass
            
            print(f"Credit error detected ({credit_error_count}/{MAX_CREDIT_ERRORS}): {error_msg}")
            
            if credit_error_count >= MAX_CREDIT_ERRORS:
                print(f"Maximum credit errors ({MAX_CREDIT_ERRORS}) reached. Stopping execution.")
                raise SystemExit("Stopping due to insufficient credits (HTTP 402)")
            
            return ""  # Return empty for this chunk but continue
        
        # Check for HTTP errors (other than 402)
        resp.raise_for_status()
        
        # Parse response
        response_json = resp.json()
        
        # Check if there's an error in the response (shouldn't happen with successful HTTP status)
        if "error" in response_json:
            error_message = response_json["error"].get("message", "Unknown error")
            print(f"API error (continuing): {error_message}")
            return ""
        
        # Extract content from successful response
        result = response_json["choices"][0]["message"]["content"].strip()

        # Clean and filter response lines
        lines = []
        for line in result.split('\n'):
            line = line.strip()
            if line and re.match(r'^Row \d+:', line):
                lines.append(line)
        return "\n".join(lines)

    except requests.exceptions.HTTPError as e:
        # Handle other HTTP status errors (400, 401, 403, 408, 429, 502, 503)
        status_code = e.response.status_code if e.response else "Unknown"
        error_descriptions = {
            400: "Bad Request (invalid or missing params, CORS)",
            401: "Invalid credentials (OAuth session expired, disabled/invalid API key)",
            403: "Input was flagged by moderation",
            408: "Request timed out",
            429: "Rate limited",
            502: "Model is down or invalid response",
            503: "No available model provider"
        }
        
        error_desc = error_descriptions.get(status_code, f"HTTP {status_code}")
        print(f"HTTP error {status_code} (continuing): {error_desc}")
        return ""
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed (continuing): {e}")
        return ""
    except Exception as e:
        print(f"Error processing API response (continuing): {e}")
        return ""


def process_csv_in_chunks(file_path):
    """Process CSV in chunks to avoid memory constraints"""
    total_processed = 0
    total_valid = 0
    
    try:
        # Calculate skip rows and number of rows to read
        skip_rows = list(range(1, START_ROW)) if START_ROW > 1 else None  # Skip rows before START_ROW (excluding header)
        nrows = (END_ROW - START_ROW + 1) if END_ROW is not None else None
        
        print(f"Processing rows {START_ROW} to {END_ROW if END_ROW else 'end'}")
        
        # Read CSV in chunks with row range
        chunk_iter = pd.read_csv(
            file_path, 
            chunksize=CSV_CHUNK_SIZE, 
            dtype={"Consumer complaint narrative": str},
            skiprows=skip_rows,
            nrows=nrows
        )
        
        for chunk_num, df_chunk in enumerate(chunk_iter, 1):
            print(f"Processing CSV chunk {chunk_num} ({len(df_chunk)} rows)...")
            
            # Process this chunk in sub-chunks of 5 rows
            valid_data = []
            for idx, row in df_chunk.iterrows():
                # Calculate the actual row number in the original CSV
                csv_row = START_ROW + total_processed + (idx - df_chunk.index[0])
                narrative = row.get("Consumer complaint narrative", "")
                text = clean_narrative(narrative)
                if text and len(text) > 20 and "consumer complaint narrative" not in text.lower():
                    valid_data.append((csv_row, narrative))
            
            total_valid += len(valid_data)
            print(f"  Found {len(valid_data)} valid narratives in this chunk")
            
            # Process valid data in sub-chunks of 5
            for i in range(0, len(valid_data), PROCESS_CHUNK_SIZE):
                sub_chunk = valid_data[i:i+PROCESS_CHUNK_SIZE]
                nums = [r for r, _ in sub_chunk]
                print(f"    Processing sub-chunk - CSV rows {nums}")
                
                res = send_chunk(sub_chunk)
                if res:
                    with open(OUTPUT_TXT_PATH, "a", encoding="utf-8") as f:
                        f.write(f"Rows {nums}:\n")
                        f.write(res + "\n\n")
                else:
                    print(f"    Warning: No results for sub-chunk with rows {nums}")
                
                time.sleep(PAUSE_SEC)
            
            total_processed += len(df_chunk)
            current_row = START_ROW + total_processed - 1
            print(f"  Completed chunk {chunk_num}. Current row: {current_row}")
    
    except SystemExit:
        # Re-raise SystemExit to stop execution
        raise
    except Exception as e:
        print(f"Error processing CSV chunks: {e}")
        return False
    
    return total_valid


def main():
    print("=== Starting Optimized Topic Modeling Process ===")
    
    # Check if API key is loaded
    if not OPENROUTER_API_KEY:
        print("Error: Could not load OpenRouter API key from env.local file")
        return
    
    print(f"Configuration:")
    print(f"  CSV chunk size: {CSV_CHUNK_SIZE} rows")
    print(f"  Process chunk size: {PROCESS_CHUNK_SIZE} rows")
    print(f"  Row range: {START_ROW} to {END_ROW if END_ROW else 'end'}")
    print(f"  Input file: {CLEAN_PATH}")
    print(f"  Output file: {OUTPUT_TXT_PATH}")
    print(f"  Max credit errors before stopping: {MAX_CREDIT_ERRORS}")
    print()
    
    # Check if input file exists
    if not os.path.exists(CLEAN_PATH):
        print(f"Error: Input file '{CLEAN_PATH}' not found.")
        print("Please run filter_complaints.py first to create the filtered CSV.")
        return
    
    # Initialize output
    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("Topic Modeling Results (Optimized)\n")
        f.write("="*50 + "\n")
        f.write(f"Row range: {START_ROW} to {END_ROW if END_ROW else 'end'}\n\n")
    
    # Process CSV in chunks
    print("Starting CSV processing...")
    try:
        total_valid = process_csv_in_chunks(CLEAN_PATH)
        
        if total_valid is False:
            print("Failed to process CSV file")
            return
        
        print(f"\n=== Processing Complete ===")
        print(f"Total valid narratives processed: {total_valid}")
        print(f"Results saved to: {OUTPUT_TXT_PATH}")
        print(f"Credit errors encountered: {credit_error_count}")
        
    except SystemExit as e:
        print(f"\n=== Processing Stopped ===")
        print(f"Reason: {e}")
        print(f"Credit errors encountered: {credit_error_count}")
        print(f"Partial results saved to: {OUTPUT_TXT_PATH}")


if __name__ == "__main__":
    main()