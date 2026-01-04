#!/usr/bin/env python3
"""
Compute coherence (c_v, u_mass, c_npmi, c_uci) using topics from combined_rows.txt
and narratives from a CSV that may have ONLY ONE COLUMN (text). In that case, we
use the DataFrame index as the ID source.

Use --index_offset if your CSV's index doesn't match the Row IDs directly.
Example: if CSV index starts at 0 but rows file uses 10000.., pass --index_offset 10000
"""

import argparse
import io
import random
import re
import sys
from typing import Dict, List, Tuple, Optional
import math
from collections import Counter

import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

ROW_LINE_RE = re.compile(r'^\s*Row\s+(\d+):\s*(.*\S)\s*$')

# ----------------- helpers -----------------

def to_int_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to integer, handling various edge cases.
    """
    try:
        # Try direct conversion to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        # Convert to int, this will raise an error if there are NaN values
        return numeric_series.astype(int)
    except (ValueError, TypeError):
        # If conversion fails, try to extract integers from strings
        def extract_int(val):
            if pd.isna(val):
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                # Try to extract first number from string
                str_val = str(val)
                numbers = re.findall(r'\d+', str_val)
                return int(numbers[0]) if numbers else None
        
        converted = series.apply(extract_int)
        # Drop NaN values and convert to int
        return converted.dropna().astype(int)

def parse_rows_file(path: str) -> Dict[int, List[str]]:
    print(f"[INFO] Reading topics from {path} ...")
    topics_by_row = {}
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ROW_LINE_RE.match(line)
            if not m:
                continue
            rid = int(m.group(1))
            payload = m.group(2)
            phrases = [p.strip() for p in payload.split(",") if p.strip()]
            
            # Break multi-word phrases into individual words
            all_words = []
            for phrase in phrases:
                # Split phrase into words and clean them
                words = phrase.lower().split()
                cleaned_words = [re.sub(r'[^\w]', '', word) for word in words if word.strip()]
                all_words.extend([w for w in cleaned_words if w])
            
            if all_words:
                topics_by_row[rid] = all_words
    print(f"[DONE] Parsed {len(topics_by_row)} rows with topics.")
    return topics_by_row

def preprocess_texts(texts: List[str]) -> List[List[str]]:
    print(f"[INFO] Preprocessing {len(texts)} texts ...")
    def tok(doc: str) -> List[str]:
        return [w for w in simple_preprocess(str(doc), deacc=True, min_len=2) if w not in STOPWORDS]
    out = [tok(t) for t in texts]
    print("[DONE] Text preprocessing complete.")
    return out

def build_dict_corpus(texts_tok):
    print("[INFO] Building dictionary and corpus ...")
    dictionary = Dictionary(texts_tok)
    dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=100000)
    corpus = [dictionary.doc2bow(t) for t in texts_tok]
    print(f"[DONE] Dictionary size: {len(dictionary)} | Documents: {len(corpus)}.")
    if sum(len(b) for b in corpus) == 0:
        raise ValueError("Corpus is empty after preprocessing/filter_extremes. Increase sample size or relax filters.")
    return dictionary, corpus

def sanitize_topics(raw_topics, dictionary):
    """
    Ensure topics is a list[list[str]] of tokens present in dictionary.
    - Converts to lowercase and strips non-alphanumeric characters
    - Filters tokens not found in dictionary
    - Drops topics with < 2 kept tokens (coherence becomes unstable with singletons)
    """
    clean = []
    dropped_empty = 0
    dropped_short = 0
    kept_tokens = 0
    total_tokens = 0

    for idx, topic in enumerate(raw_topics):
        # Make sure topic is an iterable of items
        if topic is None:
            dropped_empty += 1
            continue
        if isinstance(topic, str):
            # A single string is not a valid topic; split on spaces as last resort
            topic_items = topic.split()
        else:
            try:
                topic_items = list(topic)
            except Exception:
                # not iterable
                dropped_empty += 1
                continue

        # Normalize -> strings
        toks = []
        for it in topic_items:
            if it is None:
                continue
            s = str(it).strip().lower()
            if not s:
                continue
            # Clean the word (remove non-alphanumeric chars)
            cleaned = re.sub(r'[^\w]', '', s)
            if cleaned:
                toks.append(cleaned)

        total_tokens += len(toks)
        # keep only tokens that exist in dictionary
        toks = [w for w in toks if w in dictionary.token2id]
        kept_tokens += len(toks)

        if len(toks) == 0:
            dropped_empty += 1
            continue
        if len(toks) == 1:
            dropped_short += 1
            continue

        clean.append(toks)

    print(f"[INFO] Topic sanitization: {len(clean)} usable topics "
          f"(dropped empty: {dropped_empty}, dropped singletons: {dropped_short}).")
    print(f"[INFO] Tokens: kept {kept_tokens} / raw {total_tokens} after dictionary filter.")
    if not clean:
        raise ValueError("All topics became empty after sanitization. "
                         "Consider increasing sample size or relaxing preprocessing.")
    return clean

def compute_diversity_scores(topics):
    """
    Compute diversity scores for topics.
    """
    if not topics:
        return {"topic_diversity": 0.0, "avg_pairwise_jaccard_distance": 0.0}
    
    # Topic diversity: proportion of unique words across all topics
    all_words = []
    unique_words = set()
    for topic in topics:
        all_words.extend(topic)
        unique_words.update(topic)
    
    topic_diversity = len(unique_words) / len(all_words) if all_words else 0.0
    
    # Average pairwise Jaccard distance
    jaccard_distances = []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            set_i = set(topics[i])
            set_j = set(topics[j])
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            jaccard_similarity = intersection / union if union > 0 else 0.0
            jaccard_distance = 1 - jaccard_similarity
            jaccard_distances.append(jaccard_distance)
    
    avg_pairwise_jaccard_distance = np.mean(jaccard_distances) if jaccard_distances else 0.0
    
    return {
        "topic_diversity": topic_diversity,
        "avg_pairwise_jaccard_distance": avg_pairwise_jaccard_distance
    }

def compute_coherence_for_topics(topics, texts_tok, dictionary, corpus):
    print("[INFO] Sanitizing topics before coherence ...")
    topics = sanitize_topics(topics, dictionary)

    print("[INFO] Computing coherence metrics ...")
    cm_cv   = CoherenceModel(topics=topics, texts=texts_tok, dictionary=dictionary, coherence="c_v")
    cm_npmi = CoherenceModel(topics=topics, texts=texts_tok, dictionary=dictionary, coherence="c_npmi")
    cm_uci  = CoherenceModel(topics=topics, texts=texts_tok, dictionary=dictionary, coherence="c_uci")
    cm_um   = CoherenceModel(topics=topics, corpus=corpus,  dictionary=dictionary, coherence="u_mass")

    coherence_scores = {
        "c_v":    cm_cv.get_coherence(),
        "u_mass": cm_um.get_coherence(),
        "c_npmi": cm_npmi.get_coherence(),
        "c_uci":  cm_uci.get_coherence(),
    }
    
    print("[INFO] Computing diversity scores ...")
    diversity_scores = compute_diversity_scores(topics)
    
    print("[DONE] All metrics computed.")
    
    return {
        "coherence": coherence_scores,
        "diversity": diversity_scores
    }

# ----------------- main -----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows_file", default="combined_rows.txt", help="Path to combined rows/topics text file")
    parser.add_argument("--csv_file", default="filtered_complaints_narratives.csv", help="CSV with narratives")
    parser.add_argument("--text_col", default=None, help="CSV text column name. If omitted and only one column is present, that column is used.")
    parser.add_argument("--sample_size", type=int, default=20000, help="Number of row IDs to sample")
    parser.add_argument("--index_offset", type=int, default=0, help="Offset to add to CSV row index to align with Row IDs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1) Rows/topics + sample
    topics_by_row = parse_rows_file(args.rows_file)
    if not topics_by_row:
        print("[ERROR] No topics found in rows file.")
        sys.exit(1)

    all_ids = sorted(topics_by_row.keys())
    k = min(args.sample_size, len(all_ids))
    print(f"[INFO] Sampling {k} IDs from {len(all_ids)} available ...")
    sample_ids = sorted(random.sample(all_ids, k=k))
    sample_set = set(sample_ids)
    print(f"[DONE] Sampling complete. Sample IDs range: {min(sample_ids)} to {max(sample_ids)}")

    # 2) Load CSV
    print(f"[INFO] Loading CSV: {args.csv_file} ...")
    try:
        df = pd.read_csv(args.csv_file)
        print(f"[DONE] CSV loaded with {len(df)} rows and {len(df.columns)} column(s).")
        print(f"[INFO] CSV index range: {df.index.min()} to {df.index.max()}")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        sys.exit(1)

    # 3) Determine text column & ID source
    if args.text_col and args.text_col in df.columns:
        text_col = args.text_col
        print(f"[INFO] Using provided text column '{text_col}'.")
    elif df.shape[1] == 1:
        text_col = df.columns[0]
        print(f"[INFO] Detected single-column CSV. Using '{text_col}' as text and DataFrame index as ID source.")
    else:
        # Fallback guess if multiple columns but no text_col provided
        candidates = [c for c in df.columns if c.lower() in ("narrative", "complaint", "complaint_text", "text", "description")]
        if candidates:
            text_col = candidates[0]
        else:
            # Get the first object (string) column
            string_cols = df.select_dtypes(include=["object"]).columns
            if len(string_cols) > 0:
                text_col = string_cols[0]
            else:
                print("[ERROR] No suitable text column found.")
                sys.exit(1)
        print(f"[INFO] Auto-detected text column '{text_col}'. Using DataFrame index as ID source.")

    # 4) Match by INDEX (+ optional offset)
    print(f"[INFO] Filtering CSV to sampled IDs using DataFrame index (offset: {args.index_offset}) ...")
    try:
        # Create an integer view of the index and apply offset
        idx_series = pd.Series(df.index)
        idx_adj = to_int_series(idx_series) + args.index_offset
        
        # Create a mask for rows that match our sample
        mask = idx_adj.isin(sample_set)
        filtered = df.loc[mask.values].copy()
        
        print(f"[DONE] Filtered to {len(filtered)} rows (ID source: index + offset {args.index_offset}).")
        
        if filtered.empty:
            print("[ERROR] No matching rows found after applying index offset.")
            print(f"       CSV index range: {df.index.min()} to {df.index.max()}")
            print(f"       Adjusted range: {idx_adj.min()} to {idx_adj.max()}")
            print(f"       Sample ID range: {min(sample_ids)} to {max(sample_ids)}")
            print("       Tip: Check if you need a different --index_offset value")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Failed to process DataFrame index: {e}")
        print("       Ensure your CSV has a numeric index or can be converted to one.")
        sys.exit(1)

    # 5) Clean text
    if text_col not in filtered.columns:
        print(f"[ERROR] Text column '{text_col}' not found after filtering.")
        sys.exit(1)
    
    # Remove rows with empty/null text
    initial_count = len(filtered)
    filtered = filtered[filtered[text_col].notna()]
    filtered = filtered[filtered[text_col].astype(str).str.strip().ne("")]
    
    if len(filtered) < initial_count:
        print(f"[INFO] Removed {initial_count - len(filtered)} rows with empty text.")
    
    if filtered.empty:
        print("[ERROR] No matching rows with non-empty text after filtering.")
        sys.exit(1)

    # 6) Align topics only for rows we kept
    print("[INFO] Aligning topics to filtered rows ...")
    # Get the actual IDs that correspond to our filtered rows
    kept_idx_adj = to_int_series(pd.Series(filtered.index)) + args.index_offset
    kept_ids = set(kept_idx_adj.tolist())
    
    topics = []
    matched_ids = []
    for rid in sample_ids:
        if rid in kept_ids and rid in topics_by_row:
            topic_list = topics_by_row[rid]
            if topic_list:  # Only add non-empty topics
                topics.append(topic_list)
                matched_ids.append(rid)
    
    if not topics:
        print("[ERROR] No topics matched with available text data.")
        print(f"       Sample IDs: {sorted(list(sample_set))[:10]}... (showing first 10)")
        print(f"       Kept IDs: {sorted(list(kept_ids))[:10]}... (showing first 10)")
        print("       Tip: Verify your --index_offset matches the relationship between CSV index and Row IDs")
        sys.exit(1)
    
    print(f"[DONE] Prepared {len(topics)} topics matching {len(matched_ids)} text records.")

    # 7) Preprocess + dictionary/corpus
    texts_tok = preprocess_texts(filtered[text_col].astype(str).tolist())
    dictionary, corpus = build_dict_corpus(texts_tok)

    # 8) Coherence and all metrics
    all_scores = compute_coherence_for_topics(topics, texts_tok, dictionary, corpus)

    # 9) Report
    print("\n" + "="*50)
    print("COMPREHENSIVE TOPIC EVALUATION RESULTS")
    print("="*50)
    print(f"Topics evaluated: {len(topics)}")
    print(f"Text documents: {len(filtered)}")
    print(f"Dictionary size: {len(dictionary)}")
    
    print("\n📊 COHERENCE SCORES:")
    for k, v in all_scores["coherence"].items():
        print(f"  {k.upper():>7}: {v:.4f}")

    print("\n🌈 DIVERSITY SCORES:")
    for k, v in all_scores["diversity"].items():
        print(f"  {k}: {v:.4f}")

    print(f"\nSample topics (first {min(10, len(topics))}):")
    for i, t in enumerate(topics[:10]):
        print(f"  Topic {i:02d}: {', '.join(t)}")
    
    print(f"\nMatched Row IDs (first {min(10, len(matched_ids))}):")
    print(f"  {matched_ids[:10]}")

if __name__ == "__main__":
    main()