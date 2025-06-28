#!/usr/bin/env python3
"""
Create a properly split dataset for ColBERT training where:
- Queries: Patient case descriptions (desc_1, desc_2, desc_3)
- Ground truth: Original ACR variants (not procedures!)
- Corpus: ACR variants split by procedure to prevent data leakage
"""

import pandas as pd
import json
from collections import defaultdict
import random
from pathlib import Path

# Load datasets
acr_df = pd.read_csv('dataset/acr.csv', sep='|')
cases_df = pd.read_csv('patient_cases_2.csv', sep='|')

print(f"Loaded {len(acr_df)} ACR variants and {len(cases_df)} patient cases")

# Extract unique procedures from ACR dataset
acr_df['procedure_name'] = acr_df['Variant'].str.split('.').str[0]
unique_procedures = acr_df['procedure_name'].unique()
print(f"Found {len(unique_procedures)} unique procedures in ACR dataset")

# Create procedure splits (same as before to prevent data leakage)
random.seed(42)
procedures_list = list(unique_procedures)
random.shuffle(procedures_list)

# Calculate split sizes
total_procedures = len(procedures_list)
train_size = int(0.7 * total_procedures)
val_size = int(0.1 * total_procedures)
test_size = total_procedures - train_size - val_size

train_procedures = set(procedures_list[:train_size])
val_procedures = set(procedures_list[train_size:train_size + val_size])
test_procedures = set(procedures_list[train_size + val_size:])

print(f"Split procedures: {len(train_procedures)} train, {len(val_procedures)} val, {len(test_procedures)} test")

# Split ACR variants by procedure
def get_split_for_acr_variant(variant):
    procedure = variant.split('.')[0]
    if procedure in train_procedures:
        return 'train'
    elif procedure in val_procedures:
        return 'val'
    else:
        return 'test'

acr_df['split'] = acr_df['Variant'].apply(get_split_for_acr_variant)

# Create split corpora
train_acr = acr_df[acr_df['split'] == 'train'].copy()
val_acr = acr_df[acr_df['split'] == 'val'].copy()
test_acr = acr_df[acr_df['split'] == 'test'].copy()

print(f"ACR variants split: {len(train_acr)} train, {len(val_acr)} val, {len(test_acr)} test")

# Create mapping from original_variant to ACR corpus indices
def create_variant_to_corpus_mapping(acr_variants):
    """Create mapping from original variant text to corpus index"""
    mapping = {}
    for idx, row in acr_variants.iterrows():
        variant_text = row['Variant']
        corpus_idx = acr_variants.index.get_loc(idx)
        mapping[variant_text] = corpus_idx
    return mapping

train_variant_mapping = create_variant_to_corpus_mapping(train_acr)
val_variant_mapping = create_variant_to_corpus_mapping(val_acr)
test_variant_mapping = create_variant_to_corpus_mapping(test_acr)

# Split patient cases by procedure
def get_split_for_patient_case(case_row):
    original_variant = case_row['original_variant']
    procedure = original_variant.split('.')[0]
    if procedure in train_procedures:
        return 'train'
    elif procedure in val_procedures:
        return 'val'
    else:
        return 'test'

cases_df['split'] = cases_df.apply(get_split_for_patient_case, axis=1)

# Split cases
train_cases = cases_df[cases_df['split'] == 'train'].copy()
val_cases = cases_df[cases_df['split'] == 'val'].copy()
test_cases = cases_df[cases_df['split'] == 'test'].copy()

print(f"Patient cases split: {len(train_cases)} train, {len(val_cases)} val, {len(test_cases)} test")

# Create output directory
output_dir = Path('corrected_colbert_split')
output_dir.mkdir(exist_ok=True)

# Function to write corpus
def write_corpus(acr_variants, filename):
    corpus_data = []
    for idx, row in acr_variants.iterrows():
        corpus_data.append({
            'id': acr_variants.index.get_loc(idx),
            'text': row['Variant']
        })
    
    with open(output_dir / filename, 'w') as f:
        for item in corpus_data:
            f.write(json.dumps(item) + '\n')
    
    return len(corpus_data)

# Function to write queries with CORRECT ground truth (original variants, not procedures!)
def write_queries(cases, variant_mapping, acr_variants, filename):
    queries_data = []
    missing_variants = 0
    
    for _, case in cases.iterrows():
        case_id = case.name
        original_variant = case['original_variant']
        
        # Find the corpus index for this original variant
        if original_variant in variant_mapping:
            ground_truth_corpus_idx = variant_mapping[original_variant]
        else:
            print(f"Warning: Original variant not found in corpus: {original_variant[:100]}...")
            missing_variants += 1
            continue
        
        # Create queries for each description variant
        for desc_type in ['desc_1', 'desc_2', 'desc_3']:
            query_text = case[desc_type]
            
            queries_data.append({
                'query': query_text,
                'query_id': f"{case_id}_{desc_type}",
                'case_id': case_id,
                'query_type': desc_type,
                'ground_truth_variant': original_variant,  # The CORRECT ground truth!
                'ground_truth_corpus_idx': ground_truth_corpus_idx
            })
    
    with open(output_dir / filename, 'w') as f:
        for item in queries_data:
            f.write(json.dumps(item) + '\n')
    
    if missing_variants > 0:
        print(f"Warning: {missing_variants} cases had variants not found in {filename.split('_')[0]} corpus")
    
    return len(queries_data)

# Write corpora
train_corpus_size = write_corpus(train_acr, 'train_corpus.jsonl')
val_corpus_size = write_corpus(val_acr, 'val_corpus.jsonl') 
test_corpus_size = write_corpus(test_acr, 'test_corpus.jsonl')

# Write queries
train_queries_size = write_queries(train_cases, train_variant_mapping, train_acr, 'train_queries.jsonl')
val_queries_size = write_queries(val_cases, val_variant_mapping, val_acr, 'val_queries.jsonl')
test_queries_size = write_queries(test_cases, test_variant_mapping, test_acr, 'test_queries.jsonl')

# Generate training triplets (query, positive, negative)
def generate_triplets(cases, variant_mapping, acr_variants, filename, num_negatives=5):
    triplets = []
    acr_variants_list = acr_variants['Variant'].tolist()
    
    for _, case in cases.iterrows():
        case_id = case.name
        original_variant = case['original_variant']
        
        if original_variant not in variant_mapping:
            continue
            
        positive_idx = variant_mapping[original_variant]
        
        # Generate negative samples (random variants from same split)
        available_negatives = [i for i in range(len(acr_variants_list)) if i != positive_idx]
        
        for desc_type in ['desc_1', 'desc_2', 'desc_3']:
            query_text = case[desc_type]
            query_id = f"{case_id}_{desc_type}"
            
            # Sample negatives
            negative_indices = random.sample(available_negatives, min(num_negatives, len(available_negatives)))
            
            for neg_idx in negative_indices:
                triplets.append({
                    'query_id': query_id,
                    'query': query_text,
                    'positive_id': positive_idx,
                    'positive': original_variant,
                    'negative_id': neg_idx,
                    'negative': acr_variants_list[neg_idx]
                })
    
    with open(output_dir / filename, 'w') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet) + '\n')
    
    return len(triplets)

# Generate training triplets
triplets_size = generate_triplets(train_cases, train_variant_mapping, train_acr, 'train_triplets.jsonl')

# Write metadata
metadata = {
    'description': 'ColBERT training data with CORRECTED ground truth (original ACR variants, not procedures)',
    'data_split': {
        'train_procedures': len(train_procedures),
        'val_procedures': len(val_procedures), 
        'test_procedures': len(test_procedures)
    },
    'corpus_sizes': {
        'train': train_corpus_size,
        'val': val_corpus_size,
        'test': test_corpus_size
    },
    'query_sizes': {
        'train': train_queries_size,
        'val': val_queries_size,
        'test': test_queries_size
    },
    'triplets_size': triplets_size,
    'ground_truth_type': 'original_acr_variant',  # CORRECTED!
    'task_description': 'Given patient description, retrieve the most relevant ACR appropriateness criteria variant'
}

with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n=== CORRECTED ColBERT Split Summary ===")
print(f"Train: {train_corpus_size} ACR variants, {train_queries_size} queries, {triplets_size} triplets")
print(f"Val: {val_corpus_size} ACR variants, {val_queries_size} queries")
print(f"Test: {test_corpus_size} ACR variants, {test_queries_size} queries")
print(f"\nGround truth: ORIGINAL ACR VARIANTS (corrected!)")
print(f"Task: Patient description → Most relevant ACR clinical scenario")
print(f"Files saved to: {output_dir}/") 