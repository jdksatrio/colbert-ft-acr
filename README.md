# ColBERT Fine-tuning for ACR Appropriateness Criteria

## Overview
Fine-tuning ColBERT on ACR (American College of Radiology) appropriateness criteria to improve retrieval accuracy for patient case matching.

## Dataset
- **ACR variants**: 12,206 appropriateness criteria from `dataset/acr.csv`
- **Patient cases**: 1,105 synthetic cases from `patient_cases_2.csv`
- **Training data**: Generated in `colbert_split/` with proper train/val/test splits

## Files

### Core Scripts
- `train_colbert_ragatouille.py` - Main training script using RAGatouille
- `trainsplit_acr_colbert.py` - Data preparation and splitting script

### CPU Training Fix
- `apply_colbert_cpu_fix.sh` - Automated fix for CPU-only training
- `COLBERT_CPU_FIX.md` - Technical documentation of the fix

### Data
- `colbert_split/` - Training data (8,961 corpus, 2,385 queries, 11,925 triplets)
- `.ragatouille/` - Trained model checkpoint (428MB)

## Usage

### Data Preparation
```bash
python trainsplit_acr_colbert.py
```

### Apply CPU Fix (required for Apple Silicon/CPU training)
```bash
conda activate colbert
./apply_colbert_cpu_fix.sh
```

### Training
```bash
python train_colbert_ragatouille.py
```

## Technical Notes
- Fixes DistributedDataParallel incompatibility with CPU training
- Uses RAGatouille wrapper around ColBERT v2.0
- Training completed successfully on CPU with 5,961 steps
- Model saved as 'ACR_ColBERT_FineTuned' in `.ragatouille/colbert/`

## Environment
- Python 3.9
- conda environment named 'colbert'
- RAGatouille with ColBERT backend
- CPU training (Apple Silicon compatible) 