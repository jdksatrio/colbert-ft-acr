# ColBERT CPU Training Fix

## Problem
ColBERT's default training code fails on CPU-only systems (like Apple Silicon Macs) with the error:
```
ValueError: DistributedDataParallel device_ids and output_device arguments only work with single-device/multiple-device GPU modules or CPU modules, but got device_ids [0], output_device 0, and module parameters {device(type='cpu')}.
```

## Root Cause
The issue occurs in the ColBERT training code which unconditionally wraps the model with `DistributedDataParallel`, even when running on CPU with a single process (`nranks=1`).

## Solution
Modify the system-installed ColBERT training code to conditionally use DistributedDataParallel only when appropriate.

## Files Modified

### 1. Launcher Fix
**File:** `/Users/satrio/anaconda3/envs/colbert/lib/python3.9/site-packages/colbert/infra/launcher.py`

**Changes:**
- Added logic to use `launch_without_fork` for CPU-only training
- Enhanced error handling for empty return values
- Improved GPU detection and CPU fallback

### 2. Training Fix (Main Fix)
**File:** `/Users/satrio/anaconda3/envs/colbert/lib/python3.9/site-packages/colbert/training/training.py`

**Original Code (lines 51-55):**
```python
colbert = colbert.to(DEVICE)
colbert.train()

colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                    output_device=config.rank,
                                                    find_unused_parameters=True)
```

**Fixed Code:**
```python
colbert = colbert.to(DEVICE)
colbert.train()

# Only use DistributedDataParallel if we have multiple ranks or GPU training
if config.nranks > 1 or (torch.cuda.is_available() and torch.cuda.device_count() > 0 and DEVICE.type == 'cuda'):
    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)
# For CPU-only single rank training, don't use DistributedDataParallel

## Environment Configuration
The training script also sets these environment variables for CPU-only training:
```python
os.environ['COLBERT_AVOID_FORK_IF_POSSIBLE'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode
```

And configures RAGTrainer with:
```python
self.trainer = RAGTrainer(
    model_name=self.model_name,
    pretrained_model_name="colbert-ir/colbertv2.0",
    n_usable_gpus=0  # Force CPU mode to avoid DDP issues
)
```

## Verification
After applying the fix, training should complete successfully with output like:
```
[Jun 28, 22:05:19] #> Done with all triples!
#> Saving a checkpoint to .ragatouille/colbert/...
✅ Fine-tuning completed successfully!
```

## Notes
- This fix is specifically for CPU-only training environments
- GPU training should continue to work normally with DistributedDataParallel
- The fix maintains backward compatibility for multi-GPU setups
- All changes are made to the system-installed ColBERT package, not local repository files

## Created
Date: 2025-06-28
Environment: macOS (Apple Silicon), Python 3.9, ColBERT via RAGatouille 