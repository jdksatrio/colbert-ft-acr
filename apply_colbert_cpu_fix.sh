#!/bin/bash

# ColBERT CPU Training Fix Application Script
# This script applies the necessary patches to enable ColBERT training on CPU-only systems

set -e  # Exit on any error

echo "🔧 ColBERT CPU Training Fix Application Script"
echo "=============================================="

# Check if we're in a conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Error: No conda environment detected. Please activate your ColBERT environment first."
    echo "   Run: conda activate colbert"
    exit 1
fi

echo "Conda environment detected: ${CONDA_DEFAULT_ENV}"

# Find the ColBERT installation directory
COLBERT_DIR=$(python -c "import colbert; import os; print(os.path.dirname(colbert.__file__))" 2>/dev/null)

if [[ -z "${COLBERT_DIR}" ]]; then
    echo "Error: ColBERT package not found. Please install it first."
    echo "   Run: pip install ragatouille"
    exit 1
fi

echo "ColBERT installation found: ${COLBERT_DIR}"

# Backup original files
BACKUP_DIR="./colbert_original_backups_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"

echo "Creating backups in: ${BACKUP_DIR}"

# Backup training.py
cp "${COLBERT_DIR}/training/training.py" "${BACKUP_DIR}/training.py.backup"
echo "-Backed up training.py"

# Backup launcher.py
cp "${COLBERT_DIR}/infra/launcher.py" "${BACKUP_DIR}/launcher.py.backup"
echo "-Backed up launcher.py"

# Apply training.py fix
echo "-Applying training.py fix..."

# Create the patched training.py content
cat > "${BACKUP_DIR}/training_patch.py" << 'EOF'
import time
import torch
import random
import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints



def train(config: ColBERTConfig, triples, queries=None, collection=None):
    config.checkpoint = config.checkpoint or 'bert-base-uncased'

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    # Only use DistributedDataParallel if we have multiple ranks or GPU training
    if config.nranks > 1 or (torch.cuda.is_available() and torch.cuda.device_count() > 0 and DEVICE.type == 'cuda'):
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                            output_device=config.rank,
                                                            find_unused_parameters=True)
    # For CPU-only single rank training, don't use DistributedDataParallel

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                if config.use_ib_negatives:
                    if config.rank < 1:
                        print('\t\t\t\t', loss.item(), ib_loss.item())

                    loss += ib_loss

                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            print_message(batch_idx, train_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.



def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        # Try accessing through .module (for DistributedDataParallel)
        try:
            set_bert_grad(colbert.module, value)
        except AttributeError:
            # If both fail, the model might not have a bert attribute
            print(f"Warning: Could not set bert grad to {value} - model structure may be different")
EOF

# Apply the training.py patch
cp "${BACKUP_DIR}/training_patch.py" "${COLBERT_DIR}/training/training.py"
echo "-Applied training.py fix"

# Apply launcher.py fixes
echo "-Applying launcher.py fixes..."

# Use Python to apply the launcher.py fixes
python << EOF
import re

# Read the original launcher.py
with open("${COLBERT_DIR}/infra/launcher.py", 'r') as f:
    content = f.read()

# Apply fix 1: Add CPU-only check in launch method
launch_pattern = r'(def launch\(self, custom_config, \*args\):\s+assert isinstance\(custom_config, BaseConfig\)\s+assert isinstance\(custom_config, RunSettings\)\s+)'
launch_replacement = r'''\1
        # If we have 0 ranks or forced CPU-only mode, use launch_without_fork
        if (self.nranks == 0 or 
            (self.nranks == 1 and (custom_config.avoid_fork_if_possible or self.run_config.avoid_fork_if_possible))):
            return self.launch_without_fork(custom_config, *args)
        
'''
content = re.sub(launch_pattern, launch_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Apply fix 2: Handle empty return values
return_pattern = r'(if not self\.return_all:\s+)(return_values = return_values\[0\])'
return_replacement = r'''\1# Handle the case where no processes were created (e.g., CPU-only mode with 0 GPUs)
            if len(return_values) == 0:
                return None
            \2'''
content = re.sub(return_pattern, return_replacement, content, flags=re.MULTILINE)

# Apply fix 3: Update launch_without_fork for nranks=0
launch_without_fork_pattern = r'(def launch_without_fork\(self, custom_config, \*args\):\s+assert isinstance\(custom_config, BaseConfig\)\s+assert isinstance\(custom_config, RunSettings\)\s+)(assert self\.nranks == 1\s+assert \(custom_config\.avoid_fork_if_possible or self\.run_config\.avoid_fork_if_possible\)\s+)(new_config = type\(custom_config\)\.from_existing\(custom_config, self\.run_config, RunConfig\(rank=0\)\))'
launch_without_fork_replacement = r'''\1# Allow nranks == 0 for CPU-only mode
        assert self.nranks <= 1, f"launch_without_fork requires nranks <= 1, got {self.nranks}"
        
        # If nranks is 0, set it to 1 for the single process
        effective_nranks = max(1, self.nranks)
        
        \3'''
content = re.sub(launch_without_fork_pattern, launch_without_fork_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Apply fix 4: Update run_process_without_mp for CPU handling
run_process_pattern = r'(def run_process_without_mp\(callee, config, \*args\):\s+set_seed\(12345\)\s+)(os\.environ\["CUDA_VISIBLE_DEVICES"\] = \',\'\.join\(map\(str, config\.gpus_\[:config\.nranks\]\)\)\s+)(with Run\(\)\.context\(config, inherit_config=False\):\s+return_val = callee\(config, \*args\)\s+)(torch\.cuda\.empty_cache\(\)\s+return return_val)'
run_process_replacement = r'''\1
    # Handle case where there are no GPUs available
    if hasattr(config, 'gpus_') and len(config.gpus_) > 0:
        \2
    else:
        # Force CPU-only mode
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

    \3# Only empty CUDA cache if CUDA is available
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()
        return return_val'''
content = re.sub(run_process_pattern, run_process_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write the patched content
with open("${COLBERT_DIR}/infra/launcher.py", 'w') as f:
    f.write(content)

print("Applied launcher.py fixes")
EOF

echo " Applied launcher.py fixes"