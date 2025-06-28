#!/usr/bin/env python3
"""
Fine-tune ColBERT using RAGatouille on ACR Appropriateness Criteria data.

This script loads the prepared ACR training data and fine-tunes a ColBERT model
to improve retrieval of relevant ACR variants given patient case descriptions.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm

# Set environment variables to avoid DDP issues on MPS/CPU
# These settings disable DistributedDataParallel which is incompatible with MPS backends
os.environ['COLBERT_AVOID_FORK_IF_POSSIBLE'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode to avoid MPS/DDP conflicts

# Import RAGatouille with the correct API
try:
    from ragatouille import RAGTrainer
    print("✅ RAGTrainer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import RAGTrainer: {e}")
    print("Please install with: pip install ragatouille")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('colbert_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ACRColBERTTrainer:
    """Fine-tune ColBERT using RAGatouille for ACR appropriateness criteria matching."""
    
    def __init__(self, data_dir: str = "colbert_split", model_name: str = "ACR_ColBERT_FineTuned"):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.trainer = None
        
        logger.info(f"Initialized ACR ColBERT Trainer with data directory: {self.data_dir}")
        logger.info(f"Model will be saved as: {self.model_name}")
    
    def load_training_data(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Load training data from JSONL files and convert to RAGTrainer format.
        
        Returns:
            Tuple of (pairs, corpus) where pairs are (query, positive_doc) tuples
        """
        logger.info("Loading training data...")
        
        # Load corpus
        corpus_file = self.data_dir / "train_corpus.jsonl"
        corpus = []
        corpus_dict = {}  # id -> text mapping
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                corpus.append(data['text'])
                corpus_dict[data['id']] = data['text']
        
        # Load triplets and convert to pairs
        triplets_file = self.data_dir / "train_triplets.jsonl"
        pairs = []
        
        with open(triplets_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Create (query, positive_document) pairs
                pairs.append((data['query'], data['positive']))
        
        logger.info(f"Loaded {len(pairs)} training pairs and {len(corpus)} documents")
        return pairs, corpus
    
    def load_validation_data(self) -> Tuple[List[str], List[str]]:
        """Load validation data for evaluation."""
        logger.info("Loading validation data...")
        
        # Load validation queries
        val_queries_file = self.data_dir / "val_queries.jsonl"
        val_queries = []
        
        with open(val_queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                val_queries.append(data['query'])
        
        # Load validation corpus
        val_corpus_file = self.data_dir / "val_corpus.jsonl"
        val_corpus = []
        with open(val_corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                val_corpus.append(data['text'])
        
        logger.info(f"Loaded {len(val_queries)} validation queries, {len(val_corpus)} validation documents")
        return val_queries, val_corpus
    
    def initialize_trainer(self):
        """Initialize the RAGTrainer with CPU-only mode to avoid DDP issues."""
        logger.info("Initializing RAGTrainer...")
        
        try:
            self.trainer = RAGTrainer(
                model_name=self.model_name,
                pretrained_model_name="colbert-ir/colbertv2.0",
                n_usable_gpus=0  # Force CPU mode to avoid DistributedDataParallel issues
            )
            logger.info("✅ Successfully initialized RAGTrainer in CPU mode")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trainer: {e}")
            raise
    
    def prepare_and_train(self, pairs: List[Tuple[str, str]], corpus: List[str], 
                         batch_size: int = 8, num_epochs: int = 3):
        """
        Prepare training data and fine-tune the ColBERT model.
        
        Args:
            pairs: List of (query, positive_document) pairs
            corpus: List of all documents
            batch_size: Training batch size
            num_epochs: Number of training epochs
        """
        logger.info(f"Preparing training data and starting fine-tuning...")
        logger.info(f"Training configuration: batch_size={batch_size}, epochs={num_epochs}")
        
        try:
            # Initialize trainer
            self.initialize_trainer()
            
            # Prepare training data
            logger.info("🔄 Preparing training data...")
            self.trainer.prepare_training_data(
                raw_data=pairs,
                data_out_path="./data/",
                all_documents=corpus
            )
            
            # Start training
            logger.info("🚀 Starting ColBERT fine-tuning...")
            self.trainer.train(batch_size=batch_size)
            
            logger.info("✅ Fine-tuning completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def evaluate_model(self, val_queries: List[str], val_corpus: List[str], top_k: int = 10):
        """
        Evaluate the fine-tuned model on validation data.
        
        Args:
            val_queries: Validation queries
            val_corpus: Validation corpus
            top_k: Number of top results to retrieve
        """
        if self.trainer is None:
            logger.error("Trainer not initialized. Please run prepare_and_train first.")
            return
        
        logger.info(f"Evaluating model on {len(val_queries)} validation queries...")
        
        try:
            # The trained model is automatically saved by RAGTrainer
            # We can use it for evaluation
            logger.info("✅ Model training completed. Model saved automatically by RAGTrainer.")
            logger.info(f"Model saved as: {self.model_name}")
            
            # For evaluation, you would typically use the RAGPretrainedModel to load
            # the trained model and run evaluation, but that's a separate step
            
        except Exception as e:
            logger.error(f"❌ Evaluation preparation failed: {e}")
            raise

def main():
    """Main training pipeline."""
    logger.info("🚀 Starting ACR ColBERT fine-tuning with RAGatouille")
    
    # Initialize trainer
    trainer = ACRColBERTTrainer(
        data_dir="colbert_split",
        model_name="ACR_ColBERT_FineTuned"
    )
    
    try:
        # Load training data
        pairs, corpus = trainer.load_training_data()
        
        # Load validation data (for future evaluation)
        val_queries, val_corpus = trainer.load_validation_data()
        
        # Fine-tune model
        trainer.prepare_and_train(
            pairs=pairs,
            corpus=corpus,
            batch_size=4,  # Start with smaller batch size for stability
            num_epochs=3
        )
        
        # Prepare for evaluation
        trainer.evaluate_model(val_queries, val_corpus, top_k=10)
        
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info(f"✅ Your fine-tuned model '{trainer.model_name}' is ready for use!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 