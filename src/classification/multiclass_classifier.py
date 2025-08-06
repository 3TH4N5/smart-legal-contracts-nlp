"""
Multi-Class Clause Classification Model for CUAD Dataset
Classifies contract text into 41 different clause types using LegalBERT
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
import logging
import sys
import os
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.settings import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CUADMultiClassDataset(Dataset):
    """Dataset class for CUAD multi-class classification"""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize with truncation and padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class CUADMultiClassClassifier:
    """FIXED Multi-class classifier for CUAD clause types with LegalBERT"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.model_config = self.config.classification_model_config
        self.training_config = self.config.training_config

        # Setup paths
        self.data_dir = self.config.paths.processed_data
        self.model_dir = self.config.paths.saved_models / "multiclass_classifier"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.clause_types = []

        # Results storage
        self.train_results = {}
        self.eval_results = {}

        # Setup logging from config
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging from config"""
        log_config = self.config.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        format_str = log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(level=level, format=format_str)

    def load_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load processed multi-class classification data"""
        logger.info("Loading multi-class classification data...")

        splits = {}
        files_to_load = [
            ("train_multiclass_classification.json", "train"),
            ("test_multiclass_classification.json", "test"),
            ("main_multiclass_classification.json", "main"),
        ]

        for filename, split_name in files_to_load:
            file_path = self.data_dir / filename

            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Filter out 'unknown' clause types and samples without answers
                filtered_data = []
                for item in data:
                    if item.get("label") != "unknown" and item.get(
                        "has_answer", True
                    ):  # Only include samples with answers
                        filtered_data.append(item)

                splits[split_name] = filtered_data
                logger.info(
                    f"Loaded {len(filtered_data)} samples from {split_name} (filtered from {len(data)})"
                )
            else:
                logger.warning(f"File not found: {filename}")
                splits[split_name] = []

        return splits.get("train", []), splits.get("test", []), splits.get("main", [])

    def prepare_data(self, data: List[Dict]) -> Tuple[List[str], List[str], List[int]]:
        """Prepare texts and labels from raw data"""
        texts = []
        labels = []
        raw_labels = []

        for item in data:
            text = item.get("text", "")
            label = item.get("label", "unknown")

            if label != "unknown":
                # Smart context truncation
                text = self._truncate_intelligently(text)

                texts.append(text)
                raw_labels.append(label)

        if raw_labels:
            # Encode labels to integers
            if not hasattr(self.label_encoder, "classes_"):
                # Fit label encoder on first use
                labels = self.label_encoder.fit_transform(raw_labels)
                self.clause_types = self.label_encoder.classes_.tolist()
                logger.info(f"Found {len(self.clause_types)} clause types")
            else:
                # Transform using existing encoder
                labels = self.label_encoder.transform(raw_labels)

        return texts, raw_labels, labels.tolist()

    def _truncate_intelligently(self, text: str) -> str:
        """Intelligently truncate text to fit model constraints"""
        # Split on [SEP] to separate question and context
        if "[SEP]" in text:
            question, context = text.split("[SEP]", 1)
            question = question.strip()
            context = context.strip()
        else:
            question = ""
            context = text

        # Estimate token count
        max_context_tokens = self.model_config.max_length - len(question.split()) - 10
        max_context_words = int(max_context_tokens * 0.75)

        # Truncate context if too long
        context_words = context.split()
        if len(context_words) > max_context_words:
            context = " ".join(context_words[:max_context_words])

        # Reconstruct text
        if question:
            return f"{question} [SEP] {context}"
        else:
            return context

    def initialize_model(self, num_labels: int):
        """Initialize tokenizer and model with LegalBERT"""
        # Use config for model selection
        model_name = self.config.config["models"]["classification"]["legal_model"]
        fallback_model = self.config.config["models"]["classification"]["base_model"]

        logger.info(f"Initializing LegalBERT MultiClass model: {model_name}")
        logger.info(f"Number of clause types: {num_labels}")
        logger.info(
            "LegalBERT should excel at distinguishing between legal clause types!"
        )

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("LegalBERT tokenizer loaded successfully")

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="single_label_classification",
            )
            logger.info("LegalBERT multiclass model loaded successfully")

            # Resize token embeddings if needed
            self.model.resize_token_embeddings(len(self.tokenizer))

            logger.info("LegalBERT MultiClass initialization complete!")

        except Exception as e:
            logger.error(f"Failed to load LegalBERT: {e}")
            logger.info("Falling back to standard BERT...")

            # Fallback to config model if LegalBERT fails
            logger.info(f"Loading fallback model: {fallback_model}")

            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForSequenceClassification.from_pretrained(
                fallback_model,
                num_labels=num_labels,
                problem_type="single_label_classification",
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info("Fallback model loaded successfully")

    def create_datasets(
        self, train_data: List[Dict], test_data: List[Dict]
    ) -> Tuple[CUADMultiClassDataset, CUADMultiClassDataset]:
        """Create PyTorch datasets"""
        logger.info("Creating datasets...")

        # Prepare data
        train_texts, train_raw_labels, train_labels = self.prepare_data(train_data)
        test_texts, test_raw_labels, test_labels = self.prepare_data(test_data)

        # Create datasets
        train_dataset = CUADMultiClassDataset(
            train_texts, train_labels, self.tokenizer, self.model_config.max_length
        )

        test_dataset = CUADMultiClassDataset(
            test_texts, test_labels, self.tokenizer, self.model_config.max_length
        )

        logger.info(f"Created train dataset: {len(train_dataset)} samples")
        logger.info(f"Created test dataset: {len(test_dataset)} samples")

        # Print class distribution
        from collections import Counter

        train_dist = Counter(train_raw_labels)
        test_dist = Counter(test_raw_labels)

        logger.info(f"Train - Top 5 clause types: {train_dist.most_common(5)}")
        logger.info(f"Test - Top 5 clause types: {test_dist.most_common(5)}")

        return train_dataset, test_dataset

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train_model(
        self, train_dataset: CUADMultiClassDataset, test_dataset: CUADMultiClassDataset
    ):
        """Train the multi-class classification model"""
        logger.info("Starting LegalBERT multiclass model training...")

        # Get config values
        multiclass_config = self.config.config["models"]["classification"]["multiclass"]
        training_config = self.config.config["training"]

        # Ensure proper data types (handle potential string conversion issues)
        learning_rate = float(multiclass_config["learning_rate"])
        weight_decay = float(multiclass_config["weight_decay"])

        # Setup training arguments from config
        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            num_train_epochs=int(multiclass_config["num_epochs"]),
            per_device_train_batch_size=int(multiclass_config["batch_size"]),
            per_device_eval_batch_size=int(multiclass_config["eval_batch_size"]),
            learning_rate=learning_rate,
            warmup_steps=int(multiclass_config["warmup_steps"]),
            weight_decay=weight_decay,
            logging_dir=str(self.config.paths.logs),
            logging_steps=int(training_config["logging_steps"]),
            eval_strategy="steps",
            eval_steps=int(training_config["eval_steps"]),
            save_strategy="steps",
            save_steps=int(training_config["save_steps"]),
            load_best_model_at_end=bool(training_config["save_best_model"]),
            metric_for_best_model=str(
                training_config["metric_for_best_model_multiclass"]
            ),
            greater_is_better=bool(training_config["greater_is_better"]),
            save_total_limit=int(training_config["save_total_limit"]),
            report_to=None,
            dataloader_pin_memory=bool(training_config["dataloader_pin_memory"]),
            remove_unused_columns=bool(training_config["remove_unused_columns"]),
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience
                )
            ],
        )

        # Train model
        try:
            train_result = trainer.train()

            # Save training results
            self.train_results = train_result.metrics

            # Evaluate model
            eval_result = trainer.evaluate()
            self.eval_results = eval_result

            # Save model and tokenizer
            trainer.save_model()
            self.tokenizer.save_pretrained(self.model_dir)

            # Save label encoder
            import pickle

            with open(self.model_dir / "label_encoder.pkl", "wb") as f:
                pickle.dump(self.label_encoder, f)

            # Save clause types
            with open(self.model_dir / "clause_types.json", "w") as f:
                json.dump(self.clause_types, f, indent=2)

            logger.info("LegalBERT multiclass training completed successfully")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        return trainer

    def evaluate_detailed(self, trainer, test_dataset: CUADMultiClassDataset):
        """Perform detailed evaluation"""
        logger.info("Performing detailed evaluation...")

        try:
            # Get predictions
            predictions = trainer.predict(test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids

            # Calculate detailed metrics
            accuracy = accuracy_score(y_true, y_pred)

            # Classification report
            target_names = [f"{i}:{name}" for i, name in enumerate(self.clause_types)]
            class_report = classification_report(
                y_true, y_pred, target_names=target_names, output_dict=True
            )

            # Create detailed results
            detailed_results = {
                "accuracy": float(accuracy),
                "classification_report": class_report,
                "clause_types": self.clause_types,
                "num_classes": len(self.clause_types),
                "model_type": "LegalBERT MultiClass",
            }

            # Print top-level results
            print(f"\nLEGALBERT MULTI-CLASS EVALUATION RESULTS:")
            print(f"  Overall Accuracy: {accuracy:.3f}")
            print(
                f"  Weighted Precision: {class_report['weighted avg']['precision']:.3f}"
            )
            print(f"  Weighted Recall: {class_report['weighted avg']['recall']:.3f}")
            print(f"  Weighted F1: {class_report['weighted avg']['f1-score']:.3f}")
            print(f"  Number of Classes: {len(self.clause_types)}")

            # Print best and worst performing classes
            class_f1s = [
                (name, class_report[f"{i}:{name}"]["f1-score"])
                for i, name in enumerate(self.clause_types)
                if f"{i}:{name}" in class_report
            ]

            if class_f1s:
                class_f1s_sorted = sorted(class_f1s, key=lambda x: x[1], reverse=True)
                print(f"\nTop 5 Performing Classes (LegalBERT):")
                for name, f1 in class_f1s_sorted[:5]:
                    print(f"    {name}: {f1:.3f}")

                print(f"\nBottom 5 Performing Classes (needs improvement):")
                for name, f1 in class_f1s_sorted[-5:]:
                    print(f"    {name}: {f1:.3f}")

            # Save detailed results
            results_file = self.model_dir / "evaluation_results.json"
            with open(results_file, "w") as f:
                json.dump(detailed_results, f, indent=2)

            return detailed_results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}

    def save_training_summary(self):
        """Save training summary"""
        summary = {
            "model_config": {
                "base_model": self.config.config["models"]["classification"][
                    "legal_model"
                ],
                "max_length": self.model_config.max_length,
                "batch_size": self.config.config["models"]["classification"][
                    "multiclass"
                ]["batch_size"],
                "learning_rate": self.config.config["models"]["classification"][
                    "multiclass"
                ]["learning_rate"],
                "num_epochs": self.config.config["models"]["classification"][
                    "multiclass"
                ]["num_epochs"],
                "model_type": "LegalBERT MultiClass - Legal Domain Specialized",
            },
            "improvements": {
                "legal_bert_model": True,
                "legal_domain_knowledge": True,
                "clause_type_specialization": True,
            },
            "train_results": self.train_results,
            "eval_results": self.eval_results,
            "clause_types": self.clause_types,
            "num_classes": len(self.clause_types),
        }

        summary_file = self.model_dir / "training_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {summary_file}")


def main():
    """Main training pipeline"""
    print("FIXED CUAD Multi-Class Classification Training with LegalBERT")
    print("Using LegalBERT: Specialized for legal clause type classification")
    print("=" * 70)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Initialize classifier
        classifier = CUADMultiClassClassifier()

        # Load data
        train_data, test_data, main_data = classifier.load_data()

        if not train_data or not test_data:
            print("Insufficient data for training. Need both train and test sets.")
            return

        # Prepare data to get number of classes
        train_texts, train_raw_labels, train_labels = classifier.prepare_data(
            train_data
        )
        num_labels = len(classifier.clause_types)

        if num_labels < 2:
            print("Insufficient clause types for multi-class classification.")
            return

        # Initialize model with LegalBERT
        print(f"\nLoading LegalBERT for {num_labels} clause types...")
        classifier.initialize_model(num_labels)

        # Create datasets
        train_dataset, test_dataset = classifier.create_datasets(train_data, test_data)

        # Train model
        print("\nTraining LegalBERT multiclass classifier...")
        trainer = classifier.train_model(train_dataset, test_dataset)

        # Detailed evaluation
        detailed_results = classifier.evaluate_detailed(trainer, test_dataset)

        # Save summary
        classifier.save_training_summary()

        print("\nLegalBERT Multi-class classification training complete!")
        print(f"Model saved to: {classifier.model_dir}")

        if "error" not in detailed_results:
            weighted_f1 = detailed_results["classification_report"]["weighted avg"][
                "f1-score"
            ]
            print(f"LegalBERT Weighted F1 Score: {weighted_f1:.3f}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
