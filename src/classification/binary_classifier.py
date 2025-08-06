"""
FIXED Binary Classification Model for CUAD Dataset
Predicts whether a context contains an answer to a given question
IMPROVEMENTS: Less conservative, better data augmentation, custom thresholds
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
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
    confusion_matrix,
)
import logging
import sys
import os
from typing import Dict, List, Tuple, Optional
import warnings
import random

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.settings import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Custom trainer with class weights for imbalanced data"""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class CUADBinaryDataset(Dataset):
    """Dataset class for CUAD binary classification"""

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


class CUADBinaryClassifier:
    """FIXED Binary classifier for CUAD dataset - Less Conservative"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.model_config = self.config.classification_model_config
        self.training_config = self.config.training_config

        # Setup paths
        self.data_dir = self.config.paths.processed_data
        self.model_dir = self.config.paths.saved_models / "binary_classifier"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None

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
        """Load processed binary classification data"""
        logger.info("Loading binary classification data...")

        splits = {}
        files_to_load = [
            ("train_binary_classification.json", "train"),
            ("test_binary_classification.json", "test"),
            ("main_binary_classification.json", "main"),
        ]

        for filename, split_name in files_to_load:
            file_path = self.data_dir / filename

            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                splits[split_name] = data
                logger.info(f"Loaded {len(data)} samples from {split_name}")
            else:
                logger.warning(f"File not found: {filename}")
                splits[split_name] = []

        return splits.get("train", []), splits.get("test", []), splits.get("main", [])

    def prepare_data(
        self, data: List[Dict], augment_positive: bool = True
    ) -> Tuple[List[str], List[int]]:
        """Prepare texts and labels with optional positive class augmentation"""
        texts = []
        labels = []

        for item in data:
            text = item.get("text", "")
            label = item.get("label", 0)

            # Smart context truncation for long texts
            text = self._truncate_intelligently(text)

            texts.append(text)
            labels.append(label)

            # FIXED: Augment positive samples to reduce classifier conservatism
            if augment_positive and label == 1:
                # Add 1-2 variations of positive samples
                for _ in range(random.randint(1, 2)):
                    augmented_text = self._augment_text(text)
                    texts.append(augmented_text)
                    labels.append(label)

        return texts, labels

    def _augment_text(self, text: str) -> str:
        """Simple text augmentation for positive samples"""
        if "[SEP]" in text:
            question, context = text.split("[SEP]", 1)
            question = question.strip()
            context = context.strip()

            # Simple augmentations
            augmentations = [
                f"{question} [SEP] {context}",  # Original
                f"{question.rstrip('?').strip()}? [SEP] {context}",  # Ensure question mark
                f"{question} [SEP] {context.strip('.')}.",  # Ensure period
            ]

            return random.choice(augmentations)

        return text

    def _truncate_intelligently(self, text: str) -> str:
        """Intelligently truncate text to fit model constraints"""
        # Split on [SEP] to separate question and context
        if "[SEP]" in text:
            question, context = text.split("[SEP]", 1)
            question = question.strip()
            context = context.strip()
        else:
            # Fallback: treat entire text as context
            question = ""
            context = text

        # Estimate token count (rough: 1 token ≈ 0.75 words)
        max_context_tokens = (
            self.model_config.max_length - len(question.split()) - 10
        )  # Buffer for special tokens
        max_context_words = int(max_context_tokens * 0.75)

        # Truncate context if too long
        context_words = context.split()
        if len(context_words) > max_context_words:
            # Try to keep the beginning (often contains key info)
            context = " ".join(context_words[:max_context_words])

        # Reconstruct text
        if question:
            return f"{question} [SEP] {context}"
        else:
            return context

    def initialize_model(self):
        """Initialize tokenizer and model with LegalBERT"""
        # Use config for model selection
        model_name = self.config.config["models"]["classification"]["legal_model"]
        fallback_model = self.config.config["models"]["classification"]["base_model"]

        logger.info(f"Initializing LegalBERT model: {model_name}")
        logger.info(
            "LegalBERT is specifically trained on legal documents - expect better performance!"
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
                num_labels=2,  # Binary classification
                problem_type="single_label_classification",
            )
            logger.info("LegalBERT model loaded successfully")

            # Resize token embeddings if needed
            self.model.resize_token_embeddings(len(self.tokenizer))

            logger.info(
                "LegalBERT initialization complete - ready for legal contract classification!"
            )

        except Exception as e:
            logger.error(f"Failed to load LegalBERT: {e}")
            logger.info("Falling back to standard BERT...")

            # Fallback to config model if LegalBERT fails
            logger.info(f"Loading fallback model: {fallback_model}")

            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForSequenceClassification.from_pretrained(
                fallback_model, num_labels=2, problem_type="single_label_classification"
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info("Fallback model loaded successfully")

    def create_datasets(
        self, train_data: List[Dict], test_data: List[Dict]
    ) -> Tuple[CUADBinaryDataset, CUADBinaryDataset]:
        """Create PyTorch datasets"""
        logger.info("Creating datasets...")

        # Prepare data with augmentation for training set
        train_texts, train_labels = self.prepare_data(train_data, augment_positive=True)
        test_texts, test_labels = self.prepare_data(
            test_data, augment_positive=False
        )  # No augmentation for test

        # Create datasets
        train_dataset = CUADBinaryDataset(
            train_texts, train_labels, self.tokenizer, self.model_config.max_length
        )

        test_dataset = CUADBinaryDataset(
            test_texts, test_labels, self.tokenizer, self.model_config.max_length
        )

        logger.info(
            f"Created train dataset: {len(train_dataset)} samples (augmented from {len(train_data)})"
        )
        logger.info(f"Created test dataset: {len(test_dataset)} samples")

        # Print class distribution
        train_pos = sum(train_labels)
        test_pos = sum(test_labels)

        logger.info(
            f"Train positive samples: {train_pos}/{len(train_labels)} ({train_pos/len(train_labels):.1%})"
        )
        logger.info(
            f"Test positive samples: {test_pos}/{len(test_labels)} ({test_pos/len(test_labels):.1%})"
        )

        return train_dataset, test_dataset

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train_model(
        self,
        train_dataset: CUADBinaryDataset,
        test_dataset: CUADBinaryDataset,
        train_data: List[Dict],
    ):
        """Train the binary classification model"""
        logger.info("Starting model training...")

        # Get config values
        binary_config = self.config.config["models"]["classification"]["binary"]
        training_config = self.config.config["training"]

        # Ensure proper data types (handle potential string conversion issues)
        learning_rate = float(binary_config["learning_rate"])
        weight_decay = float(binary_config["weight_decay"])
        conservative_factor = float(binary_config["conservative_factor"])

        # Setup training arguments from config
        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            num_train_epochs=int(binary_config["num_epochs"]),
            per_device_train_batch_size=int(binary_config["batch_size"]),
            per_device_eval_batch_size=int(binary_config["eval_batch_size"]),
            learning_rate=learning_rate,
            warmup_steps=int(binary_config["warmup_steps"]),
            weight_decay=weight_decay,
            logging_dir=str(self.config.paths.logs),
            logging_steps=int(training_config["logging_steps"]),
            eval_strategy="steps",
            eval_steps=int(training_config["eval_steps"]),
            save_strategy="steps",
            save_steps=int(training_config["save_steps"]),
            load_best_model_at_end=bool(training_config["save_best_model"]),
            metric_for_best_model=str(training_config["metric_for_best_model_binary"]),
            greater_is_better=bool(training_config["greater_is_better"]),
            save_total_limit=int(training_config["save_total_limit"]),
            report_to=None,
            dataloader_pin_memory=bool(training_config["dataloader_pin_memory"]),
            remove_unused_columns=bool(training_config["remove_unused_columns"]),
        )

        # FIXED: Calculate less conservative class weights
        train_texts, train_labels = self.prepare_data(train_data, augment_positive=True)
        pos_count = train_labels.count(1)
        neg_count = train_labels.count(0)

        # Make classifier less conservative - reduce positive class penalty
        if pos_count > 0:
            base_pos_weight = neg_count / pos_count
            # FIXED: Use conservative factor from config
            conservative_factor = conservative_factor
            pos_weight = base_pos_weight * conservative_factor
        else:
            pos_weight = 1.0

        # Move class weights to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)

        logger.info(
            f"FIXED Class weights (less conservative): {class_weights.tolist()}"
        )
        logger.info(f"Positive class weight reduced by factor: {conservative_factor}")

        # Initialize trainer with class weights
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            class_weights=class_weights,
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

            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        return trainer

    def predict_with_threshold(
        self, texts: List[str], threshold: float = None
    ) -> List[int]:
        """FIXED: Predict with custom threshold (from config or default 0.3 for less conservatism)"""
        if threshold is None:
            threshold = self.config.config["models"]["classification"]["binary"][
                "recommended_threshold"
            ]

        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        predictions = []
        self.model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.model_config.max_length,
                    return_tensors="pt",
                ).to(device)

                # Get prediction
                outputs = self.model(**encoding)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Use custom threshold instead of argmax
                has_answer_prob = probs[0][1].item()  # Probability of "has answer"
                prediction = 1 if has_answer_prob > threshold else 0
                predictions.append(prediction)

        return predictions

    def evaluate_detailed(self, trainer, test_dataset: CUADBinaryDataset):
        """Perform detailed evaluation with multiple thresholds"""
        logger.info("Performing detailed evaluation...")

        try:
            # Get predictions
            predictions = trainer.predict(test_dataset)
            y_true = predictions.label_ids

            # Get thresholds from config
            threshold_config = self.config.config["evaluation"]["threshold_analysis"]
            thresholds = threshold_config["thresholds"]
            recommended_threshold = threshold_config["recommended_threshold"]

            threshold_results = {}

            for threshold in thresholds:
                # Apply threshold
                y_pred = (predictions.predictions[:, 1] > threshold).astype(int)

                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, average=None
                )
                conf_matrix = confusion_matrix(y_true, y_pred)

                # Handle case where only one class is predicted
                if len(precision) == 1:
                    precision = np.append(precision, 0.0)
                    recall = np.append(recall, 0.0)
                    f1 = np.append(f1, 0.0)
                    support = np.append(support, 0)

                threshold_results[threshold] = {
                    "accuracy": float(accuracy),
                    "precision_class_1": (
                        float(precision[1]) if len(precision) > 1 else 0.0
                    ),
                    "recall_class_1": float(recall[1]) if len(recall) > 1 else 0.0,
                    "f1_class_1": float(f1[1]) if len(f1) > 1 else 0.0,
                    "confusion_matrix": conf_matrix.tolist(),
                }

            # Create detailed results
            detailed_results = {
                "threshold_analysis": threshold_results,
                "class_names": ["No Answer", "Has Answer"],
                "recommended_threshold": recommended_threshold,
            }

            # Print results
            print(f"\nFIXED BINARY CLASSIFIER EVALUATION:")
            print(f"  Multiple Threshold Analysis:")
            for threshold, metrics in threshold_results.items():
                print(
                    f"    Threshold {threshold}: Precision={metrics['precision_class_1']:.3f}, "
                    f"Recall={metrics['recall_class_1']:.3f}, F1={metrics['f1_class_1']:.3f}"
                )

            print(
                f"\nRECOMMENDED: Use threshold={recommended_threshold} for less conservative predictions"
            )

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
                "batch_size": self.config.config["models"]["classification"]["binary"][
                    "batch_size"
                ],
                "learning_rate": self.config.config["models"]["classification"][
                    "binary"
                ]["learning_rate"],
                "num_epochs": self.config.config["models"]["classification"]["binary"][
                    "num_epochs"
                ],
                "model_type": "LegalBERT - Legal Domain Specialized",
            },
            "improvements": {
                "legal_bert_model": True,
                "less_conservative_class_weights": True,
                "positive_class_augmentation": True,
                "custom_threshold_support": True,
                "recall_optimized": True,
                "legal_domain_knowledge": True,
            },
            "train_results": self.train_results,
            "eval_results": self.eval_results,
        }

        summary_file = self.model_dir / "training_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {summary_file}")


def main():
    """Main training pipeline"""
    print("FIXED CUAD Binary Classification Training with LegalBERT")
    print("Using LegalBERT: Specialized for legal document understanding")
    print("IMPROVEMENTS: Less conservative + Legal domain knowledge")
    print("=" * 70)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Initialize classifier
        classifier = CUADBinaryClassifier()

        # Load data
        train_data, test_data, main_data = classifier.load_data()

        if not train_data or not test_data:
            print("Insufficient data for training. Need both train and test sets.")
            return

        # Initialize model with LegalBERT
        print("\nLoading LegalBERT model...")
        classifier.initialize_model()

        # Create datasets
        train_dataset, test_dataset = classifier.create_datasets(train_data, test_data)

        # Train model
        print("\nTraining LegalBERT classifier...")
        trainer = classifier.train_model(train_dataset, test_dataset, train_data)

        # Detailed evaluation with multiple thresholds
        detailed_results = classifier.evaluate_detailed(trainer, test_dataset)

        # Save summary
        classifier.save_training_summary()

        print("\nLegalBERT Binary classification training complete!")
        print(f"Model saved to: {classifier.model_dir}")

        if "error" not in detailed_results:
            # Get recommended threshold results
            rec_threshold = detailed_results.get("recommended_threshold", 0.3)
            threshold_metrics = detailed_results["threshold_analysis"].get(
                rec_threshold, {}
            )
            if threshold_metrics:
                print(f"\nLegalBERT + Threshold {rec_threshold} Performance:")
                print(
                    f"   Precision: {threshold_metrics.get('precision_class_1', 0):.3f}"
                )
                print(f"   Recall: {threshold_metrics.get('recall_class_1', 0):.3f}")
                print(f"   F1: {threshold_metrics.get('f1_class_1', 0):.3f}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
