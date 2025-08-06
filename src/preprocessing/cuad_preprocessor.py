"""
CUAD Data Preprocessing Pipeline
"""

import json
import pandas as pd
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContractSample:
    """Represents a single contract sample with metadata"""

    contract_id: str
    title: str
    context: str
    question: str
    answers: List[Dict]
    clause_type: str
    has_answer: bool
    answer_text: Optional[str] = None
    answer_start: Optional[int] = None


class UpdatedCUADPreprocessor:
    """Updated preprocessing class that works with fixed config"""

    def __init__(self, config_path: str = "config/cuad_config.yaml"):
        # Load YAML configuration
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Setup paths
        self.raw_dir = Path(self.config["paths"]["data"]["raw"]) / "cuad"
        self.processed_dir = Path(self.config["paths"]["data"]["processed"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Get configurations
        self.clause_mappings = self.config["clause_mappings"]
        self.text_cleaning = self.config["text_cleaning"]
        self.clause_types = list(set(self.clause_mappings.values()))

        # Setup answer detection
        self._setup_answer_detection()

        logger.info(f"Loaded config with {len(self.clause_mappings)} clause mappings")
        logger.info(f"Min answer length: {self.min_answer_length}")

    def _load_config(self) -> Dict:
        """Load YAML configuration file"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise

    def _setup_answer_detection(self):
        """Setup answer detection using config values"""

        # Get non-answers from config
        config_non_answers = self.text_cleaning.get("non_answers", [])
        self.definitive_non_answers = set(
            answer.lower().strip() for answer in config_non_answers
        )

        # Add some obvious non-answers
        self.definitive_non_answers.update(
            {
                "not mentioned",
                "not provided",
                "not found",
                "not included",
                "does not specify",
                "is not mentioned",
                "not disclosed",
                "null",
                "",
            }
        )

        # Get min answer length from config
        self.min_answer_length = self.text_cleaning.get("min_answer_length", 2)

        # Patterns for definitive non-answers
        self.non_answer_patterns = [
            r"^no\.?$",
            r"^none\.?$",
            r"^not?\s+applicable\.?$",
            r"^n/?a\.?$",
            r"^not?\s+(mentioned|specified|provided|found|included)\.?$",
            r"^\s*$",  # Empty
            r"^\.+$",  # Just periods
            r"^-+$",  # Just dashes
        ]

        logger.info(f"Non-answers: {self.definitive_non_answers}")
        logger.info(f"Min answer length: {self.min_answer_length}")

    def load_data(self) -> Dict[str, List[ContractSample]]:
        """Load and parse CUAD JSON files"""
        logger.info("Loading CUAD data...")

        data_splits = {}

        # Load main files - map to expected names for classification
        files_to_load = [
            ("CUADv1.json", "main"),  # Will create main_*.json
            ("test.json", "test"),  # Will create test_*.json
            ("train_separate_questions.json", "train"),  # Will create train_*.json
        ]

        for filename, split_name in files_to_load:
            file_path = self.raw_dir / filename

            if file_path.exists():
                logger.info(f"Loading {filename}...")

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                samples = self._parse_json_data(data, split_name)
                data_splits[split_name] = samples

                logger.info(f"Loaded {len(samples)} samples from {filename}")

                # Show answer statistics
                total_with_answers = sum(1 for s in samples if s.has_answer)
                logger.info(
                    f"  Samples with valid answers: {total_with_answers} ({total_with_answers/len(samples)*100:.1f}%)"
                )

            else:
                logger.warning(f"File not found: {filename}")

        return data_splits

    def _parse_json_data(self, data: Dict, split_name: str) -> List[ContractSample]:
        """Parse JSON data with improved answer validation"""
        samples = []

        if "data" not in data:
            logger.error(f"Invalid data format for {split_name}")
            return samples

        answer_stats = {
            "total_processed": 0,
            "has_answers_list": 0,
            "valid_answers": 0,
            "filtered_non_answers": 0,
            "filtered_too_short": 0,
        }

        for contract_idx, contract in enumerate(data["data"]):
            contract_title = contract.get("title", f"Contract_{contract_idx}")

            if "paragraphs" not in contract:
                continue

            for para_idx, paragraph in enumerate(contract["paragraphs"]):
                context = paragraph.get("context", "")

                if "qas" not in paragraph:
                    continue

                for qa_idx, qa in enumerate(paragraph["qas"]):
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    qa_id = qa.get("id", f"{contract_idx}_{para_idx}_{qa_idx}")

                    answer_stats["total_processed"] += 1

                    # Extract clause type
                    clause_type = self._get_clause_type(question)

                    # Process answers
                    has_answer, answer_text, answer_start = self._process_answers(
                        answers
                    )

                    # Update stats
                    if len(answers) > 0:
                        answer_stats["has_answers_list"] += 1
                    if has_answer:
                        answer_stats["valid_answers"] += 1
                    elif len(answers) > 0:
                        first_answer = answers[0].get("text", "").strip()
                        if self._is_non_answer(first_answer):
                            answer_stats["filtered_non_answers"] += 1
                        elif len(first_answer) < self.min_answer_length:
                            answer_stats["filtered_too_short"] += 1

                    sample = ContractSample(
                        contract_id=qa_id,
                        title=contract_title,
                        context=context,
                        question=question,
                        answers=answers,
                        clause_type=clause_type,
                        has_answer=has_answer,
                        answer_text=answer_text,
                        answer_start=answer_start,
                    )

                    samples.append(sample)

        # Log statistics
        logger.info(f"Answer processing stats for {split_name}:")
        for key, value in answer_stats.items():
            percentage = (
                (value / answer_stats["total_processed"] * 100)
                if answer_stats["total_processed"] > 0
                else 0
            )
            logger.info(f"  {key}: {value} ({percentage:.1f}%)")

        return samples

    def _process_answers(
        self, answers: List[Dict]
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Process answers with improved validation"""

        if not answers:
            return False, None, None

        # Find the best answer
        for answer in answers:
            answer_text = answer.get("text", "").strip()
            answer_start = answer.get("answer_start", 0)

            if not answer_text:
                continue

            # Check if this is a non-answer
            if self._is_non_answer(answer_text):
                continue

            # Check minimum length
            if len(answer_text) < self.min_answer_length:
                continue

            # This is a valid answer
            return True, answer_text, answer_start

        return False, None, None

    def _is_non_answer(self, text: str) -> bool:
        """Check if text represents a non-answer"""

        if not text:
            return True

        text_lower = text.lower().strip()

        # Check exact matches
        if text_lower in self.definitive_non_answers:
            return True

        # Check patterns
        for pattern in self.non_answer_patterns:
            if re.match(pattern, text_lower):
                return True

        return False

    def _get_clause_type(self, question: str) -> str:
        """Extract clause type from question using config mappings"""
        question_lower = question.lower()

        # Try exact matches first
        for cuad_name, clause_type in self.clause_mappings.items():
            if cuad_name.lower() in question_lower:
                return clause_type

        # Try partial matches
        for cuad_name, clause_type in self.clause_mappings.items():
            cuad_words = cuad_name.lower().replace("-", " ").split()
            # Remove common words
            key_words = [
                w
                for w in cuad_words
                if w not in ["of", "to", "the", "and", "or", "for"]
            ]
            if key_words:
                matches = sum(1 for word in key_words if word in question_lower)
                if matches >= len(key_words) * 0.7:  # 70% match
                    return clause_type

        return "unknown"

    def _clean_text(self, text: str) -> str:
        """Clean text using config patterns"""
        if not text:
            return text

        cleaned_text = text

        # Apply removal patterns from config
        remove_patterns = self.text_cleaning.get("remove_patterns", [])
        for pattern in remove_patterns:
            try:
                cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue

        # Clean up whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return cleaned_text

    def _extract_base_contract_id(self, contract_id: str) -> str:
        """Extract base contract ID for grouping"""
        if "_" in contract_id:
            parts = contract_id.split("_")
            if len(parts) >= 3:
                return parts[0]  # Take first part as base contract
            elif len(parts) >= 2:
                return "_".join(parts[:2])
        return contract_id

    def analyze_dataset(self, data_splits: Dict[str, List[ContractSample]]) -> Dict:
        """Analyze dataset with focus on contract diversity"""
        logger.info("Analyzing dataset...")

        analysis = {}

        for split_name, samples in data_splits.items():
            logger.info(f"Analyzing {split_name} split...")

            # Basic statistics
            total_samples = len(samples)
            samples_with_answers = sum(1 for s in samples if s.has_answer)

            # Clause type distribution
            clause_counts = Counter(s.clause_type for s in samples)
            answered_clause_counts = Counter(
                s.clause_type for s in samples if s.has_answer
            )

            # Contract-level analysis
            contract_analysis = self._analyze_contract_diversity(samples)

            split_analysis = {
                "total_samples": total_samples,
                "samples_with_answers": samples_with_answers,
                "answer_ratio": (
                    samples_with_answers / total_samples if total_samples > 0 else 0
                ),
                "clause_type_distribution": dict(clause_counts.most_common(15)),
                "answered_clause_distribution": dict(
                    answered_clause_counts.most_common(15)
                ),
                "contract_diversity": contract_analysis,
            }

            analysis[split_name] = split_analysis

            # Print statistics
            print(f"\n{split_name.upper()} SPLIT ANALYSIS:")
            print(f"  Total samples: {total_samples:,}")
            print(
                f"  Samples with answers: {samples_with_answers:,} ({split_analysis['answer_ratio']:.1%})"
            )
            print(f"  Unique contracts: {contract_analysis['total_contracts']:,}")
            print(
                f"  Contracts with answers: {contract_analysis['contracts_with_answers']:,}"
            )
            print(
                f"  Multi-clause contracts: {contract_analysis['multi_clause_contracts']:,}"
            )
            print(
                f"  Top answered clause types: {list(answered_clause_counts.most_common(5))}"
            )

        return analysis

    def _analyze_contract_diversity(self, samples: List[ContractSample]) -> Dict:
        """Analyze contract-level diversity"""

        # Group by base contract ID
        contract_groups = defaultdict(list)

        for sample in samples:
            base_contract_id = self._extract_base_contract_id(sample.contract_id)
            contract_groups[base_contract_id].append(sample)

        # Analyze diversity
        total_contracts = len(contract_groups)
        contracts_with_answers = 0
        multi_clause_contracts = 0

        for contract_id, contract_samples in contract_groups.items():
            answered_samples = [s for s in contract_samples if s.has_answer]
            answered_clause_types = set(s.clause_type for s in answered_samples)

            if answered_samples:
                contracts_with_answers += 1

            if len(answered_clause_types) > 1:
                multi_clause_contracts += 1

        return {
            "total_contracts": total_contracts,
            "contracts_with_answers": contracts_with_answers,
            "multi_clause_contracts": multi_clause_contracts,
        }

    def create_processed_datasets(self, data_splits: Dict[str, List[ContractSample]]):
        """Create processed datasets"""
        logger.info("Creating processed datasets...")

        for split_name, samples in data_splits.items():
            # Create DataFrame
            df_data = []

            for sample in samples:
                # Clean text
                cleaned_context = self._clean_text(sample.context)
                cleaned_question = self._clean_text(sample.question)
                cleaned_answer = (
                    self._clean_text(sample.answer_text) if sample.answer_text else None
                )

                # Handle context length
                max_context_length = self.text_cleaning.get("max_context_length", 2048)
                context_words = cleaned_context.split()
                was_truncated = len(context_words) > max_context_length

                if was_truncated:
                    cleaned_context = " ".join(context_words[:max_context_length])

                # Extract base contract ID
                base_contract_id = self._extract_base_contract_id(sample.contract_id)

                df_data.append(
                    {
                        "contract_id": sample.contract_id,
                        "base_contract_id": base_contract_id,
                        "title": sample.title,
                        "context": cleaned_context,
                        "question": cleaned_question,
                        "clause_type": sample.clause_type,
                        "has_answer": sample.has_answer,
                        "answer_text": cleaned_answer,
                        "answer_start": sample.answer_start,
                        "context_length": len(cleaned_context.split()),
                        "answer_length": (
                            len(cleaned_answer.split()) if cleaned_answer else 0
                        ),
                        "was_truncated": was_truncated,
                    }
                )

            df = pd.DataFrame(df_data)

            # Save processed data
            output_file = self.processed_dir / f"{split_name}_processed.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")

            # Create clause-specific datasets
            self._create_clause_datasets(df, split_name)

    def _create_clause_datasets(self, df: pd.DataFrame, split_name: str):
        """Create separate datasets for each clause type"""
        clause_dir = self.processed_dir / "clause_datasets"
        clause_dir.mkdir(exist_ok=True)

        # Get clause types that have answers
        clause_types_with_answers = df[df["has_answer"] == True]["clause_type"].unique()

        for clause_type in clause_types_with_answers:
            if clause_type == "unknown":
                continue

            clause_df = df[df["clause_type"] == clause_type].copy()

            if len(clause_df) > 0:
                answered_count = len(clause_df[clause_df["has_answer"] == True])
                total_count = len(clause_df)
                unique_contracts = clause_df["base_contract_id"].nunique()

                # Save file
                safe_clause_type = clause_type.replace("/", "_").replace(" ", "_")
                output_file = clause_dir / f"{split_name}_{safe_clause_type}.csv"
                clause_df.to_csv(output_file, index=False)

    def create_training_samples(
        self, data_splits: Dict[str, List[ContractSample]]
    ) -> Dict:
        """Create training samples for binary and multiclass classification"""
        logger.info("Creating training samples for classification...")

        training_data = {}

        for split_name, samples in data_splits.items():
            # Only include samples with known clause types
            valid_samples = [s for s in samples if s.clause_type != "unknown"]

            # Binary classification: has_answer vs no_answer
            binary_samples = []

            # Multi-class classification: clause_type prediction
            multiclass_samples = []

            for sample in valid_samples:
                # Clean text for training
                cleaned_context = self._clean_text(sample.context)
                cleaned_question = self._clean_text(sample.question)

                # Get max length from config
                models_config = self.config.get("models", {})
                classification_config = models_config.get("classification", {})
                max_length = classification_config.get("max_length", 512)

                # Truncate context if needed (leave room for question + special tokens)
                max_context_length = max_length - len(cleaned_question.split()) - 20
                context_words = cleaned_context.split()

                if len(context_words) > max_context_length:
                    cleaned_context = " ".join(context_words[:max_context_length])

                # Create combined text for classification
                combined_text = f"{cleaned_question} [SEP] {cleaned_context}"

                # Binary classification sample
                binary_samples.append(
                    {
                        "text": combined_text,
                        "label": 1 if sample.has_answer else 0,
                        "contract_id": sample.contract_id,
                        "clause_type": sample.clause_type,
                    }
                )

                # Multi-class classification sample
                multiclass_samples.append(
                    {
                        "text": combined_text,
                        "label": sample.clause_type,
                        "contract_id": sample.contract_id,
                        "has_answer": sample.has_answer,
                    }
                )

            training_data[split_name] = {
                "binary_classification": binary_samples,
                "multiclass_classification": multiclass_samples,
            }

            # Save training samples
            for task_type, task_samples in training_data[split_name].items():
                output_file = self.processed_dir / f"{split_name}_{task_type}.json"

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(task_samples, f, indent=2, ensure_ascii=False)

                # Count answered samples
                if task_type == "binary_classification":
                    answered_samples = len([s for s in task_samples if s["label"] == 1])
                    logger.info(
                        f"Saved {len(task_samples)} {task_type} samples ({answered_samples} with answers)"
                    )
                else:
                    answered_samples = len([s for s in task_samples if s["has_answer"]])
                    logger.info(
                        f"Saved {len(task_samples)} {task_type} samples ({answered_samples} with answers)"
                    )

        return training_data


def main():
    """Main preprocessing function"""
    print("Updated CUAD Data Preprocessing Pipeline")
    print("=" * 50)

    # Initialize preprocessor
    preprocessor = UpdatedCUADPreprocessor()

    # Load data
    data_splits = preprocessor.load_data()

    if not data_splits:
        print("No data loaded. Check your data files.")
        return

    # Analyze dataset
    analysis = preprocessor.analyze_dataset(data_splits)

    # Create processed datasets
    preprocessor.create_processed_datasets(data_splits)

    # Create training samples for classification
    training_data = preprocessor.create_training_samples(data_splits)

    # Summary
    total_samples = sum(analysis[split]["total_samples"] for split in analysis)
    total_answered = sum(analysis[split]["samples_with_answers"] for split in analysis)

    print(f"  Total samples: {total_samples:,}")
    print(
        f"  Samples with answers: {total_answered:,} ({total_answered/total_samples:.1%})"
    )
    print(f"  Processed data saved to: {preprocessor.processed_dir}")

    # Show training data summary
    if training_data:
        print(f"\nTraining Data Created:")
        for split_name, tasks in training_data.items():
            print(f"  {split_name.capitalize()}:")
            for task_name, task_samples in tasks.items():
                answered_count = 0
                if task_name == "binary_classification":
                    answered_count = len([s for s in task_samples if s["label"] == 1])
                else:
                    answered_count = len([s for s in task_samples if s["has_answer"]])
                print(
                    f"    {task_name}: {len(task_samples):,} samples ({answered_count:,} with answers)"
                )


if __name__ == "__main__":
    main()
