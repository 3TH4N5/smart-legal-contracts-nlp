"""
Legal Variable Extractor

"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from hashlib import sha256
from dateutil import parser as dateparser
from word2number import w2n
from ollama import chat
from tqdm import tqdm
import logging
import argparse
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalVariableExtractor:
    """Variable extractor optimized for legal contract analysis"""

    def __init__(self, data_dir: str = "data", mode: str = "default"):
        self.data_dir = Path(data_dir)
        self.outputs_dir = Path("outputs/extracted_variables")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Set mode and contract limits
        self.mode = mode.lower()
        self.contract_limits = {"default": 15, "production": 100, "test": 5, "debug": 1}

        if self.mode not in self.contract_limits:
            logger.warning(f"Unknown mode '{mode}', defaulting to 'default'")
            self.mode = "default"

        self.max_contracts = self.contract_limits[self.mode]

        # Load configuration
        self.variable_targets = self._load_variable_targets()

        # Initialize LegalBERT integration
        self._setup_legalbert_integration()

        # Initialize extraction utilities
        self._setup_extraction_utilities()

        logger.info(
            f"{'LegalBERT' if self.has_legalbert else 'Baseline'} Variable Extractor"
        )
        logger.info(f"Mode: {self.mode.upper()} (max {self.max_contracts} contracts)")
        logger.info(
            f"Configured for {len(self.variable_targets)} clause types with {sum(len(vars) for vars in self.variable_targets.values())} total variables"
        )

    def _load_variable_targets(self) -> Dict:
        """Load variable extraction targets"""
        path = Path("config/variable_targets.json")
        if not path.exists():
            raise FileNotFoundError(f"Variable targets not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _setup_legalbert_integration(self):
        """Setup LegalBERT similarity integration"""

        # Check for LegalBERT files
        legalbert_files = {
            "metadata": self.data_dir
            / "embeddings"
            / "legalbert_clause_embeddings_all-MiniLM-L6-v2_metadata.json",
            "similarity": self.data_dir / "results" / "legalbert_similarity_matrix.npy",
        }

        baseline_files = {
            "metadata": self.data_dir
            / "embeddings"
            / "clause_embeddings_all-MiniLM-L6-v2_metadata.json",
            "similarity": self.data_dir / "results" / "similarity_matrix.npy",
        }

        # Try LegalBERT first, fallback to baseline
        self.similarity_matrix = None
        self.clause_metadata = []
        self.clause_index = {}
        self.has_legalbert = False

        for file_set, name in [
            (legalbert_files, "LegalBERT"),
            (baseline_files, "Baseline"),
        ]:
            if self._load_similarity_files(file_set):
                if name == "LegalBERT":
                    self.has_legalbert = True
                logger.info(f"Loaded {name} similarity infrastructure")
                break
        else:
            logger.warning(
                "No similarity files found - running without similarity enhancement"
            )

    def _load_similarity_files(self, files: Dict[str, Path]) -> bool:
        """Load similarity files with validation"""
        try:
            if not files["metadata"].exists():
                return False

            with open(files["metadata"], "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

            if isinstance(metadata_dict, dict):
                max_key = max(int(k) for k in metadata_dict.keys() if k.isdigit())
                self.clause_metadata = [
                    metadata_dict.get(str(i), {}) for i in range(max_key + 1)
                ]
                self.clause_metadata = [m for m in self.clause_metadata if m]
            else:
                return False

            if files["similarity"].exists():
                self.similarity_matrix = np.load(files["similarity"])
            else:
                return False

            self.clause_index = {
                meta["clause_id"]: idx
                for idx, meta in enumerate(self.clause_metadata)
                if "clause_id" in meta
            }

            logger.info(f"Loaded {len(self.clause_metadata)} clause metadata entries")
            return True

        except Exception as e:
            logger.debug(f"Failed to load similarity files: {e}")
            return False

    def _setup_extraction_utilities(self):
        """Setup extraction utilities"""

        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.llm_cache_path = self.cache_dir / "extraction_cache.json"
        self.llm_cache = self._load_json_file(self.llm_cache_path, {})

        # Pattern definitions
        self.patterns = {
            "date": [
                r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
                r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
                r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b",
                r"\b(\d{1,2})(?:st|nd|rd|th)\s+(?:day\s+of\s+)?([A-Za-z]+),?\s+(\d{4})\b",
            ],
            "money": [
                r"(?:\$|USD)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
                r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|USD)",
                r"(?:€|EUR)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
                r"(?:£|GBP)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
            ],
            "duration": [
                r"(\d+)\s+(calendar\s+)?(days?|weeks?|months?|years?)",
                r"\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\s+(?:days?|weeks?|months?|years?))\b",
                r"(\d+)-(?:day|week|month|year)",
            ],
            "percentage": [
                r"(\d+(?:\.\d+)?)\s*%",
                r"(\d+(?:\.\d+)?)\s+percent",
                r"(\d+(?:\.\d+)?)\s+per\s+cent",
            ],
        }

    def discover_contracts(self) -> Dict[str, Dict[str, int]]:
        """Discover contracts and their clause coverage"""
        logger.info("Discovering contracts and clause coverage...")

        contract_coverage = defaultdict(lambda: defaultdict(int))

        for clause_type in self.variable_targets.keys():
            dataset_path = (
                self.data_dir
                / "processed"
                / "clause_datasets"
                / f"main_{clause_type}.csv"
            )

            if not dataset_path.exists():
                continue

            try:
                df = pd.read_csv(dataset_path)
                answered_df = df[df["has_answer"].fillna(False) == True].copy()

                for _, row in answered_df.iterrows():
                    contract_id = self._normalize_contract_id(row["contract_id"])
                    contract_coverage[contract_id][clause_type] += 1

            except Exception as e:
                logger.warning(f"Error processing {clause_type}: {e}")
                continue

        return dict(contract_coverage)

    def _normalize_contract_id(self, contract_id: str) -> str:
        """Enhanced contract ID normalization"""
        if not contract_id:
            return contract_id

        contract_id = str(contract_id).strip()

        if "__" in contract_id:
            parts = contract_id.split("__")
            if len(parts) >= 2:
                clause_indicators = [
                    # Document Information
                    "document",
                    "name",
                    "parties",
                    "agreement",
                    "date",
                    "effective",
                    "expiration",
                    # Termination and Renewal
                    "renewal",
                    "term",
                    "notice",
                    "period",
                    "termination",
                    "convenience",
                    # Legal and Governance
                    "governing",
                    "law",
                    "most",
                    "favored",
                    "nation",
                    # Competition and Restrictions
                    "non",
                    "compete",
                    "exclusivity",
                    "solicit",
                    "customers",
                    "employees",
                    "competitive",
                    "restriction",
                    "exception",
                    "disparagement",
                    # Rights and Controls
                    "rofr",
                    "rofo",
                    "rofn",
                    "right",
                    "first",
                    "refusal",
                    "change",
                    "control",
                    "anti",
                    "assignment",
                    # Financial Terms
                    "revenue",
                    "profit",
                    "sharing",
                    "price",
                    "restrictions",
                    "minimum",
                    "commitment",
                    "volume",
                    "restriction",
                    # Intellectual Property
                    "ip",
                    "ownership",
                    "assignment",
                    "joint",
                    "license",
                    "grant",
                    "transferable",
                    "affiliate",
                    "licensor",
                    "licensee",
                    "unlimited",
                    "all",
                    "you",
                    "can",
                    "eat",
                    "irrevocable",
                    "perpetual",
                    "source",
                    "code",
                    "escrow",
                    # Services and Operations
                    "post",
                    "termination",
                    "services",
                    "audit",
                    "rights",
                    # Liability and Risk
                    "uncapped",
                    "liability",
                    "cap",
                    "on",
                    "liquidated",
                    "damages",
                    "warranty",
                    "duration",
                    "insurance",
                    "covenant",
                    "not",
                    "to",
                    "sue",
                    "third",
                    "party",
                    "beneficiary",
                ]

                last_part = parts[-1].lower()
                if any(indicator in last_part for indicator in clause_indicators):
                    return "__".join(parts[:-1])

        return contract_id

    def select_contracts(
        self, contract_coverage: Dict[str, Dict[str, int]], max_contracts: int = 20
    ) -> List[str]:
        """Enhanced contract selection"""

        contract_scores = []

        for contract_id, clause_counts in contract_coverage.items():
            num_types = len(clause_counts)
            total_clauses = sum(clause_counts.values())

            if num_types <= 1:
                continue

            diversity_score = num_types * 15
            volume_score = total_clauses
            complexity_bonus = 5 if num_types >= 10 else 0

            score = diversity_score + volume_score + complexity_bonus

            contract_scores.append(
                {
                    "contract_id": contract_id,
                    "num_types": num_types,
                    "total_clauses": total_clauses,
                    "score": score,
                    "coverage": clause_counts,
                }
            )

        contract_scores.sort(key=lambda x: x["score"], reverse=True)
        selected = contract_scores[:max_contracts]

        logger.info(f"Selected {len(selected)} contracts")
        return [c["contract_id"] for c in selected]

    def extract_contract_variables(self, contract_id: str) -> List[Dict]:
        """Extract all variables for a specific contract - always include parties"""

        results = []

        # Always try to extract parties first from any available clause
        parties_extracted = False

        # Method 1: Look for dedicated parties clause
        if "parties" in self.variable_targets:
            parties_results = self._extract_contract_clause_type(contract_id, "parties")
            if parties_results:
                results.extend(parties_results)
                parties_extracted = True

        # Method 2: If no parties clause found, try to extract from the first available clause
        if not parties_extracted:
            # Find the first clause with substantial content for this contract
            best_clause_for_parties = None
            best_text_length = 0

            for clause_type in self.variable_targets.keys():
                if clause_type == "parties":  # Skip, already tried
                    continue

                dataset_path = (
                    self.data_dir
                    / "processed"
                    / "clause_datasets"
                    / f"main_{clause_type}.csv"
                )
                if not dataset_path.exists():
                    continue

                try:
                    df = pd.read_csv(dataset_path)
                    df["normalized_id"] = df["contract_id"].apply(
                        self._normalize_contract_id
                    )

                    matching_clauses = df[
                        (df["normalized_id"] == contract_id)
                        & (df["has_answer"].fillna(False) == True)
                    ].copy()

                    if not matching_clauses.empty:
                        # Get the clause with the most text (likely to contain party info)
                        for _, row in matching_clauses.iterrows():
                            text = (
                                str(row.get("context", ""))
                                if pd.notna(row.get("context"))
                                else ""
                            )
                            if len(text) > best_text_length:
                                best_text_length = len(text)
                                best_clause_for_parties = (row, clause_type)

                except Exception as e:
                    logger.debug(f"Error checking {clause_type} for parties: {e}")
                    continue

            # Extract parties from the best available clause
            if (
                best_clause_for_parties and best_text_length > 500
            ):  # Minimum text length
                row, source_clause_type = best_clause_for_parties

                try:
                    # Get the full context text
                    text = (
                        str(row.get("context", ""))
                        if pd.notna(row.get("context"))
                        else ""
                    )
                    if not text:
                        text = (
                            str(row.get("answer_text", ""))
                            if pd.notna(row.get("answer_text"))
                            else ""
                        )

                    if text and len(text) > 100:
                        # Extract parties using enhanced method
                        parties = self._extract_parties_enhanced(text)
                        if parties and len(parties) >= 1:  # At least one party found
                            party_result = {
                                "contract_id": contract_id,
                                "clause_type": "parties",  # Always label as parties
                                "extractions": {"parties": parties},
                                "original_text": text[:1000],  # Truncate for storage
                                "source_clause": source_clause_type,  # Track where it came from
                            }
                            results.append(party_result)
                            parties_extracted = True
                            logger.debug(
                                f"Extracted {len(parties)} parties from {source_clause_type} clause"
                            )

                except Exception as e:
                    logger.debug(
                        f"Error extracting parties from {source_clause_type}: {e}"
                    )

        # Extract all other clause types
        for clause_type in self.variable_targets.keys():
            if clause_type == "parties" and parties_extracted:
                continue  # Already handled

            clause_results = self._extract_contract_clause_type(
                contract_id, clause_type
            )
            results.extend(clause_results)

        logger.debug(
            f"Contract {contract_id}: extracted {len(results)} total clause results, parties_extracted: {parties_extracted}"
        )
        return results

    def _extract_contract_clause_type(
        self, contract_id: str, clause_type: str
    ) -> List[Dict]:
        """Extract specific clause type for contract"""

        dataset_path = (
            self.data_dir / "processed" / "clause_datasets" / f"main_{clause_type}.csv"
        )
        if not dataset_path.exists():
            return []

        try:
            df = pd.read_csv(dataset_path)
            df["normalized_id"] = df["contract_id"].apply(self._normalize_contract_id)

            matching_clauses = df[
                (df["normalized_id"] == contract_id)
                & (df["has_answer"].fillna(False) == True)
            ].copy()

            if matching_clauses.empty:
                return []

            variables = self.variable_targets.get(clause_type, {})
            if not variables:
                return []

            results = []
            for _, row in matching_clauses.iterrows():
                try:
                    result = self._extract_clause_variables(
                        row, clause_type, variables, contract_id
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.debug(
                        f"Error processing clause in {contract_id}/{clause_type}: {e}"
                    )
                    continue

            return results

        except Exception as e:
            logger.warning(f"Error extracting {clause_type} for {contract_id}: {e}")
            return []

    def _extract_clause_variables(
        self, row: pd.Series, clause_type: str, variables: Dict, contract_id: str
    ) -> Optional[Dict]:
        """Variable extraction from a single clause with enhanced party extraction"""

        # Get text
        try:
            if clause_type == "parties":
                # For parties, we want the full context to understand the contract structure
                full_text = (
                    str(row.get("context", ""))
                    if pd.notna(row.get("context"))
                    else str(row.get("answer_text", ""))
                )
                # Take more text for party extraction - first 3000 chars where parties are typically defined
                text = full_text[:3000] if len(full_text) > 3000 else full_text
            else:
                text = (
                    str(row.get("answer_text", ""))
                    if pd.notna(row.get("answer_text"))
                    else str(row.get("context", ""))
                )

            if not text or text in ["nan", "None", "", "null"]:
                return None

            text = text.strip()
            if len(text) < 10:
                return None

        except Exception as e:
            logger.debug(f"Error getting text for {clause_type}: {e}")
            return None

        # Find similar clauses for context
        similar_clauses = []
        try:
            similar_clauses = self._find_similar_clauses(
                str(row["contract_id"]), clause_type, text
            )
        except Exception as e:
            logger.debug(f"Error finding similar clauses: {e}")

        # Extract variables
        extractions = {}

        for var_name, var_config in variables.items():
            try:
                value = self._extract_variable(
                    text, var_name, var_config, similar_clauses, clause_type
                )
                extractions[var_name] = value

            except Exception as e:
                logger.debug(f"Error extracting {var_name}: {e}")
                extractions[var_name] = None

        # Apply simple consistency rules
        extractions = self._apply_consistency_rules(extractions, text)

        # Return result if we extracted anything useful
        successful_extractions = sum(1 for v in extractions.values() if v is not None)
        if successful_extractions > 0:
            return {
                "contract_id": contract_id,
                "clause_type": clause_type,
                "extractions": extractions,
                "original_text": text,
            }

        return None

    def _apply_consistency_rules(self, extractions: Dict, text: str) -> Dict:
        """Apply consistency rules - no special handling, just fix obvious logical errors"""

        # Rule 1: If any "has*" boolean is False, set related variables to None
        for var_name, value in list(extractions.items()):
            if var_name.startswith("has") and value is False:
                # Find and null related variables
                base_name = var_name[3:]  # Remove 'has' prefix
                for other_var in extractions:
                    if (
                        other_var != var_name
                        and base_name.lower() in other_var.lower()
                        and not other_var.startswith("has")
                    ):
                        extractions[other_var] = None

        # Rule 2: If monetary amount > $500M and no clear evidence, set to None
        for var_name, value in list(extractions.items()):
            if (
                isinstance(value, dict)
                and "doubleValue" in value
                and value["doubleValue"] > 500000000
            ):
                amount = value["doubleValue"]
                if not self._large_amount_mentioned(amount, text):
                    extractions[var_name] = None

        # Rule 3: Enhanced percentage validation
        for var_name, value in list(extractions.items()):
            if isinstance(value, float) and "percentage" in var_name.lower():
                if not self._validate_percentage_extraction(value, text, var_name):
                    extractions[var_name] = None

        # Rule 4: Validate duration units make sense
        for var_name, value in list(extractions.items()):
            if (
                isinstance(value, dict)
                and value.get("$class") == "org.accordproject.time@0.3.0.Duration"
            ):
                unit = value.get("unit", "").lower()
                if unit not in ["days", "weeks", "months", "years"]:
                    extractions[var_name] = None

        # Rule 5: Basic monetary amount validation (prevent obviously wrong amounts)
        for var_name, value in list(extractions.items()):
            if (
                isinstance(value, dict)
                and value.get("$class") == "org.accordproject.money.MonetaryAmount"
            ):
                amount = value.get("doubleValue", 0)
                # Only fix obviously wrong amounts (negative or impossibly large)
                if amount <= 0 or amount > 1000000000000:  # > $1 trillion
                    extractions[var_name] = None

        return extractions

    def _validate_percentage_extraction(
        self, value: float, text: str, var_name: str
    ) -> bool:
        # Basic range check
        if value < 0 or value > 10.0:
            return False

        text_lower = text.lower()
        percentage_as_percent = value * 100

        # More flexible pattern matching
        percentage_patterns = [
            rf"\b{int(percentage_as_percent)}\s*%",  # "30%"
            rf"\b{percentage_as_percent:g}\s*%",  # "30.0%"
            rf"\b{percentage_as_percent:.1f}\s*%",  # "30.0%"
            rf"\b{int(percentage_as_percent)}\s+percent",  # "30 percent"
            rf"\bthirty\s*%",  # Handle written numbers for common values
        ]

        # Check if any pattern matches
        for pattern in percentage_patterns:
            if re.search(pattern, text_lower):
                return True

        # If we have clear revenue context and the number makes sense, allow it
        revenue_indicators = ["advertising", "sales", "revenue", "profit", "share"]
        if any(indicator in text_lower for indicator in revenue_indicators):
            return True  # Trust the LLM extraction in revenue context

        return False

    def _get_written_percentage_pattern(self, percentage: float) -> Optional[str]:
        """Get regex pattern for written percentage numbers"""

        # Map common percentages to their written forms
        written_percentages = {
            1: r"\bone\s+percent",
            2: r"\btwo\s+percent",
            3: r"\bthree\s+percent",
            4: r"\bfour\s+percent",
            5: r"\bfive\s+percent",
            10: r"\bten\s+percent",
            15: r"\bfifteen\s+percent",
            20: r"\btwenty\s+percent",
            25: r"\btwenty-five\s+percent",
            30: r"\bthirty\s+percent",
            50: r"\bfifty\s+percent",
            75: r"\bseventy-five\s+percent",
            100: r"\bone\s+hundred\s+percent",
        }

        # Convert percentage to integer for lookup
        int_percentage = int(percentage) if percentage == int(percentage) else None

        if int_percentage and int_percentage in written_percentages:
            return written_percentages[int_percentage]

        return None

    def _validate_and_clean_string(
        self, extracted: str, var_name: str, original_text: str
    ) -> Optional[str]:
        """Strict validation for string extractions"""

        if not extracted or len(extracted.strip()) < 3:
            return None

        cleaned = extracted.strip()

        # Remove LLM fluff more aggressively
        fluff_indicators = [
            "based on",
            "according to",
            "the text",
            "appears to be",
            "seems to be",
            "likely",
            "probably",
            "indicates that",
            "suggests that",
            "mentions",
            "states that",
            "specifies that",
            "refers to",
            "describes",
            "outlines",
            "establishes",
        ]

        for fluff in fluff_indicators:
            if fluff in cleaned.lower():
                return None  # Reject if contains explanatory language

        # Check for fragment indicators
        if self._is_text_fragment(cleaned):
            return None

        # Validate it's a complete, meaningful unit
        if not self._is_meaningful_string(cleaned, var_name, original_text):
            return None

        return cleaned

    def _is_text_fragment(self, value_str: str) -> bool:
        """Enhanced fragment detection"""

        fragment_indicators = [
            # Dangling connectors
            value_str.endswith(" and "),
            value_str.endswith(" or "),
            value_str.endswith(" to "),
            value_str.endswith(" with "),
            value_str.startswith("and "),
            value_str.startswith("or "),
            value_str.startswith(") and "),
            value_str.startswith(") or "),
            # Unbalanced punctuation
            value_str.count("(") != value_str.count(")"),
            value_str.count('"') % 2 != 0,
            value_str.count("'") % 2 != 0,
            # Incomplete references
            " )" in value_str and not value_str.strip().endswith(")"),
            value_str.endswith(" ("),
            value_str.startswith("(") and ")" not in value_str,
            # Legal document fragments
            re.search(r"\bSection\s+\d+\.\d+\s*\([^)]*$", value_str),
            re.search(r"^[^(]*\)\s+and\s+Section", value_str),
            # Grammatically incomplete
            len(value_str.split()) < 3
            and any(
                word in value_str.lower()
                for word in [
                    "section",
                    "clause",
                    "grant",
                    "scope",
                    "pursuant",
                    "accordance",
                ]
            ),
        ]

        return any(fragment_indicators)

    def _is_meaningful_string(
        self, extracted: str, var_name: str, original_text: str
    ) -> bool:
        """Check if extracted string is meaningful and complete"""

        # Should be reasonable length
        if len(extracted) < 5 or len(extracted) > 500:
            return False

        # Should not be just punctuation or numbers
        if not re.search(r"[a-zA-Z]", extracted):
            return False

        # Should not be mostly punctuation
        punct_ratio = len(re.findall(r"[^\w\s]", extracted)) / len(extracted)
        if punct_ratio > 0.3:  # More than 30% punctuation
            return False

        # For scope-related variables, be extra strict
        if "scope" in var_name.lower():
            # Should not contain dangling references
            if re.search(r"\bSection\s+\d+\.\d+\s*\([^)]*$", extracted):
                return False
            if extracted.endswith(" to ") or extracted.endswith(" and "):
                return False

        return True

    def _number_to_word(self, num: int) -> str:
        """Convert number to word for validation"""
        word_map = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
            19: "nineteen",
            20: "twenty",
            30: "thirty",
            40: "forty",
            50: "fifty",
            60: "sixty",
            70: "seventy",
            80: "eighty",
            90: "ninety",
            100: "hundred",
        }
        return word_map.get(num, str(num))

    def _large_amount_mentioned(self, amount: float, text: str) -> bool:
        """Check if a large monetary amount is actually mentioned in text"""
        text_lower = text.lower()

        # Check for billion/million mentions
        if amount >= 1000000000:
            return "billion" in text_lower
        elif amount >= 1000000:
            return "million" in text_lower or "billion" in text_lower

        # For amounts > $10M, require explicit large number patterns
        if amount >= 10000000:
            large_patterns = [
                r"\$\s*\d{2,}[,\d]*",  # $10,000,000 style
                r"\d{8,}",  # 10000000 style
                r"\d+\s*million",  # "10 million" style
                r"\d+\s*billion",  # "1 billion" style
            ]
            return any(re.search(pattern, text_lower) for pattern in large_patterns)

        return True  # For smaller amounts, be more lenient

    def _extract_variable(
        self,
        text: str,
        var_name: str,
        var_config: Dict,
        similar_clauses: List[Dict],
        clause_type: str,
    ) -> any:
        """Extract variable with appropriate method based on type"""

        var_type = var_config.get("type", "String")

        # For parties, use LLM-first approach with smart fallbacks
        if var_type in ["Party", "Party[]"]:
            return self._extract_parties_enhanced(text)

        # Try pattern-based extraction first for other types
        pattern_result = self._pattern_extraction(text, var_name, var_type, clause_type)
        if pattern_result is not None:
            return pattern_result

        # Use LLM extraction
        result = self._llm_extraction(
            text, var_name, var_type, similar_clauses, clause_type
        )

        # For strings, add strict validation
        if var_type == "String" and isinstance(result, str):
            validated_result = self._validate_and_clean_string(result, var_name, text)
            return validated_result

        return result

    def _extract_parties_enhanced(self, text: str) -> Optional[List[Dict]]:
        """Enhanced party extraction using LLM-first approach with smart fallbacks"""

        logger.debug("Starting enhanced party extraction")

        # Method 1: LLM-based extraction (primary method)
        llm_parties = self._extract_parties_llm(text)
        if llm_parties and len(llm_parties) >= 1:
            logger.debug(f"LLM extracted {len(llm_parties)} parties")
            return llm_parties

        # Method 2: Regex fallback with improved patterns
        regex_parties = self._extract_parties_regex_enhanced(text)
        if regex_parties and len(regex_parties) >= 1:
            logger.debug(f"Regex extracted {len(regex_parties)} parties")
            return regex_parties

        # Method 3: Simple entity extraction fallback
        entity_parties = self._extract_parties_entity_fallback(text)
        if entity_parties and len(entity_parties) >= 1:
            logger.debug(f"Entity fallback extracted {len(entity_parties)} parties")
            return entity_parties

        logger.debug("No parties extracted by any method")
        return None

    def _extract_parties_llm(self, text: str) -> Optional[List[Dict]]:
        """Extract parties using LLM with improved text preprocessing"""

        # Better preprocessing - focus on contract opening and remove table artifacts
        relevant_text = text[:2500]

        # Remove common document artifacts that confuse extraction
        artifacts_to_remove = [
            r"Exhibit\s+\d+\.\d+",
            r"TABLE OF CONTENTS",
            r"Section\s+Title\s+Page#",
            r"Initials\s+Franchisee\s+Franchisor",
            r"\$\s*\)\s*\.",  # Currency formatting artifacts
            r"☐\s+[^☐]*(?=☐|$)",  # Checkbox content
            r"Declarations\s+Page",
            r"Page\s*#?\s*\d*",
        ]

        cleaned_text = relevant_text
        for pattern in artifacts_to_remove:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

        # Focus on the actual contract parties section
        party_section_indicators = [
            r"made\s+and\s+entered\s+into.*?by\s+and\s+between",
            r"this\s+agreement.*?between",
            r"agreement.*?is.*?between",
        ]

        party_section = ""
        for indicator in party_section_indicators:
            match = re.search(
                f"{indicator}(.*?)(?:RECITALS|WHEREAS|NOW\s+THEREFORE|Introduction|Section\s+1|Grant)",
                cleaned_text,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                party_section = match.group(1)
                break

        # If we found a specific party section, use that, otherwise use the cleaned text
        extraction_text = party_section if party_section.strip() else cleaned_text

        prompt = f"""Extract ONLY the exact legal entity names that are parties to this contract.

    CRITICAL RULES:
    - Return ONLY company names like "Apple Inc." or "Microsoft Corporation"
    - DO NOT include any explanatory text, introductions, or descriptions
    - DO NOT include "Here are..." or "The parties are..." or similar phrases
    - DO NOT include jurisdictional descriptions like "a Delaware corporation", "Ohio corporation", etc.
    - DO NOT include addresses or descriptive text
    - DO NOT include "and between" or "by and between"
    - DO NOT include formatting artifacts like "Initials" or "Franchisee Franchisor"
    - Include the complete company name with entity type (Corp, Inc, LLC, etc.)
    - One company name per line, nothing else
    - If no companies found, return "NONE"

    CONTRACT TEXT:
    {extraction_text}

    LEGAL ENTITY NAMES:"""

        try:
            response = self._query_llm_simple(prompt)
            if not response or response.upper().strip() == "NONE":
                return None

            # Parse the response with better cleaning
            parties = []
            lines = response.strip().split("\n")

            for line in lines:
                clean_line = line.strip()

                # Remove numbered list markers, bullets
                clean_line = re.sub(r"^[\d\.\-\•\*\+\s]*", "", clean_line).strip()

                # Remove LLM fluff more aggressively
                fluff_patterns = [
                    r"^(?:the\s+)?(?:parties\s+are\s*:?\s*)",
                    r"^(?:based\s+on.*?:?\s*)",
                    r"^(?:from\s+the\s+contract.*?:?\s*)",
                    r"^(?:extracted.*?:?\s*)",
                    r"^(?:legal\s+entities?\s*:?\s*)",
                    r"^(?:companies?\s*:?\s*)",
                ]

                for pattern in fluff_patterns:
                    clean_line = re.sub(
                        pattern, "", clean_line, flags=re.IGNORECASE
                    ).strip()

                # Skip obvious artifacts - be more specific about what to skip
                skip_indicators = [
                    "initials",
                    "franchisee franchisor",
                    "table of contents",
                    "exhibit",
                    "page #",
                    "declarations page",
                    "here are",
                    "based on",
                    "extracted",
                    "parties are",
                    "following",
                    "legal entities",
                    "companies:",
                    "parties:",
                ]

                if any(
                    indicator in clean_line.lower() for indicator in skip_indicators
                ):
                    continue

                # Remove bad prefixes
                bad_prefixes = [
                    "and between ",
                    "by and between ",
                    "or under common control with ",
                    "under common control with ",
                ]

                for prefix in bad_prefixes:
                    if clean_line.lower().startswith(prefix):
                        clean_line = clean_line[len(prefix) :].strip()
                        break

                # More precise jurisdictional suffix removal - only remove the suffix part
                clean_line = re.sub(
                    r",\s*(?:a|an)\s+[^,]+(?:corporation|company|llc|limited|partnership).*$",
                    "",
                    clean_line,
                    flags=re.IGNORECASE,
                ).strip()
                clean_line = re.sub(
                    r",\s*having\s+.*$", "", clean_line, flags=re.IGNORECASE
                ).strip()
                clean_line = re.sub(
                    r",\s*with\s+offices?\s+.*$", "", clean_line, flags=re.IGNORECASE
                ).strip()
                clean_line = re.sub(
                    r",\s*which\s+.*$", "", clean_line, flags=re.IGNORECASE
                ).strip()
                clean_line = re.sub(
                    r",\s*organized\s+under.*$", "", clean_line, flags=re.IGNORECASE
                ).strip()

                # Clean up punctuation
                clean_line = clean_line.strip(".,;:\"'()")

                # Enhanced validation - must be actual company names, not standalone jurisdictional terms
                if (
                    clean_line
                    and len(clean_line) > 5
                    and len(clean_line) < 100
                    # Must contain actual entity indicators
                    and any(
                        entity in clean_line.lower()
                        for entity in [
                            "corp",
                            "inc",
                            "llc",
                            "ltd",
                            "limited",
                            "company",
                            "co",
                            "systems",
                            "enterprises",
                            "holdings",
                            "group",
                            "partners",
                            "solutions",
                        ]
                    )
                    # Must not be document artifacts
                    and not any(
                        bad in clean_line.lower()
                        for bad in [
                            "initials",
                            "franchisee",
                            "franchisor",
                            "exhibit",
                            "declarations",
                            "page",
                        ]
                    )
                    # Must not be ONLY jurisdictional terms (standalone)
                    and not re.match(
                        r"^(?:delaware|ohio|california|nevada|texas|florida|new\s+york|new\s+jersey|massachusetts)\s+(?:corporation|company|llc|limited)$",
                        clean_line,
                        re.IGNORECASE,
                    )
                    # Must not start with articles followed by jurisdictions
                    and not re.match(
                        r"^(?:a|an)\s+(?:delaware|ohio|california|nevada|texas|florida|new\s+york|new\s+jersey|massachusetts)",
                        clean_line,
                        re.IGNORECASE,
                    )
                    # Must not start with document artifacts
                    and not clean_line.lower().startswith(
                        ("initials", "franchisee", "franchisor", "exhibit")
                    )
                    # Must have at least one substantial word that's not just a jurisdiction
                    and len(
                        [
                            w
                            for w in clean_line.split()
                            if len(w) > 2
                            and w.lower()
                            not in [
                                "delaware",
                                "ohio",
                                "california",
                                "nevada",
                                "texas",
                                "florida",
                                "corporation",
                                "company",
                                "llc",
                                "limited",
                            ]
                        ]
                    )
                    > 0
                ):
                    parties.append(
                        {
                            "$class": "org.accordproject.organization.Organization",
                            "partyId": clean_line,
                        }
                    )

            # Remove duplicates while preserving order
            seen_names = set()
            unique_parties = []
            for party in parties:
                name = party["partyId"]
                if name not in seen_names:
                    seen_names.add(name)
                    unique_parties.append(party)

            return unique_parties[:4] if unique_parties else None

        except Exception as e:
            logger.debug(f"LLM party extraction failed: {e}")

        return None

    def _extract_parties_regex_enhanced(self, text: str) -> Optional[List[Dict]]:
        """Enhanced regex-based party extraction with better patterns"""

        # Focus on the contract opening, but take more text to capture both parties
        opening_text = text[:3000]
        found_parties = []

        # Method 1: Enhanced "by and between" pattern with better parsing
        between_patterns = [
            # Pattern for "made and entered into ... by and between X and Y"
            r"(?:made|entered\s+into).*?(?:by\s+and\s+)?between\s+(.*?)(?:\s+(?:NOW\s+THEREFORE|WHEREAS|BACKGROUND|RECITALS|WITNESSETH|Introduction))",
            # Pattern for simple "between X and Y"
            r"(?:by\s+and\s+)?between\s+(.*?)(?:\s+(?:NOW\s+THEREFORE|WHEREAS|BACKGROUND|RECITALS|WITNESSETH|Introduction))",
            # Pattern for "agreement is between X and Y"
            r"agreement.*?between\s+(.*?)(?:\s+(?:NOW\s+THEREFORE|WHEREAS|BACKGROUND|RECITALS|WITNESSETH|Introduction|Grant))",
        ]

        for pattern in between_patterns:
            match = re.search(pattern, opening_text, re.IGNORECASE | re.DOTALL)
            if match:
                parties_text = match.group(1).strip()

                # Clean up the parties text first
                parties_text = re.sub(r"Exhibit\s+\d+\.\d+", "", parties_text)
                parties_text = re.sub(
                    r"TABLE OF CONTENTS.*", "", parties_text, flags=re.DOTALL
                )
                parties_text = re.sub(
                    r"Initials\s+Franchisee\s+Franchisor", "", parties_text
                )

                # Enhanced splitting - look for "and" that separates two complete party definitions
                # This regex looks for "and" followed by something that looks like a party
                company_separator = r"\s+and\s+(?=(?:[A-Z][^,]*(?:Corporation|Corp\.?|Inc\.?|LLC|Ltd\.?|Limited|Company|Co\.?)|●\s*[A-Z]))"
                segments = re.split(
                    company_separator, parties_text, flags=re.IGNORECASE
                )

                for segment in segments:
                    segment = segment.strip().strip("●").strip()  # Remove bullet points

                    if not segment:
                        continue

                    # Extract company name from each segment
                    party_name = self._extract_complete_company_name(segment)
                    if party_name and self._is_valid_party_name(party_name):
                        found_parties.append(
                            {
                                "$class": "org.accordproject.organization.Organization",
                                "partyId": party_name,
                            }
                        )

                if len(found_parties) >= 1:  # At least one party found
                    return found_parties[:4]

        # Method 2: Look for bullet point format (● CompanyName)
        bullet_pattern = r"●\s+([A-Z][^●\n]*?(?:Corporation|Corp\.?|Inc\.?|LLC|Ltd\.?|Limited|Company|Co\.?)[^●\n]*?)(?:,\s*(?:a|an)\s+[^,\n]*(?:corporation|company|llc|limited).*?)?(?=●|$)"
        bullet_matches = re.findall(
            bullet_pattern, opening_text, re.IGNORECASE | re.DOTALL
        )

        for match in bullet_matches:
            cleaned = self._clean_company_name_enhanced(match.strip())
            if cleaned and self._is_valid_party_name(cleaned):
                found_parties.append(
                    {
                        "$class": "org.accordproject.organization.Organization",
                        "partyId": cleaned,
                    }
                )

        if len(found_parties) >= 1:
            # Remove duplicates while preserving order
            seen_names = set()
            unique_parties = []
            for party in found_parties:
                name = party["partyId"].lower().strip()
                if name not in seen_names:
                    seen_names.add(name)
                    unique_parties.append(party)

            return unique_parties[:4] if unique_parties else None

        # Method 3: Direct company definition format
        definition_patterns = [
            # "Company Name, Inc., a Delaware corporation"
            r"([A-Z][^,\n]{5,}(?:Corporation|Corp\.?|Inc\.?|LLC|Ltd\.?|Limited|Company|Co\.?)[^,\n]*),\s*a\s+[^,\n]*(?:corporation|company|llc|limited|partnership)",
            # Company names in quotes
            r'"([A-Z][^"]{5,}(?:Corporation|Corp\.?|Inc\.?|LLC|Ltd\.?|Limited|Company|Co\.?)[^"]*)"',
        ]

        for pattern in definition_patterns:
            matches = re.findall(pattern, opening_text, re.IGNORECASE)
            for match in matches:
                cleaned = self._clean_company_name_enhanced(match)
                if cleaned and self._is_valid_party_name(cleaned):
                    found_parties.append(
                        {
                            "$class": "org.accordproject.organization.Organization",
                            "partyId": cleaned,
                        }
                    )

        # Remove duplicates
        seen_names = set()
        unique_parties = []
        for party in found_parties:
            name = party["partyId"].lower().strip()
            if name not in seen_names:
                seen_names.add(name)
                unique_parties.append(party)

        return unique_parties[:4] if unique_parties else None

    def _extract_complete_company_name(self, segment: str) -> Optional[str]:
        """Extract complete company name from a text segment with better parsing"""
        if not segment or len(segment) < 5:
            return None

        # Clean up the segment first
        segment = segment.strip().strip(",.\"'")

        # Look for company name pattern - everything up to first comma or descriptive phrase
        # Pattern: "Company Name, Inc." or "Company Name Corp" etc.
        entity_suffixes = [
            "Corporation",
            "Corp\.?",
            "Inc\.?",
            "Incorporated",
            "LLC",
            "L\.L\.C\.?",
            "Ltd\.?",
            "Limited",
            "Company",
            "Co\.?",
            "LLP",
            "L\.L\.P\.?",
            "LP",
            "L\.P\.?",
            "PLC",
            "Systems",
            "Enterprises",
            "Group",
            "Holdings",
            "Partners",
            "Solutions",
        ]
        suffix_pattern = "|".join(entity_suffixes)

        # Try to find complete company name with entity suffix
        # Look for pattern: "Company Name + Entity Type" before any description
        company_patterns = [
            # Pattern 1: "Company Name, Inc.," or "Company Name Corp,"
            rf"([A-Z][^,\n]*?(?:{suffix_pattern})\.?)\s*(?:,|$|\s+(?:a\s+|having\s+|with\s+|organized\s+))",
            # Pattern 2: "Company Name Inc" (no comma)
            rf"([A-Z][^,\n]*?(?:{suffix_pattern})\.?)(?:\s+(?:a\s+|having\s+|with\s+|organized\s+)|$)",
            # Pattern 3: Just the company name if it ends with entity suffix
            rf"([A-Z][^,\n]*?(?:{suffix_pattern})\.?)(?:\s|$)",
        ]

        for pattern in company_patterns:
            match = re.search(pattern, segment, re.IGNORECASE)
            if match:
                company_name = match.group(1).strip()
                # Clean up trailing punctuation but preserve internal punctuation
                company_name = company_name.rstrip(".,;")

                # Additional validation - should be reasonable length
                if 5 <= len(company_name) <= 100:
                    return self._clean_company_name_enhanced(company_name)

        return None

    def _clean_company_name_enhanced(self, name: str) -> Optional[str]:
        """Enhanced cleaning of company names"""
        if not name:
            return None

        # Basic cleanup
        name = name.strip().strip(".,;()[]\"'")

        # Remove leading articles or conjunctions that might have been captured
        name = re.sub(
            r"^(?:and\s+|the\s+|by\s+|between\s+)", "", name, flags=re.IGNORECASE
        ).strip()

        # Remove any remaining leading/trailing whitespace and normalize internal whitespace
        name = " ".join(name.split())

        # Remove any text after certain stopping words that indicate descriptions
        stopping_phrases = [
            r",\s*a\s+(?:delaware|california|new\s+york|nevada|texas|florida)\s+(?:corporation|company|llc)",
            r",\s*having\s+its\s+principal",
            r",\s*with\s+offices\s+(?:at|located)",
            r",\s*organized\s+under",
            r",\s*incorporated\s+in",
        ]

        for phrase_pattern in stopping_phrases:
            name = re.split(phrase_pattern, name, flags=re.IGNORECASE)[0].strip()

        # Final validation
        if len(name) < 3 or len(name) > 120:  # Reasonable length bounds
            return None

        return name

    def _is_valid_party_name(self, name: str) -> bool:
        """Simplified validation for party names"""
        if not name or len(name) < 5:
            return False

        name_lower = name.lower().strip()

        # Must contain entity indicators
        entity_indicators = [
            "corp",
            "corporation",
            "inc",
            "incorporated",
            "llc",
            "ltd",
            "limited",
            "company",
            "co.",
            "co",
            "llp",
            "lp",
            "plc",
            "l.p.",
            "l.l.c.",
            "systems",
            "enterprises",
            "group",
            "holdings",
            "partners",
            "solutions",
        ]

        has_entity_indicator = any(
            indicator in name_lower for indicator in entity_indicators
        )
        if not has_entity_indicator:
            return False

        # Comprehensive exclusion patterns
        exclusions = [
            # Jurisdictional descriptions
            "delaware corporation",
            "delaware company",
            "new jersey llc",
            "california company",
            "new york corporation",
            "nevada corporation",
            "massachusetts corporation",
            "texas corporation",
            "organized under",
            "incorporated in",
            "formed under",
            "chartered in",
            "domiciled in",
            "a corporation",
            "a company",
            "a llc",
            "a limited",
            "a partnership",
            # Location/address information
            "having principal",
            "with offices",
            "located at",
            "situated at",
            "based in",
            "principal place",
            "place of business",
            "office at",
            "address at",
            "headquarters at",
            # Contract language patterns
            "and between",
            "by and between",
            "made between",
            "entered into between",
            "or under common control",
            "under common control with",
            "controlling, controlled by",
            "affiliates controlling",
            "controlled by or under",
            "under common control",
            # Document/procedural language
            "table of contents",
            "exhibit",
            "schedule",
            "attachment",
            "addendum",
            "whereas",
            "agreement",
            "contract",
            "confidential",
            "execution version",
            # Common LLM response patterns
            "here are the",
            "based on the",
            "extracted from",
            "following companies",
            "parties are:",
            "entities are:",
            "companies are:",
            "note:",
            "contract:",
        ]

        # Check for exclusions
        if any(exclusion in name_lower for exclusion in exclusions):
            return False

        # Check if starts with conjunctions/prepositions/articles
        bad_starts = [
            "and ",
            "or ",
            "by ",
            "with ",
            "under ",
            "for ",
            "from ",
            "to ",
            "in ",
            "at ",
            "the ",
            "a ",
            "an ",
            "of ",
            "on ",
            "as ",
            "but ",
            "if ",
            "then ",
            "than ",
            "between ",
            "among ",
            "through ",
            "during ",
            "before ",
            "after ",
        ]
        if any(name_lower.startswith(start) for start in bad_starts):
            return False

        # Should be mostly letters and spaces (not mostly punctuation/numbers)
        letter_count = sum(1 for c in name if c.isalpha() or c.isspace())
        if letter_count < len(name) * 0.7:  # At least 70% letters/spaces
            return False

        # Length validation - reasonable bounds for company names
        if len(name) < 5 or len(name) > 120:
            return False

        # Should contain at least one substantial word (3+ chars) that's not an entity type
        words = name.split()
        entity_words = {
            "corp",
            "corporation",
            "inc",
            "incorporated",
            "llc",
            "ltd",
            "limited",
            "company",
            "co",
            "llp",
            "lp",
            "plc",
            "systems",
            "group",
            "enterprises",
            "holdings",
            "partners",
            "solutions",
        }
        substantial_words = [
            w for w in words if len(w) >= 3 and w.lower() not in entity_words
        ]

        if len(substantial_words) == 0:  # No proper nouns besides entity suffixes
            return False

        return True

    def _extract_parties_entity_fallback(self, text: str) -> Optional[List[Dict]]:
        """Enhanced fallback method to extract any entities that look like companies"""

        opening_text = text[:2000]  # Focus on contract opening
        entities = []

        # Look for company name patterns
        patterns = [
            r"\b([A-Z][a-zA-Z\s&\.,\-\']{5,60}(?:Corporation|Corp\.?|Inc\.?|LLC|Ltd\.?|Limited|Company|Co\.?)\.?)\b",
            r'"([A-Z][^"]{5,60}(?:Corporation|Corp\.?|Inc\.?|LLC|Ltd\.?|Limited|Company|Co\.?)\.?)"',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, opening_text, re.IGNORECASE)

            for match in matches:
                # Clean the match
                clean_match = match.strip(".,;:\"'()")

                # Remove jurisdictional suffixes
                clean_match = re.sub(
                    r",\s*a\s+[^,]+(?:corporation|company|llc|limited).*$",
                    "",
                    clean_match,
                    flags=re.IGNORECASE,
                ).strip()
                clean_match = re.sub(
                    r",\s*having\s+.*$", "", clean_match, flags=re.IGNORECASE
                ).strip()
                clean_match = re.sub(
                    r",\s*with\s+offices?\s+.*$", "", clean_match, flags=re.IGNORECASE
                ).strip()

                # Skip bad patterns
                if any(
                    bad in clean_match.lower()
                    for bad in [
                        "delaware corporation",
                        "new jersey llc",
                        "california company",
                        "having its principal",
                        "with offices",
                        "located at",
                        "organized under",
                        "table of contents",
                        "exhibit",
                        "schedule",
                        "agreement",
                    ]
                ):
                    continue

                # Skip if starts with bad words
                if any(
                    clean_match.lower().startswith(bad)
                    for bad in [
                        "and ",
                        "or ",
                        "by ",
                        "with ",
                        "under ",
                        "for ",
                        "from ",
                        "to ",
                        "in ",
                        "at ",
                        "the ",
                        "a ",
                        "an ",
                        "of ",
                        "on ",
                        "as ",
                        "but ",
                        "if ",
                        "then ",
                    ]
                ):
                    continue

                # Basic validation - removed uppercase requirement
                if (
                    clean_match
                    and len(clean_match) > 5
                    and len(clean_match) < 100
                    and any(
                        entity in clean_match.lower()
                        for entity in [
                            "corp",
                            "inc",
                            "llc",
                            "ltd",
                            "limited",
                            "company",
                            "co",
                        ]
                    )
                ):

                    entities.append(
                        {
                            "$class": "org.accordproject.organization.Organization",
                            "partyId": clean_match,
                        }
                    )

        # Remove duplicates
        seen_names = set()
        unique_entities = []
        for entity in entities:
            name = entity["partyId"]
            normalized_name = re.sub(r"\s+", " ", name.lower().strip())
            if normalized_name not in seen_names:
                seen_names.add(normalized_name)
                unique_entities.append(entity)

        return unique_entities[:4] if unique_entities else None

    def _is_legitimate_company_name(self, name: str) -> bool:
        """Very strict validation for company names in fallback method"""
        if not name or len(name) < 5:
            return False

        name_lower = name.lower()

        # Must have entity suffix
        entity_suffixes = [
            "corporation",
            "corp",
            "inc",
            "incorporated",
            "llc",
            "l.l.c",
            "ltd",
            "limited",
            "company",
            "co",
            "llp",
            "l.l.p",
            "lp",
            "l.p",
            "plc",
            "systems",
            "enterprises",
            "group",
            "holdings",
            "partners",
            "solutions",
        ]

        has_entity_suffix = any(
            name_lower.endswith(f" {suffix}") or name_lower.endswith(f" {suffix}.")
            for suffix in entity_suffixes
        )
        if not has_entity_suffix:
            return False

        # Aggressive exclusion of common false positives
        strict_exclusions = [
            # Jurisdictional/descriptive text
            "delaware",
            "california",
            "new york",
            "nevada",
            "texas",
            "florida",
            "corporation organized",
            "company having",
            "llc formed",
            "principal place",
            "place of business",
            "offices located",
            # Document references
            "table of contents",
            "exhibit",
            "schedule",
            "agreement",
            "franchise",
            "master",
            "strategic alliance",
            "collaboration",
            "license",
            "hosting",
            "network affiliate",
            "content license",
            # Generic business terms that aren't actual company names
            "the franchised business",
            "the system",
            "the company",
            "franchisor llc",
            "franchise systems",
            "original cookies",
            # LLM artifacts
            "here are",
            "following",
            "contract",
            "parties",
            "entities",
            "note",
            "there is only",
            "mentioned",
            "extracted",
        ]

        if any(exclusion in name_lower for exclusion in strict_exclusions):
            return False

        # Must have at least one proper noun (capitalized word) beyond entity suffix
        words = name.split()
        entity_words = {
            "corp",
            "corporation",
            "inc",
            "incorporated",
            "llc",
            "ltd",
            "limited",
            "company",
            "co",
            "llp",
            "lp",
            "plc",
            "systems",
            "enterprises",
            "group",
            "holdings",
            "partners",
            "solutions",
        }
        proper_nouns = [
            w
            for w in words
            if len(w) > 1 and w[0].isupper() and w.lower() not in entity_words
        ]

        if len(proper_nouns) == 0:
            return False

        # Should not be mostly punctuation or numbers
        alpha_ratio = sum(1 for c in name if c.isalpha()) / len(name)
        if alpha_ratio < 0.75:  # At least 75% letters
            return False

        # Should not be unreasonably long (likely grabbed too much text)
        if len(name) > 80:
            return False

        return True

    def _query_llm_simple(self, prompt: str) -> str:
        """Simple LLM query without caching for one-off extractions"""
        try:
            response = chat(
                model="llama3",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise legal document analyzer. Follow instructions exactly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.1,
                    "seed": 42,
                    "num_predict": 100,
                    "top_p": 0.9,
                },
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Simple LLM query failed: {e}")
            return ""

    def _llm_extraction(
        self,
        text: str,
        var_name: str,
        var_type: str,
        similar_clauses: List[Dict],
        clause_type: str,
    ) -> any:
        """LLM-based variable extraction"""

        if var_type == "String":
            prompt = f"""Extract ONLY the exact {var_name} content. No explanations.

    CRITICAL: 
    - Return ONLY the extracted text EXACTLY as written
    - NO phrases like "The {var_name} is:" or "Based on:"
    - NO explanatory text whatsoever
    - NO partial sentences or fragments
    - DO NOT remove parentheses, section references, or legal citations
    - DO NOT clean up or reformat the text
    - Maximum 200 characters
    - Return 'null' if not clearly and completely present

    Text: {text[:1000]}

    {var_name}:"""

        elif var_type == "Boolean":
            prompt = f"""Analyze this legal text to determine if {var_name} is present.

    GUIDELINES:
    - Look for the underlying concept or substance, not just explicit wording
    - If the text creates, describes, implies, or establishes the concept → TRUE
    - If the text prohibits, excludes, or negates the concept → FALSE  
    - Consider both direct statements and practical effects
    - When the concept exists in substance (even if not explicitly named) → TRUE
    - Only return FALSE if clearly absent or explicitly prohibited
    - Default to TRUE unless you're certain the concept is absent or negated

    RESPONSE FORMAT: Answer with exactly one word: true, false, or null

    Text: {text[:1000]}

    Is {var_name} present? (true/false/null):"""

        elif var_type == "Duration":
            prompt = f"""Extract ONLY the time duration for {var_name} if stated.

    CRITICAL: Return ONLY the duration (e.g., "30 days", "2 years"). No introductory text.

    Format: "X unit" (e.g., "30 days", "2 years")
    Return 'null' if not found.

    Text: {text[:1000]}

    Duration:"""

        elif var_type == "Double":
            # Special handling for percentage variables
            if any(
                keyword in var_name.lower()
                for keyword in ["percentage", "percent", "share", "rate"]
            ):
                prompt = f"""Extract the percentage value for {var_name} from this legal text.

        CRITICAL INSTRUCTIONS:
        - Look for percentage signs (%) or the word "percent"
        - Convert percentages to decimal format: 30% = 0.30, 15% = 0.15
        - Return ONLY the decimal number (e.g., 0.30, 0.15)
        - If multiple percentages exist, choose the one most relevant to {var_name}
        - Return 'null' if no clear percentage is found

        EXAMPLES:
        - "receive 30% of sales" → 0.30
        - "15 percent commission" → 0.15  
        - "25% revenue sharing" → 0.25

        Text: {text[:1000]}

        Decimal value:"""
            else:
                # Regular Double extraction
                prompt = f"""Extract ONLY the number/decimal for {var_name} if stated.

        CRITICAL: 
        - For decimals like "0.15", return: 0.15  
        - For whole numbers like "5", return: 5.0
        - Return ONLY the numeric value. No explanatory text.
        - Return 'null' if not found.

        Text: {text[:1000]}

        Value:"""
        elif var_type == "MonetaryAmount":
            prompt = f"""Extract ONLY the monetary amount for {var_name} if stated.

        CRITICAL: 
    - Only extract if there's a clear monetary value mentioned (with $, dollars, currency)
    - For {var_name}: look for the specific amount related to this field
    - If multiple amounts exist, choose the one most relevant to {var_name}
    - Handle financial notation: MM = million, M = million, K = thousand, B = billion
    - DO NOT extract percentages like "100%" or "90%" as dollar amounts
    - Return 'null' if no clear amount is found


        Text: {text[:1000]}

        Amount:"""

        else:
            prompt = f"""Extract ONLY the {var_name} value from this text.

        CRITICAL: Return ONLY the extracted value. No explanations.

        Rules:
        - Return only the specific value, no explanations
        - Maximum 100 characters
        - Return 'null' if not found

        Text: {text[:1000]}

        Value:"""

        return self._query_llm_with_cache(prompt, var_name, var_type)

    def _pattern_extraction(
        self, text: str, var_name: str, var_type: str, clause_type: str
    ) -> any:
        """Pattern-based extraction"""

        if var_type == "DateTime":
            return self._extract_date(text)
        elif var_type == "MonetaryAmount":
            return self._extract_money(text)
        elif var_type == "Duration":
            return self._extract_duration(text)
        elif var_type == "Double":
            # Handle both integers and decimals as Double
            integer_result = self._extract_integer_as_double(text)
            if integer_result is not None:
                return integer_result
            return self._extract_percentage(text)
        elif var_type == "Boolean":
            return self._extract_boolean(text, var_name)

        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date with ISO formatting"""
        for pattern in self.patterns["date"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 3:
                        if pattern == self.patterns["date"][0]:  # ISO format
                            year, month, day = map(int, match.groups())
                        elif pattern == self.patterns["date"][1]:  # US format
                            month, day, year = map(int, match.groups())
                        else:  # Month name formats
                            if pattern == self.patterns["date"][3]:  # Ordinal
                                day, month_name, year = match.groups()
                            else:  # Regular month name
                                month_name, day, year = match.groups()

                            month_map = {
                                "january": 1,
                                "february": 2,
                                "march": 3,
                                "april": 4,
                                "may": 5,
                                "june": 6,
                                "july": 7,
                                "august": 8,
                                "september": 9,
                                "october": 10,
                                "november": 11,
                                "december": 12,
                                "jan": 1,
                                "feb": 2,
                                "mar": 3,
                                "apr": 4,
                                "jun": 6,
                                "jul": 7,
                                "aug": 8,
                                "sep": 9,
                                "oct": 10,
                                "nov": 11,
                                "dec": 12,
                            }
                            month = month_map.get(month_name.lower())
                            if not month:
                                continue
                            day, year = int(day), int(year)

                        date_obj = datetime(year, month, day)
                        return date_obj.strftime("%Y-%m-%dT00:00:00.000Z")
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_money(self, text: str) -> Optional[Dict]:
        """Extract monetary amount with enhanced patterns for multipliers"""

        # Enhanced patterns that capture multipliers and financial notation
        money_patterns = [
            # Pattern 1: $20MM, $5M, $10B, $2K - currency symbol with multipliers
            r"(?:\$|USD)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(MM|M|B|K|million|billion|thousand)?",
            # Pattern 2: 20MM, 5M, 10B - standalone with multipliers
            r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(MM|M|B|K|million|billion|thousand)\b",
            # Pattern 3: Standard formats - $7000, $70000, $2,000, $1,500.50
            r"(?:\$|USD)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)",
            # Pattern 4: Number followed by currency word
            r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:dollars?|USD)",
            # Pattern 5: Euro formats
            r"(?:€|EUR)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)",
            r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:€|EUR)",
            # Pattern 6: GBP formats
            r"(?:£|GBP)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)",
            r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:£|GBP)",
        ]

        for i, pattern in enumerate(money_patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Extract base amount
                    amount_str = match.group(1).replace(",", "")
                    amount = float(amount_str)

                    # Apply multiplier if present (for patterns 1 and 2)
                    if i <= 1 and len(match.groups()) > 1 and match.group(2):
                        multiplier = self._get_multiplier_with_financial(match.group(2))
                        amount *= multiplier

                    # Determine currency
                    currency = "USD"
                    matched_text = match.group(0).upper()
                    if "€" in matched_text or "EUR" in matched_text:
                        currency = "EUR"
                    elif "£" in matched_text or "GBP" in matched_text:
                        currency = "GBP"

                    return {
                        "$class": "org.accordproject.money.MonetaryAmount",
                        "doubleValue": round(amount, 2),
                        "currencyCode": currency,
                    }
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_duration(self, text: str) -> Optional[Dict]:
        """Extract duration with improved number parsing"""

        # Enhanced patterns to handle complex formats like "ten (10) year"
        enhanced_patterns = [
            # "ten (10) year", "four (4) additional ten (10) year"
            r"(?:(?:four|one|two|three|five|six|seven|eight|nine|ten|eleven|twelve)\s*\(\d+\)\s+(?:additional\s+)?)?(?:ten|one|two|three|four|five|six|seven|eight|nine|eleven|twelve)\s*\((\d+)\)\s+(year|month|week|day)s?",
            # "10 years", "5 days"
            r"(\d+)\s+(calendar\s+)?(days?|weeks?|months?|years?)",
            # "ten years", "five days" (word format)
            r"\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\s+(?:days?|weeks?|months?|years?))\b",
            # "10-year", "5-day"
            r"(\d+)-(?:day|week|month|year)",
        ]

        for i, pattern in enumerate(enhanced_patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if i == 0:  # Enhanced pattern for "ten (10) year"
                        amount = int(match.group(1))  # Extract from parentheses
                        unit = match.group(2).capitalize().rstrip("s") + "s"
                    elif i == 1:  # Numeric format "10 years"
                        amount, calendar_modifier, unit = match.groups()
                        amount = int(amount)
                        unit = unit.capitalize().rstrip("s") + "s"
                    elif i == 2:  # Word format "ten years"
                        duration_text = match.group(1)
                        parts = duration_text.split()
                        if len(parts) >= 2:
                            amount_word = parts[0]
                            unit = parts[1]
                            amount = w2n.word_to_num(amount_word)
                            unit = unit.capitalize().rstrip("s") + "s"
                        else:
                            continue
                    else:  # Hyphenated format "10-year"
                        amount = int(match.group(1))
                        unit_part = match.group(0).split("-")[1]
                        unit = unit_part.capitalize().rstrip("s") + "s"

                    # Validate unit
                    if unit.lower() not in ["days", "weeks", "months", "years"]:
                        continue

                    return {
                        "$class": "org.accordproject.time@0.3.0.Duration",
                        "amount": amount,
                        "unit": unit,
                    }
                except (ValueError, AttributeError):
                    continue

        return None

    def _extract_percentage(self, text: str) -> Optional[float]:
        """Extract percentage as decimal - improved to avoid section numbers"""

        # Enhanced patterns that explicitly look for percentage indicators
        enhanced_patterns = [
            # Explicit percentage with % symbol: "15%", "15.5%"
            r"(\d+(?:\.\d+)?)\s*%",
            # Written percentage: "fifteen percent", "15 percent"
            r"(\d+(?:\.\d+)?)\s+percent(?:\s|\b)",
            # Written percentage: "15 per cent"
            r"(\d+(?:\.\d+)?)\s+per\s+cent(?:\s|\b)",
            # Parenthetical percentage: "fifteen percent (15%)"
            r"(?:percent|per\s+cent)\s*\((\d+(?:\.\d+)?)\s*%\)",
            # Reverse parenthetical: "(15%) percent"
            r"\((\d+(?:\.\d+)?)\s*%\)\s*(?:percent|per\s+cent)",
            # Written numbers followed by percent: "fifteen (15) percent"
            r"(?:percent|per\s+cent)\s*\((\d+(?:\.\d+)?)\)",
        ]

        for pattern in enhanced_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    percentage = float(match.group(1))

                    # Validate this is actually a percentage context
                    if self._is_valid_percentage_context(match, text):
                        return round(percentage / 100.0, 4)

                except (ValueError, TypeError):
                    continue

        return None

    def _is_valid_percentage_context(self, match, text: str) -> bool:
        """Validate that a number is actually a percentage, not a section reference"""

        # Get surrounding context (50 chars before and after)
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end].lower()

        # Exclude if this looks like a section reference
        section_indicators = [
            r"section\s+\d+(?:\.\d+)*",
            r"clause\s+\d+(?:\.\d+)*",
            r"paragraph\s+\d+(?:\.\d+)*",
            r"article\s+\d+(?:\.\d+)*",
            r"subsection\s+\d+(?:\.\d+)*",
            r"\d+\.\d+\([a-z]\)",  # Like "6.3(e)"
            r"exhibit\s+[a-z0-9]+",
            r"schedule\s+[a-z0-9]+",
            r"appendix\s+[a-z0-9]+",
        ]

        # Check if the match is part of a section reference
        for pattern in section_indicators:
            if re.search(pattern, context):
                section_match = re.search(pattern, context)
                if section_match:
                    # If our percentage is within 10 characters of a section reference, likely part of it
                    if abs(section_match.start() - (match.start() - start)) < 10:
                        return False

        # EXPANDED: Revenue/financial context indicators
        percentage_indicators = [
            # Core percentage indicators
            "percent",
            "per cent",
            "%",
            # Traditional revenue sharing terms
            "revenue",
            "profit",
            "share",
            "sharing",
            "royalty",
            "commission",
            "fee",
            "portion",
            "equity",
            "ownership",
            "interest",
            "stake",
            "split",
            "distribution",
            # Payment/compensation terms
            "receive",
            "receives",
            "receiving",
            "paid",
            "pay",
            "payment",
            "payout",
            "compensation",
            "remuneration",
            "earn",
            "earned",
            "income",
            "proceeds",
            "generated",
            "originated",
            # Sales/business terms
            "advertising",
            "sales",
            "revenue",
            "gross",
            "net",
            "total",
            "amount",
            "sum",
            "value",
            "worth",
            # Allocation/division terms
            "allocated",
            "divided",
            "distributed",
            "shared",
            "split",
            "portion",
            "percentage",
            "fraction",
            "part",
            "share",
            # Contract/business context
            "agreement",
            "contract",
            "party",
            "parties",
            "shall",
            "entitled",
            "right",
            "obligation",
        ]

        # Must have at least one percentage/financial indicator nearby
        has_percentage_indicator = any(
            indicator in context for indicator in percentage_indicators
        )

        return has_percentage_indicator

    def _extract_integer_as_double(self, text: str) -> Optional[float]:
        """Extract integer value and return as Double (float) - improved to avoid section numbers"""

        # Skip if text contains obvious non-numeric contexts
        exclusion_patterns = [
            r"\d+[/-]\d+[/-]\d+",  # Dates
            r"[\$€£]\d+",  # Money
            r"\d+\s*%",  # Percentages
            r"section\s+\d+",  # Section references
            r"clause\s+\d+",  # Clause references
            r"paragraph\s+\d+",  # Paragraph references
            r"article\s+\d+",  # Article references
            r"exhibit\s+\d+",  # Exhibit references
            r"schedule\s+\d+",  # Schedule references
            r"table\s+\d+",  # Table references
            r"\d+\.\d+\([a-z]\)",  # Subsection references like "6.3(e)"
        ]

        text_lower = text.lower()
        for pattern in exclusion_patterns:
            if re.search(pattern, text_lower):
                return None

        # Find standalone numbers that aren't part of references
        number_pattern = r"\b(\d+)\b"
        matches = list(re.finditer(number_pattern, text))

        for match in matches:
            num_str = match.group(1)
            num = int(num_str)

            # Only consider reasonable standalone numbers
            if 1 <= num <= 999999:
                # Check context around this number
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].lower()

                # Skip if context suggests this is a reference number
                reference_indicators = [
                    "section",
                    "clause",
                    "paragraph",
                    "article",
                    "exhibit",
                    "schedule",
                    "table",
                    "appendix",
                    "subsection",
                    "part",
                    "chapter",
                ]

                # Skip if number is adjacent to reference indicators
                if any(indicator in context for indicator in reference_indicators):
                    continue

                # Skip if number is followed by decimal point (likely section number)
                if match.end() < len(text) and text[match.end()] == ".":
                    continue

                return float(num)

        return None

    def _extract_boolean(self, text: str, var_name: str) -> Optional[bool]:
        """Extract boolean value with consistent logic - no special cases"""
        text_lower = text.lower()

        # Strong negative indicators - universal across all boolean types
        strong_negative = [
            r"\bshall\s+not\b",
            r"\bmay\s+not\b",
            r"\bcannot\b",
            r"\bis\s+not\s+(?:required|applicable|permitted|subject)\b",
            r"\bno\s+(?:such|liability|cap|obligation|restrictions?)\b",
            r"\bexcluded\b",
            r"\bprohibited\b",
            r"\bnot\s+subject\s+to\b",
            r"\bexempt\s+from\b",
            r"\bdoes\s+not\s+(?:apply|exist|have|contain|include)\b",
            r"\bwithout\s+(?:any\s+)?(?:liability|obligation|restriction)\b",
            r"\bno\s+(?:minimum|maximum|limit|cap|restriction)\b",
        ]

        # Strong positive indicators - universal across all boolean types
        strong_positive = [
            r"\bshall\s+(?:be\s+)?(?:required|subject|liable|apply|have|contain|include)\b",
            r"\bmust\s+(?:be\s+|have\s+|contain\s+|include\s+)?\w+",
            r"\bis\s+required\s+to\b",
            r"\bmandatory\b",
            r"\bobligated\s+to\b",
            r"\bsubject\s+to\b",
            r"\bliable\s+for\b",
            r"\bapplies\s+to\b",
            r"\bin\s+effect\b",
            r"\benforceable\b",
            r"\bshall\s+(?:not\s+)?exceed\b",  # "shall not exceed" indicates restrictions exist
            r"\bmaximum\s+(?:of\s+)?\$",  # Maximum with dollar amount
            r"\bminimum\s+(?:of\s+)?\$",  # Minimum with dollar amount
            r"\bmust\s+not\s+exceed\b",  # "must not exceed" indicates restrictions
            r"\bcannot\s+exceed\b",  # "cannot exceed" indicates restrictions
            r"\blimited\s+to\b",  # "limited to" indicates restrictions
        ]

        # Check strong indicators first
        for pattern in strong_negative:
            if re.search(pattern, text_lower):
                return False

        for pattern in strong_positive:
            if re.search(pattern, text_lower):
                return True

        # If no strong indicators, return None (let LLM handle it)
        return None

    def _find_similar_clauses(
        self, contract_id: str, clause_type: str, text: str
    ) -> List[Dict]:
        """Find similar clauses for context"""

        if not self.similarity_matrix or not self.clause_metadata:
            return []

        clause_idx = None

        for idx, meta in enumerate(self.clause_metadata):
            if (
                meta.get("original_id") == contract_id
                and meta.get("predicted_clause_type", meta.get("clause_type"))
                == clause_type
            ):
                clause_idx = idx
                break

        if clause_idx is None:
            contract_base = (
                contract_id.split("__")[0] if "__" in contract_id else contract_id
            )
            for idx, meta in enumerate(self.clause_metadata):
                meta_id = meta.get("original_id", "")
                meta_type = meta.get("predicted_clause_type", meta.get("clause_type"))
                if contract_base in meta_id and meta_type == clause_type:
                    clause_idx = idx
                    break

        if clause_idx is None or clause_idx >= len(self.similarity_matrix):
            return []

        similarities = self.similarity_matrix[clause_idx]
        similar_indices = np.argsort(similarities)[::-1]

        similar_clauses = []
        for idx in similar_indices[:10]:
            if idx == clause_idx:
                continue

            if idx < len(self.clause_metadata):
                meta = self.clause_metadata[idx]
                meta_type = meta.get("predicted_clause_type", meta.get("clause_type"))

                if (
                    meta_type == clause_type
                    and similarities[idx] > 0.7
                    and meta.get("answer_text", "")
                ):

                    similar_clauses.append(
                        {
                            "similarity": float(similarities[idx]),
                            "text": meta.get("answer_text", "")[:200],
                            "metadata": meta,
                        }
                    )

                    if len(similar_clauses) >= 3:
                        break

        return similar_clauses

    def _query_llm_with_cache(self, prompt: str, var_name: str, var_type: str) -> any:
        """Query LLM with caching and improved settings"""

        # Use more of the prompt text in cache key to prevent cross-contamination
        prompt_hash = sha256(prompt.encode()).hexdigest()[:16]
        cache_key = f"{prompt_hash}|{var_name}|{var_type}"

        if cache_key in self.llm_cache:
            response = self.llm_cache[cache_key]
        else:
            try:
                llm_response = chat(
                    model="llama3",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise legal text extractor. Follow the instructions exactly and return only what is requested.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    options={
                        "temperature": 0.1,  # Slightly less conservative
                        "seed": 42,
                        "num_predict": 50,
                        "top_p": 0.9,  # Allow some randomness
                        "repeat_penalty": 1.0,
                    },
                )
                response = llm_response["message"]["content"].strip()
                self.llm_cache[cache_key] = response
                self._save_json_file(self.llm_cache_path, self.llm_cache)
            except Exception as e:
                logger.error(f"LLM extraction failed for {var_name}: {e}")
                return None

        return self._parse_response(response, var_type)

    def _parse_response(self, response: str, var_type: str) -> any:
        """Enhanced response parsing with better percentage handling and fixed money extraction"""

        if not response or response.lower().strip() in [
            "null",
            "none",
            "n/a",
            "not found",
            "not specified",
            "not mentioned",
            "not applicable",
            "not clear",
            "unclear",
        ]:
            return None

        # Remove any remaining fluff more aggressively
        response = response.strip().strip("\"'`").rstrip(".")

        # Aggressive fluff removal
        fluff_patterns = [
            r"^(?:the\s+)?(?:extracted\s+)?(?:value\s+(?:is|for)\s*:?\s*)",
            r"^(?:answer\s*:?\s*)",
            r"^(?:result\s*:?\s*)",
            r"^(?:based\s+on.*?:?\s*)",
            r"^(?:according\s+to.*?:?\s*)",
            r"^(?:from\s+the\s+text.*?:?\s*)",
            r"^(?:it\s+(?:appears|seems)\s+.*?:?\s*)",
            r"(?:appears?\s+to\s+be|seems?\s+to\s+be)",
            r"(?:based\s+on|according\s+to|from\s+the)",
            r"(?:the\s+text\s+(?:states|mentions|indicates))",
        ]

        original_response = response
        for pattern in fluff_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE).strip()

        # If we removed too much, it was probably all fluff
        if len(response) < len(original_response) * 0.3:
            return None

        response = response.strip("\"'`.,;:").strip()

        # Type-specific parsing
        if var_type == "String":
            # Reject if still contains meta-language
            meta_indicators = [
                "extract",
                "mention",
                "state",
                "indicate",
                "refer",
                "describe",
                "text",
                "document",
                "clause",
                "contract",
                "based",
                "according",
            ]
            if any(indicator in response.lower() for indicator in meta_indicators):
                return None

            # Must be substantial
            if len(response) < 3 or len(response) > 300:
                return None

            return response

        elif var_type == "Boolean":
            response_lower = response.lower()
            if any(word in response_lower for word in ["true", "yes", "present"]):
                return True
            elif any(word in response_lower for word in ["false", "no", "not"]):
                return False
            # Neutral rule: if response mentions "exceed", "limit", "cap", "restrict" - likely indicates restrictions exist
            elif any(
                word in response_lower
                for word in ["exceed", "limit", "cap", "restrict", "maximum", "minimum"]
            ):
                return True
            return None

        elif var_type == "DateTime":
            if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", response):
                return response
            try:
                parsed_date = dateparser.parse(response)
                if parsed_date:
                    return parsed_date.strftime("%Y-%m-%dT00:00:00.000Z")
            except:
                pass
            return None

        elif var_type == "Double":
            # Look for percentage indicators first
            if any(
                indicator in response.lower()
                for indicator in ["%", "percent", "per cent"]
            ):
                # Extract percentage and convert to decimal
                percentage_match = re.search(r"(\d+(?:\.\d+)?)", response)
                if percentage_match:
                    num = float(percentage_match.group(1))
                    # If number is > 1 and has percentage indicator, convert to decimal
                    if num > 1:
                        return round(num / 100.0, 4)
                    else:
                        return round(num, 4)
            else:
                # Regular number extraction - improved to capture full numbers
                numbers = re.findall(r"\d+(?:\.\d+)?", response.replace(",", ""))
                if numbers:
                    num = float(numbers[0])
                    return round(num, 4)
            return None

        elif var_type == "Duration":
            # Handle various duration formats
            duration_match = re.search(r"(\d+)\s+(\w+)", response)
            if duration_match:
                amount, unit = duration_match.groups()
                unit = unit.lower().rstrip("s") + "s"
                return {
                    "$class": "org.accordproject.time@0.3.0.Duration",
                    "amount": int(amount),
                    "unit": unit.capitalize(),
                }

            # Handle special cases like "term duration" or "until termination"
            if any(
                phrase in response.lower()
                for phrase in [
                    "term duration",
                    "until termination",
                    "until expiration",
                    "during the term",
                ]
            ):
                return {
                    "$class": "org.accordproject.time@0.3.0.Duration",
                    "amount": 1,
                    "unit": "Term",  # Special unit for term-based durations
                }

            return None

        elif var_type == "MonetaryAmount":
            # First try the enhanced extraction
            amount = self._extract_monetary_amount(response)
            if amount is not None:
                return {
                    "$class": "org.accordproject.money.MonetaryAmount",
                    "doubleValue": round(amount, 2),
                    "currencyCode": "USD",
                }

            # Fallback: try pattern-based extraction on the response
            money_result = self._extract_money(response)
            if money_result is not None:
                return money_result

            return None

        else:  # String and other types
            response = " ".join(response.split())  # Normalize whitespace
            return response if len(response) > 0 else None

    def _extract_monetary_amount(self, text: str) -> Optional[float]:
        """Extract monetary amount from LLM response - enhanced to handle multipliers but exclude percentages"""
        if not text or text.lower().strip() in ["null", "none", "n/a"]:
            return None

        text = text.lower().strip()

        # CRITICAL: Skip if this is clearly a percentage, not a monetary amount
        percentage_indicators = ["%", "percent", "per cent"]
        if any(indicator in text for indicator in percentage_indicators):
            # Only extract if there's also a clear monetary symbol/word
            monetary_indicators = ["$", "€", "£", "dollar", "usd", "eur", "gbp"]
            if not any(indicator in text for indicator in monetary_indicators):
                return None

        # Handle written patterns with multipliers - improved order and patterns
        multiplier_patterns = [
            # Pattern 1: $20MM, $5M, $10B style with currency symbols
            (
                r"[\$€£](\d+(?:\.\d+)?)\s*(mm|m|b|k|million|billion|thousand)?",
                lambda m: float(m.group(1))
                * self._get_multiplier_with_financial(m.group(2) or ""),
            ),
            # Pattern 2: 20MM, 5M, 10B - numbers with multiplier suffixes (but not percentages)
            (
                r"(\d+(?:\.\d+)?)\s*(mm|m|b|k|million|billion|thousand)\b(?!\s*%)",
                lambda m: float(m.group(1))
                * self._get_multiplier_with_financial(m.group(2)),
            ),
            # Pattern 3: "three times" - multiplier expressions (only if followed by monetary context)
            (
                r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+times.*?[\$€£](\d+(?:,\d{3})*(?:\.\d{1,2})?)",
                lambda m: float(w2n.word_to_num(m.group(1)))
                * float(m.group(2).replace(",", "")),
            ),
            # Pattern 4: Direct monetary extraction with currency symbols
            (
                r"[\$€£](\d+(?:,\d{3})*(?:\.\d{1,2})?)",
                lambda m: float(m.group(1).replace(",", "")),
            ),
            # Pattern 5: Number followed by explicit currency words
            (
                r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:dollars?|usd|euros?|eur|pounds?|gbp)",
                lambda m: float(m.group(1).replace(",", "")),
            ),
            # Pattern 6: "7 thousand dollars", "7.5 million dollars" - with currency context
            (
                r"(\d+(?:\.\d+)?)\s+(thousand|million|billion|hundred)\s*(?:dollars?|usd|euros?|eur|pounds?|gbp)",
                lambda m: float(m.group(1)) * self._get_multiplier(m.group(2)),
            ),
        ]

        for pattern, converter in multiplier_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    result = converter(match)
                    # Basic sanity check - amounts should be positive and reasonable
                    if result > 0 and result <= 1000000000000:  # Up to $1 trillion
                        return result
                except (ValueError, AttributeError):
                    continue

        # Try word2number for pure written numbers with monetary context
        try:
            # Only try if there are monetary indicators
            monetary_indicators = ["dollar", "usd", "euro", "eur", "pound", "gbp"]
            if any(indicator in text for indicator in monetary_indicators):
                # Clean up common monetary text
                clean_text = re.sub(
                    r"dollars?|usd|euro?s?|eur|pounds?|gbp|\$|€|£", "", text
                ).strip()
                clean_text = re.sub(r"amount|fee|cost|price", "", clean_text).strip()

                # Only try if it's pure words (no digits) and not too short
                if (
                    clean_text
                    and not re.search(r"\d", clean_text)
                    and len(clean_text.split()) > 1
                ):
                    result = float(w2n.word_to_num(clean_text))
                    if result > 0 and result <= 1000000000000:
                        return result
        except (ValueError, AttributeError):
            pass

        return None

    def _get_multiplier(self, word: str) -> float:
        """Get numeric multiplier for word"""
        multipliers = {
            "hundred": 100,
            "thousand": 1000,
            "million": 1000000,
            "billion": 1000000000,
        }
        return multipliers.get(word.lower(), 1)

    def _get_multiplier_enhanced(self, word: str) -> float:
        """Enhanced multiplier with financial abbreviations"""
        if not word:
            return 1

        word = word.lower().strip()
        multipliers = {
            "hundred": 100,
            "thousand": 1000,
            "k": 1000,
            "million": 1000000,
            "m": 1000000,
            "mm": 1000000,  # Financial notation: MM = million
            "billion": 1000000000,
            "b": 1000000000,
        }
        return multipliers.get(word, 1)

    def extract_all_contracts(self, max_contracts: Optional[int] = None) -> List[Dict]:
        """Main extraction method with mode-based defaults"""

        # Use provided max_contracts or fall back to mode default
        contracts_to_process = (
            max_contracts if max_contracts is not None else self.max_contracts
        )

        logger.info(
            f"Starting Variable Extraction in {self.mode.upper()} mode (max {contracts_to_process} contracts)"
        )

        contract_coverage = self.discover_contracts()
        if not contract_coverage:
            logger.error("No contracts found")
            return []

        selected_contracts = self.select_contracts(
            contract_coverage, contracts_to_process
        )
        if not selected_contracts:
            logger.error("No contracts selected")
            return []

        all_results = []
        successful_contracts = 0

        # Track processed contract-clause combinations to avoid duplicates
        processed_combinations = set()

        for i, contract_id in enumerate(
            tqdm(selected_contracts, desc=f"Extracting ({self.mode} mode)")
        ):
            logger.info(
                f"[{i+1}/{len(selected_contracts)}] Processing: {contract_id[:50]}..."
            )

            try:
                contract_results = self.extract_contract_variables(contract_id)

                # Deduplicate results
                unique_results = []
                for result in contract_results:
                    combination_key = f"{result['contract_id']}||{result['clause_type']}||{hash(result['original_text'][:200])}"
                    if combination_key not in processed_combinations:
                        processed_combinations.add(combination_key)
                        unique_results.append(result)

                if unique_results:
                    all_results.extend(unique_results)
                    successful_contracts += 1

                    total_variables = sum(len(r["extractions"]) for r in unique_results)
                    successful_variables = sum(
                        len([v for v in r["extractions"].values() if v is not None])
                        for r in unique_results
                    )
                    success_rate = (
                        successful_variables / total_variables
                        if total_variables > 0
                        else 0
                    )

                    logger.info(
                        f"{len(unique_results)} unique clauses, {successful_variables}/{total_variables} variables extracted ({success_rate:.1%})"
                    )
                else:
                    logger.info(f"No unique variables extracted")

            except Exception as e:
                logger.error(f"Error processing contract: {e}")
                continue

        logger.info(
            f"Extraction complete: {len(all_results)} unique extractions from {successful_contracts}/{len(selected_contracts)} contracts"
        )
        return all_results

    def save_results(self, results: List[Dict], filename: Optional[str] = None) -> str:
        """Save results with mode-based filename"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extracted_variables_{self.mode}_{timestamp}.json"

        output_path = self.outputs_dir / filename

        # Add metadata about the run
        output_data = {
            "metadata": {
                "mode": self.mode,
                "max_contracts": self.max_contracts,
                "total_extractions": len(results),
                "timestamp": datetime.now().isoformat(),
                "has_legalbert": self.has_legalbert,
            },
            "extractions": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")
        return str(output_path)

    def _load_json_file(self, path: Path, default: any = None) -> any:
        """Load JSON file with error handling"""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save_json_file(self, path: Path, data: any):
        """Save JSON file with error handling"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")


def main():
    """Main execution with argument parsing"""

    parser = argparse.ArgumentParser(description="Legal Variable Extractor")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["default", "production", "test", "debug"],
        default="default",
        help="Extraction mode (default: default)",
    )
    parser.add_argument(
        "--contracts", "-c", type=int, help="Override number of contracts to process"
    )
    parser.add_argument(
        "--data-dir", "-d", default="data", help="Data directory path (default: data)"
    )

    args = parser.parse_args()

    print("Legal Variable Extractor - Enhanced Party Extraction")
    print("=" * 50)

    extractor = LegalVariableExtractor(data_dir=args.data_dir, mode=args.mode)

    results = extractor.extract_all_contracts(max_contracts=args.contracts)

    if results:
        contracts = defaultdict(list)
        for result in results:
            contracts[result["contract_id"]].append(result)

        total_variables = sum(len(r["extractions"]) for r in results)
        successful_variables = sum(
            len([v for v in r["extractions"].values() if v is not None])
            for r in results
        )

        # Count party extraction success specifically
        party_extractions = [r for r in results if r["clause_type"] == "parties"]
        party_success = sum(
            1 for r in party_extractions if r["extractions"].get("parties") is not None
        )

        print(f"\nExtraction Summary ({args.mode} mode):")
        print(f"  Total contracts: {len(contracts)}")
        print(f"  Total extractions: {len(results)}")
        print(f"  Total variables: {total_variables}")
        print(f"  Successful variables: {successful_variables}")
        print(f"  Success rate: {successful_variables/total_variables:.1%}")
        print(
            f"  Party extractions: {party_success}/{len(party_extractions)} ({party_success/len(party_extractions)*100 if party_extractions else 0:.1f}%)"
        )

        output_file = extractor.save_results(results)
        print(f"\nResults saved to: {output_file}")

    else:
        print("No results generated. Check logs for issues.")


if __name__ == "__main__":
    main()
