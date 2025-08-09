"""
Template Generator for Accord Project Smart Legal Contracts
Module: src.generation.template_generator
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from collections import defaultdict
from dateutil import parser as dateparser
from hashlib import sha256
from slugify import slugify
import logging
from datetime import datetime
import sys
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging with consistent format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TemplateStats:
    """Statistics tracking for template generation runs."""

    contracts_processed: int = 0
    contracts_successful: int = 0
    templates_generated: int = 0
    clauses_skipped_null: int = 0
    clauses_skipped_no_mapping: int = 0
    original_text_preserved: int = 0
    fallback_templates_generated: int = 0
    double_replacements: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict:
        return {
            "contracts_processed": self.contracts_processed,
            "contracts_successful": self.contracts_successful,
            "templates_generated": self.templates_generated,
            "clauses_skipped_null": self.clauses_skipped_null,
            "clauses_skipped_no_mapping": self.clauses_skipped_no_mapping,
            "original_text_preserved": self.original_text_preserved,
            "fallback_templates_generated": self.fallback_templates_generated,
            "double_replacements": self.double_replacements,
            "duration_seconds": self.duration_seconds,
            "generation_rate": self.templates_generated / max(self.duration_seconds, 1),
            "success_rate": self.contracts_successful
            / max(self.contracts_processed, 1),
        }


class TemplateGenerator:
    """
    UPDATED template generator that preserves original legal text
    and converts it to proper Accord Project templates with correct CiceroMark syntax.
    Now uses unified Double type system for all numeric values.
    """

    def __init__(
        self,
        mode: str = "default",
        config_dir: str = "config",
        output_dir: str = "outputs/generated_templates",
    ):
        """Initialize the template generator."""

        # ========== PROJECT STRUCTURE SETUP ==========
        self.project_root = PROJECT_ROOT
        self.config_dir = self.project_root / config_dir
        self.output_dir = self.project_root / output_dir
        self.mode = mode.lower()

        # Ensure output directory follows project structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ========== STATISTICS TRACKING ==========
        self.stats = TemplateStats()

        logger.info(f"Initialized Template Generator (mode: {self.mode})")
        logger.info(f"Config dir: {self.config_dir}")
        logger.info(f"Output dir: {self.output_dir}")

        # ========== LOAD CONFIGURATIONS ==========
        self._load_configurations()

        # ========== SETUP MAPPING STRUCTURES ==========
        self._setup_mappings()

        # ========== LOAD GENERATION RULES ==========
        self._load_generation_rules()

        # ========== SETUP PATTERN RECOGNITION ==========
        self._setup_pattern_recognition()

    def _setup_pattern_recognition(self):
        """Setup comprehensive pattern recognition for doubles and percentages."""

        # Word to number mapping for comprehensive recognition
        self.word_to_number = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "twenty-one": 21,
            "twenty-two": 22,
            "twenty-three": 23,
            "twenty-four": 24,
            "twenty-five": 25,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
            "hundred": 100,
            "thousand": 1000,
        }

        # Number to word mapping
        self.number_to_word = {v: k for k, v in self.word_to_number.items()}

    def _load_configurations(self):
        """Load all configuration files with smart auto-discovery."""

        # Auto-discover configuration files by pattern matching
        config_patterns = {
            "variable_targets": [
                "variable_targets.json",
                "variables.json",
                "targets.json",
            ],
            "template_mapping": [
                "template_mapping*.json",
                "template_map*.json",
                "clause_map*.json",
                "*mapping*.json",
            ],
        }

        self.configs = {}

        for config_name, patterns in config_patterns.items():
            config_file = self._find_config_file(patterns)

            if not config_file:
                # List what we actually found to help debugging
                found_files = [f.name for f in self.config_dir.glob("*.json")]
                raise FileNotFoundError(
                    f"Could not find {config_name} configuration file.\n"
                    f"Searched for patterns: {patterns}\n"
                    f"In directory: {self.config_dir}\n"
                    f"Found JSON files: {found_files}\n"
                    f"Please ensure you have the required configuration files."
                )

            self.configs[config_name] = self._load_json_file(config_file)
            logger.info(
                f"✓ Auto-discovered {config_name}: {config_file.name} ({len(self.configs[config_name])} entries)"
            )

        # Extract main configuration objects
        self.variable_targets = self.configs["variable_targets"]
        self.template_mapping = self.configs["template_mapping"]

    def _find_config_file(self, patterns: List[str]) -> Optional[Path]:
        """Find configuration file using pattern matching."""

        for pattern in patterns:
            # Try glob pattern matching
            matches = list(self.config_dir.glob(pattern))
            if matches:
                # Return the first match, but prefer more specific names
                matches.sort(
                    key=lambda x: len(x.name)
                )  # Shorter names first (more specific)
                logger.debug(f"Found config file: {matches[0]} (pattern: {pattern})")
                return matches[0]

        return None

    def _setup_mappings(self):
        """Create efficient lookup structures for processing."""

        # Create reverse mapping: variable -> clause_type
        self.var_to_clause = {}
        for clause_type, variables in self.variable_targets.items():
            for var_name in variables:
                self.var_to_clause[var_name] = clause_type

        # Extract clause-to-template mapping
        self.clause_to_template = self.template_mapping.get(
            "clause_type_to_template", {}
        )

        # Create priority mapping for intelligent processing
        self.template_priorities = self.template_mapping.get("template_priorities", {})
        self.priority_order = ["essential", "high", "medium", "low"]

        logger.info(f"✓ Created mappings for {len(self.var_to_clause)} variables")
        logger.info(f"✓ Template priorities: {list(self.template_priorities.keys())}")

    def _load_generation_rules(self):
        """Load generation rules and quality settings from template mapping."""

        self.composition_rules = self.template_mapping.get(
            "contract_composition_rules", {}
        )
        self.validation_rules = self.template_mapping.get("variable_validation", {})
        self.quality_settings = self.template_mapping.get("generation_quality", {})

        # Set quality thresholds
        self.min_coverage = self.quality_settings.get(
            "completeness_thresholds", {}
        ).get("minimum_variable_coverage", 0.1)
        self.preferred_coverage = self.quality_settings.get(
            "completeness_thresholds", {}
        ).get("recommended_variable_coverage", 0.3)

        logger.info(
            f"✓ Quality thresholds - Min: {self.min_coverage}, Preferred: {self.preferred_coverage}"
        )

    def generate_template_grammar(
        self, original_text: str, clause_type: str, extractions: Dict[str, Any]
    ) -> str:
        """
        Generate proper CiceroMark template grammar using {{variableName}} format.
        Note: Double type handles both integers (5.0) and decimals (0.25) per Accord Project standards.
        """

        logger.debug(
            f"Generating template grammar for {clause_type} using original text"
        )

        if not original_text or not original_text.strip():
            logger.warning(
                f"No original text available for {clause_type}, generating fallback template"
            )
            self.stats.fallback_templates_generated += 1
            return self._generate_fallback_template(clause_type, extractions)

        # Start with the original legal text
        template_text = original_text.strip()

        # Get schema for this clause type
        schema = self.variable_targets.get(clause_type, {})
        if not schema:
            logger.warning(f"No schema found for clause type: {clause_type}")
            self.stats.original_text_preserved += 1
            return template_text

        # Replace extracted values with {{variableName}} placeholders
        replacements_made = 0

        # Process variables in order of complexity (longest first to avoid partial matches)
        sorted_vars = sorted(
            extractions.items(),
            key=lambda x: len(str(x[1])) if x[1] else 0,
            reverse=True,
        )

        for var_name, value in sorted_vars:
            if var_name not in schema or not self._is_usable_value(value):
                continue

            var_type = schema[var_name].get("type", "String")

            # Replace the value with template placeholder
            old_text = template_text
            template_text = self._replace_value_with_ciceromark_placeholder(
                template_text, var_name, value, var_type
            )

            if old_text != template_text:
                replacements_made += 1
                logger.debug(f"✓ Replaced {var_name} ({var_type}) in original text")

        logger.info(
            f"Generated template with {replacements_made} variable replacements from original text"
        )
        self.stats.original_text_preserved += 1

        return template_text

    def _replace_value_with_ciceromark_placeholder(
        self, text: str, var_name: str, value: Any, var_type: str
    ) -> str:
        """Replace extracted value with proper CiceroMark {{variableName}} placeholder."""

        if not self._is_usable_value(value):
            return text

        # Handle different data types with proper CiceroMark syntax
        if var_type == "DateTime" and isinstance(value, str):
            return self._replace_datetime_ciceromark(text, var_name, value)
        elif var_type == "MonetaryAmount" and isinstance(value, dict):
            return self._replace_monetary_ciceromark(text, var_name, value)
        elif var_type == "Duration" and isinstance(value, dict):
            return self._replace_duration_ciceromark(text, var_name, value)
        elif var_type == "Party[]" and isinstance(value, list):
            return self._replace_parties_ciceromark(text, var_name, value)
        elif var_type == "Boolean" and isinstance(value, bool):
            return self._replace_boolean_ciceromark(text, var_name, value)
        elif var_type == "Double":
            return self._replace_double_ciceromark(text, var_name, value)
        else:
            return self._replace_simple_ciceromark(text, var_name, value)

    def _replace_double_ciceromark(self, text: str, var_name: str, value: Any) -> str:
        """Replace double values with CiceroMark format - FIXED VERSION."""

        if isinstance(value, (int, float)):
            numeric_value = float(value)
        else:
            numeric_value = self._parse_numeric_value(str(value))
            if numeric_value is None:
                return text

        # Generate all possible representations
        representations = self._generate_all_numeric_representations(numeric_value)

        # Sort by length (longest first) to prioritize complete phrases
        representations.sort(key=len, reverse=True)

        # Try to replace each representation (longest matches first)
        for representation in representations:
            if representation in text:
                replacement = f"{{{{{var_name}}}}}"
                text = text.replace(representation, replacement, 1)
                self.stats.double_replacements += 1
                logger.debug(
                    f"Replaced complete phrase: '{representation}' -> '{replacement}'"
                )
                return text  # Stop after first successful replacement

        # Fallback to pattern-based replacement if no exact match
        return self._replace_numeric_patterns(text, var_name, numeric_value)

    def _parse_numeric_value(self, text: str) -> Optional[float]:
        """Parse numeric value from text, handling percentages and decimals."""

        text = text.strip().lower()

        # Direct numeric parsing
        try:
            if text.endswith("%"):
                return float(text[:-1]) / 100.0
            else:
                return float(text)
        except ValueError:
            pass

        # Word-based parsing
        if text in self.word_to_number:
            return float(self.word_to_number[text])

        # Handle compound words like "twenty-five"
        if "-" in text:
            parts = text.split("-")
            if len(parts) == 2 and all(part in self.word_to_number for part in parts):
                return float(
                    self.word_to_number[parts[0]] + self.word_to_number[parts[1]]
                )

        return None

    def _generate_all_numeric_representations(self, value: float) -> List[str]:
        """Generate all possible representations of a numeric value (including percentages AND integers)."""

        representations = []

        # For percentage context (detect if value is likely a percentage)
        if 0 < value <= 1:
            percentage = value * 100
            representations.extend(
                self._generate_percentage_representations(percentage)
            )

        # For regular numeric context (both integers and decimals)
        representations.extend(self._generate_numeric_representations(value))

        # For percentage context when value is already in percentage form
        if value > 1:
            representations.extend(self._generate_percentage_representations(value))

        return representations

    def _generate_percentage_representations(self, percentage: float) -> List[str]:
        """Generate all possible percentage representations - COMPLETE PHRASES."""

        representations = []

        # Handle whole vs decimal percentages
        if percentage == int(percentage):
            int_percentage = int(percentage)

            # Word + number formats - COMPLETE PHRASES
            word_rep = self.number_to_word.get(int_percentage)
            if word_rep:
                representations.extend(
                    [
                        f"{word_rep} percent ({int_percentage}%)",  # "one percent (1%)"
                        f"{word_rep} percent ({int_percentage})",  # "one percent (1)"
                        f"{word_rep} ({int_percentage}%)",  #  "one (1%)"
                        f"{word_rep} ({int_percentage}) percent",  # "one (1) percent"
                        f"{int_percentage} ({word_rep}) percent",  # "1 (one) percent"
                        f"{int_percentage} ({word_rep}%)",  # "1 (one%)"
                        f"{word_rep} percent",  # "one percent"
                    ]
                )

            # Basic percentage formats
            representations.extend(
                [
                    f"{int_percentage}%",  # "1%"
                    f"({int_percentage}%)",  # "(1%)"
                    f"{int_percentage} percent",  # "1 percent"
                    f"({int_percentage}) percent",  # "(1) percent"
                ]
            )

        else:
            # Decimal percentage formats
            representations.extend(
                [
                    f"{percentage}%",  # "1.5%"
                    f"({percentage}%)",  # "(1.5%)"
                    f"{percentage} percent",  # "1.5 percent"
                    f"({percentage}) percent",  # "(1.5) percent"
                ]
            )

        return representations

    def _generate_numeric_representations(self, value: float) -> List[str]:
        """Generate all possible numeric representations for both integers and decimals."""

        representations = []

        # Handle whole vs decimal numbers
        if value == int(value):  # It's a whole number like 5.0
            int_value = int(value)

            # Basic numeric formats for integers
            representations.extend(
                [
                    str(int_value),  # "5"
                    f"({int_value})",  # "(5)"
                    f" {int_value} ",  # " 5 "
                    f"{int_value}.0",  # "5.0"
                    f"{int_value}.00",  # "5.00"
                ]
            )

            # Word representations for integers
            word_rep = self.number_to_word.get(int_value)
            if word_rep:
                representations.extend(
                    [
                        word_rep,  # "five"
                        word_rep.title(),  # "Five"
                        f"({word_rep})",  # "(five)"
                        f" {word_rep} ",  # " five "
                        f"{word_rep} ({int_value})",  # "five (5)"
                        f"{int_value} ({word_rep})",  # "5 (five)"
                    ]
                )
        else:
            # Decimal representations
            representations.extend(
                [
                    str(value),  # "5.25"
                    f"({value})",  # "(5.25)"
                    f" {value} ",  # " 5.25 "
                    f"{value:.1f}",  # "5.3"
                    f"{value:.2f}",  # "5.25"
                ]
            )

        return representations

    def _replace_numeric_patterns(self, text: str, var_name: str, value: float) -> str:
        """Replace numeric patterns using regex when exact match fails - COMPREHENSIVE VERSION."""

        working_text = text

        # ========== PERCENTAGE PATTERNS ==========

        # Pattern for "word percent (number%)" - REPLACE ENTIRE PHRASE
        def replace_percentage_pattern(match):
            word_part = match.group(1).lower()
            num_part = match.group(2)

            # Check if this matches our value
            word_value = self.word_to_number.get(word_part.replace("-", ""))
            num_value = float(num_part) / 100.0

            # If either the word OR number matches our value, replace ENTIRE phrase
            if (word_value and word_value / 100.0 == value) or (
                abs(num_value - value) < 0.001
            ):
                return f"{{{{{var_name}}}}}"

            return match.group(0)

        # Apply percentage pattern replacement - REPLACES ENTIRE "word percent (number%)" phrase
        percentage_pattern = r"(\w+(?:-\w+)?)\s+percent\s*\((\d+(?:\.\d+)?)\%\)"
        working_text = re.sub(
            percentage_pattern,
            replace_percentage_pattern,
            working_text,
            flags=re.IGNORECASE,
        )

        # Pattern for simple percentages like "1%" or "(25%)"
        def replace_simple_percentage(match):
            num_part = match.group(1)
            num_value = float(num_part) / 100.0

            if abs(num_value - value) < 0.001:
                return f"{{{{{var_name}}}}}"

            return match.group(0)

        # Handle both "1%" and "(1%)" patterns
        simple_percentage_pattern = r"\(?(\d+(?:\.\d+)?)\%\)?"
        working_text = re.sub(
            simple_percentage_pattern,
            replace_simple_percentage,
            working_text,
            flags=re.IGNORECASE,
        )

        # Pattern for "word percent" without parentheses
        def replace_word_percentage(match):
            word_part = match.group(1).lower()
            word_value = self.word_to_number.get(word_part.replace("-", ""))

            if word_value and word_value / 100.0 == value:
                return f"{{{{{var_name}}}}}"

            return match.group(0)

        word_percentage_pattern = r"(\w+(?:-\w+)?)\s+percent"
        working_text = re.sub(
            word_percentage_pattern,
            replace_word_percentage,
            working_text,
            flags=re.IGNORECASE,
        )

        # ========== INTEGER/NUMBER PATTERNS ==========

        if value == int(value):
            int_value = int(value)
            word_rep = self.number_to_word.get(int_value)

            if word_rep:
                # Pattern 1: "word (number)" - like "five (5)" or "twenty-five (25)"
                def replace_word_number_pattern(match):
                    word_part = (
                        match.group(1).lower().replace("-", "-")
                    )  # Preserve hyphens
                    num_part = match.group(2)

                    # Handle hyphenated numbers like "twenty-five"
                    word_normalized = word_part.replace(" ", "-")

                    if (
                        word_normalized == word_rep.lower()
                        and int(num_part) == int_value
                    ):
                        return f"{{{{{var_name}}}}}"
                    return match.group(0)

                # Flexible pattern for "word (number)" with optional spaces and hyphens
                word_number_patterns = [
                    rf"({re.escape(word_rep)})\s*\((\d+)\)",  # "five (5)"
                    rf"({re.escape(word_rep.title())})\s*\((\d+)\)",  # "Five (5)"
                    rf"({re.escape(word_rep.upper())})\s*\((\d+)\)",  # "FIVE (5)"
                ]

                for pattern in word_number_patterns:
                    working_text = re.sub(
                        pattern,
                        replace_word_number_pattern,
                        working_text,
                        flags=re.IGNORECASE,
                    )

                # Pattern 2: "number (word)" - like "5 (five)" or "25 (twenty-five)"
                def replace_number_word_pattern(match):
                    num_part = match.group(1)
                    word_part = match.group(2).lower().replace("-", "-")

                    if int(num_part) == int_value and word_part == word_rep.lower():
                        return f"{{{{{var_name}}}}}"
                    return match.group(0)

                # Pattern: "5 (five)" with various capitalizations
                number_word_patterns = [
                    rf"(\d+)\s*\(({re.escape(word_rep)})\)",  # "5 (five)"
                    rf"(\d+)\s*\(({re.escape(word_rep.title())})\)",  # "5 (Five)"
                    rf"(\d+)\s*\(({re.escape(word_rep.upper())})\)",  # "5 (FIVE)"
                ]

                for pattern in number_word_patterns:
                    working_text = re.sub(
                        pattern,
                        replace_number_word_pattern,
                        working_text,
                        flags=re.IGNORECASE,
                    )

                # Pattern 3: Standalone word numbers - "five" or "Five"
                def replace_standalone_word(match):
                    word_part = match.group(1).lower()
                    if word_part == word_rep.lower():
                        return f"{{{{{var_name}}}}}"
                    return match.group(0)

                # Use word boundaries to avoid partial matches
                standalone_word_patterns = [
                    rf"\b({re.escape(word_rep)})\b",  # "five"
                    rf"\b({re.escape(word_rep.title())})\b",  # "Five"
                    rf"\b({re.escape(word_rep.upper())})\b",  # "FIVE"
                ]

                for pattern in standalone_word_patterns:
                    working_text = re.sub(
                        pattern,
                        replace_standalone_word,
                        working_text,
                        flags=re.IGNORECASE,
                    )

        # ========== GENERAL NUMERIC PATTERNS ==========

        # Pattern for standalone numbers (with word boundaries)
        def replace_standalone_number(match):
            num_part = match.group(1)
            try:
                found_value = float(num_part)
                if abs(found_value - value) < 0.001:
                    return f"{{{{{var_name}}}}}"
            except ValueError:
                pass
            return match.group(0)

        # Match standalone numbers with word boundaries
        standalone_number_pattern = r"\b(\d+(?:\.\d+)?)\b"
        working_text = re.sub(
            standalone_number_pattern, replace_standalone_number, working_text
        )

        # Pattern for numbers in parentheses - "(5)" or "(5.0)"
        def replace_parenthetical_number(match):
            num_part = match.group(1)
            try:
                found_value = float(num_part)
                if abs(found_value - value) < 0.001:
                    return f"{{{{{var_name}}}}}"
            except ValueError:
                pass
            return match.group(0)

        parenthetical_number_pattern = r"\((\d+(?:\.\d+)?)\)"
        working_text = re.sub(
            parenthetical_number_pattern, replace_parenthetical_number, working_text
        )

        if working_text != text:
            logger.debug(
                f"Numeric pattern replacement successful for {var_name} = {value}"
            )
            self.stats.double_replacements += 1

        return working_text

    def _replace_datetime_ciceromark(self, text: str, var_name: str, value: str) -> str:
        """Replace datetime with CiceroMark format."""

        try:
            target_date = dateparser.parse(value)
            if not target_date:
                return text
        except:
            return text

        # Look for date patterns and replace with simple placeholder
        date_patterns = [
            r"\b\d{1,2}(st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+,?\s+\d{4}\b",
            r"\b[A-Za-z]+\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        ]

        for pattern in date_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                date_text = match.group(0)
                try:
                    found_date = dateparser.parse(date_text)
                    if found_date and found_date.date() == target_date.date():
                        return text.replace(date_text, f"{{{{{var_name}}}}}", 1)
                except:
                    continue
        return text

    def _replace_monetary_ciceromark(
        self, text: str, var_name: str, value: Dict
    ) -> str:
        """Replace monetary amounts with proper CiceroMark format for MonetaryAmount objects."""

        amount = value.get("doubleValue")
        currency = value.get("currencyCode")

        if not amount:
            return text

        # Auto-detect currency from text if not provided
        if not currency:
            currency = self._detect_currency_from_text(text, amount)

        # Comprehensive currency symbol mapping
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CNY": "¥",
            "RMB": "¥",
            "CAD": "$",
            "AUD": "$",
            "NZD": "$",
            "HKD": "$",
            "SGD": "$",
            "CHF": "CHF",
            "SEK": "kr",
            "NOK": "kr",
            "DKK": "kr",
            "PLN": "zł",
            "CZK": "Kč",
            "HUF": "Ft",
            "RUB": "₽",
            "INR": "₹",
            "KRW": "₩",
            "THB": "฿",
            "MYR": "RM",
            "IDR": "Rp",
            "PHP": "₱",
            "VND": "₫",
            "TRY": "₺",
            "ILS": "₪",
            "SAR": "﷼",
            "AED": "د.إ",
            "EGP": "£",
            "ZAR": "R",
            "NGN": "₦",
            "KES": "KSh",
            "GHS": "GH₵",
            "BWP": "P",
            "BRL": "R$",
            "ARS": "$",
            "CLP": "$",
            "COP": "$",
            "PEN": "S/",
            "MXN": "$",
            "GTQ": "Q",
            "CRC": "₡",
            "PAB": "B/.",
            "DOP": "RD$",
        }
        symbol = currency_symbols.get(currency, currency or "$")

        # Format amount variations
        amount_int = int(amount) if amount == int(amount) else amount
        amount_formatted = (
            f"{amount_int:,}" if amount == int(amount) else f"{amount:,.2f}"
        )
        amount_with_decimals = f"{amount:.2f}"

        working_text = text

        # Proper CiceroMark syntax for MonetaryAmount (separates doubleValue and currencyCode)
        replacement = f"{{{{{var_name}.doubleValue}}}} {{{{{var_name}.currencyCode}}}}"

        # Monetary patterns to match (including .00 suffix variations)
        patterns = [
            f'"{symbol}{amount_formatted}.00"',  # "$2,000,000.00"
            f'"{symbol}{amount_with_decimals}.00"',  # "$2,000,000.00.00" (edge case)
            f"{symbol}{amount_formatted}.00",  # $2,000,000.00
            f"{symbol}{amount_with_decimals}.00",  # $2,000,000.00.00 (edge case)
            f'"{symbol}{amount_formatted}"',  # "$2,000,000"
            f'"{symbol}{amount_with_decimals}"',  # "$2,000,000.00"
            f"{symbol}{amount_formatted}",  # $2,000,000
            f"{symbol}{amount_with_decimals}",  # $2,000,000.00
            f'"{symbol}{amount_int}.00"',  # "$2000000.00"
            f"{symbol}{amount_int}.00",  # $2000000.00
            f'"{symbol}{amount_int}"',  # "$2000000"
            f"{symbol}{amount_int}",  # $2000000
            f"{amount_formatted}.00 {currency}",  # 2,000,000.00 USD
            f"{amount_with_decimals}.00 {currency}",  # 2,000,000.00.00 USD (edge case)
            f"{amount_formatted} {currency}",  # 2,000,000 USD
            f"{amount_with_decimals} {currency}",  # 2,000,000.00 USD
            f'"{amount_formatted}.00 {currency}"',  # "2,000,000.00 USD"
            f'"{amount_with_decimals}.00 {currency}"',  # "2,000,000.00.00 USD" (edge case)
            f'"{amount_formatted} {currency}"',  # "2,000,000 USD"
            f'"{amount_with_decimals} {currency}"',  # "2,000,000.00 USD"
        ]

        # Try exact pattern matching first
        for pattern in patterns:
            if pattern in working_text:
                working_text = working_text.replace(pattern, replacement, 1)
                return working_text

        # Fallback to regex patterns for flexible matching
        import re

        regex_patterns = [
            rf"\{re.escape(symbol)}{re.escape(str(amount_int))}(?:\.0{{1,2}})?(?=\s|[^\d]|$)",
            rf"\{re.escape(symbol)}{amount_int:,}(?:\.0{{1,2}})?(?=\s|[^\d]|$)",
            rf"\{re.escape(symbol)}{amount_with_decimals}(?:\.0{{1,2}})?(?=\s|[^\d]|$)",
            rf"{amount_int:,}(?:\.0{{1,2}})?\s+{currency}(?=\s|[^\w]|$)",
            rf"{amount_with_decimals}(?:\.0{{1,2}})?\s+{currency}(?=\s|[^\w]|$)",
        ]

        for pattern in regex_patterns:
            matches = list(re.finditer(pattern, working_text))
            if matches:
                for match in reversed(matches):
                    working_text = (
                        working_text[: match.start()]
                        + replacement
                        + working_text[match.end() :]
                    )
                return working_text

        return working_text

    def _detect_currency_from_text(self, text: str, amount: float) -> str:
        """Auto-detect currency from text context."""

        # Currency detection patterns (symbol -> code mapping)
        currency_patterns = {
            r"\$": "USD",  # Default $ to USD, could be other dollar currencies
            r"€": "EUR",
            r"£": "GBP",
            r"¥": "JPY",  # Could also be CNY
            r"₹": "INR",
            r"₽": "RUB",
            r"₩": "KRW",
            r"₪": "ILS",
            r"₺": "TRY",
            r"₦": "NGN",
            r"₫": "VND",
            r"₱": "PHP",
            r"฿": "THB",
            r"R\b": "ZAR",  # South African Rand
            r"R\$": "BRL",  # Brazilian Real
        }

        # Currency code patterns (explicit mentions)
        currency_codes = [
            "USD",
            "EUR",
            "GBP",
            "JPY",
            "CNY",
            "RMB",
            "CAD",
            "AUD",
            "NZD",
            "HKD",
            "SGD",
            "CHF",
            "SEK",
            "NOK",
            "DKK",
            "PLN",
            "CZK",
            "HUF",
            "RUB",
            "INR",
            "KRW",
            "THB",
            "MYR",
            "IDR",
            "PHP",
            "VND",
            "TRY",
            "ILS",
            "SAR",
            "AED",
            "EGP",
            "ZAR",
            "NGN",
            "KES",
            "GHS",
            "BWP",
            "BRL",
            "ARS",
            "CLP",
            "COP",
            "PEN",
            "MXN",
            "GTQ",
            "CRC",
            "PAB",
            "DOP",
        ]

        import re

        # Look for explicit currency codes first
        for code in currency_codes:
            if re.search(rf"\b{code}\b", text, re.IGNORECASE):
                return code.upper()

        # Look for currency symbols
        for pattern, code in currency_patterns.items():
            if re.search(pattern, text):
                return code

        # Default fallback
        return "USD"

    def _replace_duration_ciceromark(
        self, text: str, var_name: str, value: Dict
    ) -> str:
        """Replace duration with CiceroMark format."""

        amount = value.get("amount")
        unit = value.get("unit", "")

        if not amount or not unit:
            return text

        # Convert amount to written form for comprehensive matching
        number_words = {
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
            21: "twenty-one",
            22: "twenty-two",
            23: "twenty-three",
            24: "twenty-four",
            25: "twenty-five",
            30: "thirty",
            31: "thirty-one",
            32: "thirty-two",
            33: "thirty-three",
            34: "thirty-four",
            35: "thirty-five",
            40: "forty",
            45: "forty-five",
            48: "forty-eight",
            50: "fifty",
            60: "sixty",
            90: "ninety",
            100: "one hundred",
            120: "one hundred twenty",
            365: "three hundred sixty-five",
        }

        amount_word = number_words.get(amount, str(amount))
        unit_lower = unit.lower()
        unit_singular = unit_lower.rstrip("s")  # Remove plural 's'

        # Generate all possible number formats
        number_formats = [
            str(amount),
            amount_word,
            amount_word.title(),
            amount_word.upper(),
        ]

        # Generate all possible unit formats (both singular and plural)
        unit_formats = [
            unit_lower,  # "years"
            unit_singular,  # "year"
            unit_lower.title(),  # "Years"
            unit_singular.title(),  # "Year"
            unit_lower.upper(),  # "YEARS"
            unit_singular.upper(),  # "YEAR"
        ]

        # Common prefixes/contexts
        prefixes = [
            "",
            "within ",
            "after ",
            "before ",
            "for ",
            "during ",
            "period of ",
            "term of ",
            "additional ",
        ]

        # Different bracket/quote styles
        bracket_styles = [
            "{num} {unit}",  # "5 days"
            "{num} ({num}) {unit}",  # "five (5) days"
            "({num}) {unit}",  # "(5) days"
            '"{num} {unit}"',  # "5 days"
            "{num}-{unit}",  # "5-day"
            "{num} {unit} period",  # "5 year period"
            "{num}-{unit} period",  # "5-year period"
        ]

        # COMPLETE replacement with both amount and unit
        replacement = f"{{{{{var_name}.amount}}}} {{{{{var_name}.unit}}}}"

        # Try all combinations dynamically
        for prefix in prefixes:
            for style in bracket_styles:
                for num_format in number_formats:
                    for unit_format in unit_formats:
                        # Handle special case for written (numeric) format
                        if "{num} ({num})" in style:
                            pattern = style.format(
                                num=amount_word, unit=unit_format
                            ).replace(f"({amount_word})", f"({amount})")
                        else:
                            pattern = style.format(num=num_format, unit=unit_format)

                        full_pattern = prefix + pattern

                        # Check if this exact pattern exists in text
                        if full_pattern in text:
                            # For "period" patterns, include "period" in replacement
                            if "period" in full_pattern:
                                full_replacement = f"{{{{{var_name}.amount}}}} {{{{{var_name}.unit}}}} period"
                            else:
                                full_replacement = replacement

                            if prefix:
                                full_replacement = prefix + full_replacement

                            logger.debug(
                                f"Found duration pattern: '{full_pattern}' -> '{full_replacement}'"
                            )
                            return text.replace(full_pattern, full_replacement, 1)

        return text

    def _replace_parties_ciceromark(self, text: str, var_name: str, value: List) -> str:
        """Replace parties with CORRECT CiceroMark format - processes ALL parties."""

        working_text = text
        replacements_made = 0

        # Collect all party names for pattern matching
        party_names = []
        for party in value:
            if isinstance(party, dict):
                name = party.get("partyId", "").strip()
            else:
                name = str(party).strip()

            if name and len(name) > 2:
                party_names.append(name)

        if not party_names:
            logger.debug("No valid party names found for replacement")
            return working_text

        # Use "and" separator for natural legal language
        replacement = (
            f'{{{{#join {var_name} separator=" and "}}}}{{{{partyId}}}}{{{{/join}}}}'
        )

        # Try to find and replace party patterns
        multi_party_patterns = [
            # Pattern: "Party A" and "Party B"
            r'"([^"]+)"\s+and\s+"([^"]+)"',
            # Pattern: Party A and Party B (without quotes)
            r"\b("
            + "|".join(re.escape(name) for name in party_names)
            + r")\s+and\s+("
            + "|".join(re.escape(name) for name in party_names)
            + r")\b",
            # Pattern: "Party A", "Party B"
            r'"([^"]+)",\s*"([^"]+)"',
            # Pattern: Party A, Party B
            r"\b("
            + "|".join(re.escape(name) for name in party_names)
            + r"),\s*("
            + "|".join(re.escape(name) for name in party_names)
            + r")\b",
        ]

        # Try multi-party patterns first
        for pattern in multi_party_patterns:
            matches = list(re.finditer(pattern, working_text))
            if matches:
                # Replace the entire matched pattern with our CiceroMark template
                for match in reversed(matches):  # Reverse to maintain positions
                    working_text = (
                        working_text[: match.start()]
                        + replacement
                        + working_text[match.end() :]
                    )
                    replacements_made += 1
                    logger.debug(
                        f"✓ Replaced multi-party pattern: '{match.group()}' with CiceroMark join syntax"
                    )
                break  # Stop after first successful pattern match

        # If no multi-party patterns found, try individual party replacement
        if replacements_made == 0:
            for idx, name in enumerate(party_names):
                # Look for individual party name patterns
                patterns = [
                    f'"{name}"',  # "Company Name"
                    f"'{name}'",  # 'Company Name'
                    f"({name})",  # (Company Name)
                    f" {name} ",  # space Company Name space
                    f" {name},",  # space Company Name comma
                    f" {name}.",  # space Company Name period
                ]

                for pattern in patterns:
                    if pattern in working_text:
                        working_text = working_text.replace(pattern, replacement, 1)
                        replacements_made += 1
                        logger.debug(
                            f"✓ Replaced party {idx}: '{name}' with CiceroMark join syntax"
                        )
                        break  # Break inner loop once we find a match for this party

                if replacements_made > 0:
                    break  # Break outer loop once we make any replacement

        return working_text

    def _replace_boolean_ciceromark(self, text: str, var_name: str, value: bool) -> str:
        """Replace boolean with CiceroMark conditional."""

        if value:
            return f"{{{{#if {var_name}}}}}{text}{{{{/if}}}}"
        else:
            return f"{{{{#if {var_name}}}}}{{{{else}}}}{text}{{{{/if}}}}"

    def _replace_simple_ciceromark(self, text: str, var_name: str, value: Any) -> str:
        """Replace simple values with CiceroMark placeholder."""

        value_str = str(value).strip()
        if not value_str:
            return text

        patterns = [
            f'"{value_str}"',
            f"'{value_str}'",
            f"({value_str})",
            f" {value_str} ",
            value_str,
        ]

        for pattern in patterns:
            if pattern in text:
                return text.replace(pattern, f"{{{{{var_name}}}}}", 1)

        return text

    def _generate_fallback_template(
        self, clause_type: str, extractions: Dict[str, Any]
    ) -> str:
        """Generate fallback CiceroMark template."""

        lines = [f"## {clause_type.replace('_', ' ').title()}", ""]

        schema = self.variable_targets.get(clause_type, {})
        for var_name, value in extractions.items():
            if var_name in schema and self._is_usable_value(value):
                var_type = schema[var_name].get("type", "String")

                if var_type == "Boolean":
                    lines.extend(
                        [
                            f"{{{{#if {var_name}}}}}",
                            f"{var_name.replace('_', ' ').title()} applies.",
                            f"{{{{else}}}}",
                            f"{var_name.replace('_', ' ').title()} does not apply.",
                            f"{{{{/if}}}}",
                        ]
                    )
                elif var_type == "Party[]":
                    lines.append(
                        f'{{{{#join {var_name} separator=" and "}}}}{{{{partyId}}}}{{{{/join}}}}'
                    )
                elif var_type == "Double":
                    lines.append(
                        f"{var_name.replace('_', ' ').title()}: {{{{{var_name}}}}}"
                    )
                else:
                    lines.append(
                        f"{var_name.replace('_', ' ').title()}: {{{{{var_name}}}}}"
                    )

        return "\n".join(lines)

    def generate_sample_contract(
        self, original_text: str, clause_type: str, extractions: Dict[str, Any]
    ) -> str:
        """Generate sample contract using the original text with actual values."""

        logger.debug(f"Generating sample contract for {clause_type}")

        if not original_text or not original_text.strip():
            logger.warning(
                f"No original text available for {clause_type}, generating simple sample"
            )
            return self._generate_simple_sample(clause_type, extractions)

        # The original text IS the sample contract - it already has the real values
        sample_text = original_text.strip()

        # Add a header if it doesn't have one
        if not sample_text.startswith("#"):
            sample_text = f"## {clause_type.replace('_', ' ').title()}\n\n{sample_text}"

        return sample_text

    def _generate_simple_sample(
        self, clause_type: str, extractions: Dict[str, Any]
    ) -> str:
        """Generate a simple sample when original text is not available."""

        lines = [f"## {clause_type.replace('_', ' ').title()}", ""]

        # Add sample values
        schema = self.variable_targets.get(clause_type, {})
        for var_name, value in extractions.items():
            if var_name in schema and self._is_usable_value(value):
                var_type = schema[var_name].get("type", "String")
                formatted_value = self._format_value_for_sample(value, var_type)
                lines.append(f"{var_name.replace('_', ' ').title()}: {formatted_value}")

        return "\n".join(lines)

    def _format_value_for_sample(self, value: Any, var_type: str) -> str:
        """Format value for sample contract display."""

        if var_type == "MonetaryAmount" and isinstance(value, dict):
            amount = value.get("doubleValue")
            currency = value.get("currencyCode", "USD")
            if amount:
                return f"{amount} {currency}"

        elif var_type == "Duration" and isinstance(value, dict):
            amount = value.get("amount")
            unit = value.get("unit", "days")
            if amount:
                return f"{amount} {unit}"

        elif var_type == "Party[]" and isinstance(value, list):
            names = []
            for party in value:
                if isinstance(party, dict):
                    name = party.get("partyId", str(party))
                else:
                    name = str(party)
                names.append(f'"{name}"')
            return " and ".join(names)

        elif var_type == "Boolean":
            return "yes" if value else "no"

        elif var_type == "Double":
            # Handle both integers and decimals
            if isinstance(value, float) and value.is_integer():
                return str(int(value))  # Display "5" instead of "5.0" for whole numbers
            else:
                return str(value)

        elif var_type == "String":
            return f'"{value}"'

        return str(value)

    def generate_model_cto(
        self, clause_type: str, extractions: Dict[str, Any], template_id: str
    ) -> str:
        """Generate Accord Project v0.22+ compliant Concerto model."""

        template_info = self.clause_to_template.get(clause_type, {})
        template_name = template_info.get("template", template_id)

        # Generate dynamic class name
        class_name = self._generate_class_name(clause_type, template_id)

        # Create namespace following Accord Project conventions
        namespace = f"org.accordproject.{template_id.replace('-', '').lower()}"

        lines = [
            f"namespace {namespace}",
            "",
            f"// Generated template model for clause type: {clause_type}",
            f"// Template: {template_name}",
            f"// Model Class: {class_name}",
            f"// Priority: {self.get_template_priority(clause_type)}",
            "",
            "import org.accordproject.contract.* from https://models.accordproject.org/accordproject/contract.cto",
            "import org.accordproject.runtime.* from https://models.accordproject.org/accordproject/runtime.cto",
        ]

        # Add specialized imports using wildcards (official convention)
        needed_imports = self._get_needed_imports(extractions, clause_type)
        if needed_imports:
            lines.append("")
            lines.append("// Specialized Accord Project imports")
            lines.extend(needed_imports)

        # Start template model definition
        description = template_info.get(
            "description", f"Template model for {clause_type}"
        )
        lines.extend(
            [
                "",
                f"/**",
                f" * {description}",
                f" * Generated from extracted legal variables",
                f" */",
                f"asset {class_name} extends Clause {{",
            ]
        )

        # Add variable definitions
        schema = self.variable_targets.get(clause_type, {})
        var_count = 0

        for var_name, value in extractions.items():
            if var_name in schema and self._is_usable_value(value):
                var_config = schema[var_name]
                concerto_type = self._get_concerto_type(var_config.get("type"))

                if concerto_type:
                    lines.append(f"  o {concerto_type} {var_name} optional")
                    var_count += 1

        lines.append("}")

        # Add proper transaction definitions with current inheritance
        lines.extend(
            [
                "",
                "/**",
                " * Request transaction for template execution",
                " */",
                "transaction TemplateRequest extends Request {",
                "  o String input optional",
                "}",
                "",
                "/**",
                " * Response transaction for template execution",
                " */",
                "transaction TemplateResponse extends Response {",
                "  o String output optional",
                "}",
            ]
        )

        logger.debug(f"Generated model with {var_count} variables for {clause_type}")
        return "\n".join(lines)

    def _generate_class_name(self, clause_type: str, template_id: str) -> str:
        """Generate dynamic class name with intelligent fallbacks"""

        # Priority 1: Use template_id if it's descriptive
        if template_id and template_id != clause_type:
            candidate = self._format_template_name(template_id)
            if len(candidate) > 5 and candidate != "Clause":  # Avoid generic names
                return candidate

        # Priority 2: Use clause_type
        candidate = self._to_camel_case(clause_type)
        if candidate and len(candidate) > 3:
            if not candidate.endswith(("Clause", "Contract")):
                candidate += "Clause"
            return candidate

        # Fallback: Generic but unique
        return f"GeneratedClause{abs(hash(clause_type + template_id)) % 1000}"

    def _format_template_name(self, template_id: str) -> str:
        """Format template_id to proper class name"""
        if not template_id:
            return ""

        # Remove common suffixes/prefixes
        clean_name = (
            template_id.replace("@1.0.0", "")
            .replace("-clause", "")
            .replace("_clause", "")
            .replace("clause-", "")
            .replace("clause_", "")
        )

        # Convert to CamelCase
        parts = clean_name.replace("-", "_").split("_")
        class_name = "".join(word.capitalize() for word in parts if word)

        # Smart suffix handling
        if class_name and not class_name.endswith(("Clause", "Contract")):
            class_name += "Clause"

        return class_name

    def _to_camel_case(self, text: str) -> str:
        """Convert snake_case or kebab-case to CamelCase"""
        if not text:
            return ""

        # Handle special cases
        text = text.lower()
        parts = text.replace("-", "_").split("_")

        # Filter out empty parts and common words
        meaningful_parts = [
            part for part in parts if part and part not in ("the", "and", "or", "of")
        ]

        return "".join(word.capitalize() for word in meaningful_parts)

    def _get_needed_imports(self, extractions: Dict, clause_type: str) -> List[str]:
        """Determine required imports based on actual extracted data - using official wildcard style."""

        imports = []
        schema = self.variable_targets.get(clause_type, {})

        import_needed = {"money": False, "time": False, "organization": False}

        for var_name, value in extractions.items():
            if var_name in schema and self._is_usable_value(value):
                var_type = schema[var_name].get("type")

                if var_type == "MonetaryAmount":
                    import_needed["money"] = True
                elif var_type in ["Duration", "DateTime"]:
                    import_needed["time"] = True
                elif var_type == "Party[]":
                    import_needed["organization"] = True

        if import_needed["money"]:
            imports.append(
                "import org.accordproject.money.* from https://models.accordproject.org/money.cto"
            )
        if import_needed["time"]:
            imports.append(
                "import org.accordproject.time.* from https://models.accordproject.org/time.cto"
            )
        if import_needed["organization"]:
            imports.append(
                "import org.accordproject.organization.* from https://models.accordproject.org/organization.cto"
            )

        return imports

    def _get_concerto_type(self, var_type: str) -> Optional[str]:
        """Map variable types to Concerto types - updated for v0.22+ with unified Double system."""

        type_mapping = {
            "Boolean": "Boolean",
            "DateTime": "DateTime",
            "Duration": "Duration",
            "MonetaryAmount": "MonetaryAmount",
            "Double": "Double",
            "String": "String",
            "Party[]": "Organization[]",
        }

        return type_mapping.get(var_type)

    def generate_template_data_json(
        self, clause_type: str, extractions: Dict[str, Any], template_id: str
    ) -> Dict:
        """Generate template data (request.json) with proper Accord Project structure."""

        # Generate dynamic class name to match model
        class_name = self._generate_class_name(clause_type, template_id)
        namespace = f"org.accordproject.{template_id.replace('-', '').lower()}"

        # Template data structure with dynamic class name
        template_data = {"$class": f"{namespace}.{class_name}"}

        schema = self.variable_targets.get(clause_type, {})

        # Add extracted variables to the template data
        for var_name, value in extractions.items():
            if var_name in schema and self._is_usable_value(value):
                var_type = schema[var_name].get("type")
                formatted_value = self._format_value_for_template_data(value, var_type)
                if formatted_value is not None:
                    template_data[var_name] = formatted_value

        return template_data

    def _format_value_for_template_data(self, value: Any, var_type: str) -> Any:
        """Format values for template data JSON - updated for v0.22+ with unified Double system."""

        if value is None:
            return None

        if var_type == "Duration" and isinstance(value, dict):
            amount = value.get("amount")
            unit = value.get("unit")
            if amount is not None and unit and str(unit).strip():
                return {
                    "$class": "org.accordproject.time.Duration",
                    "amount": amount,
                    "unit": unit,
                }

        elif var_type == "MonetaryAmount" and isinstance(value, dict):
            amount = value.get("doubleValue")
            currency = value.get("currencyCode")
            if amount is not None and currency and str(currency).strip():
                return {
                    "$class": "org.accordproject.money.MonetaryAmount",
                    "doubleValue": amount,
                    "currencyCode": currency,
                }

        elif var_type == "Party[]" and isinstance(value, list) and value:
            formatted_parties = []
            for party in value:
                if isinstance(party, dict):
                    name = party.get("partyId", "").strip()
                else:
                    name = str(party).strip() if party else ""

                if name:
                    formatted_parties.append(
                        {
                            "$class": "org.accordproject.organization.Organization",
                            "partyId": name,
                        }
                    )
            return formatted_parties if formatted_parties else None

        elif var_type == "DateTime" and isinstance(value, str) and value.strip():
            return value  # Already in ISO format

        elif var_type == "String" and isinstance(value, str):
            return value.strip() if value.strip() else None

        elif (
            var_type in ["Boolean", "Double"] and value is not None
        ):  # Removed "Integer"
            return value

        return None

    def generate_state_json(self, clause_type: str, template_id: str) -> Dict:
        """Generate state.json using current Accord Project v0.22+ state structure."""

        # Use current v0.22+ state structure without stateId requirement
        state_data = {
            "$class": "org.accordproject.runtime.State",
            "$identifier": f"{template_id}-state-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        }

        return state_data

    def generate_package_json(self, template_id: str, clause_type: str) -> Dict:
        """Generate Accord Project compliant package.json"""

        template_info = self.clause_to_template.get(clause_type, {})

        # Clean template name (Accord Project style)
        clean_name = template_id.lower().replace("_", "").replace("-", "")
        display_name = template_id.replace("_", " ").replace("-", " ").title()

        return {
            "name": clean_name,
            "displayName": display_name,
            "version": "0.1.0",
            "description": template_info.get(
                "description", f"Generated template for {clause_type}"
            ),
            "author": "Smart Legal Contracts Generator",
            "license": "Apache-2.0",
            "accordproject": {
                "template": "clause",
                "cicero": "^0.24.0",
                "runtime": "typescript",
            },
            "devDependencies": {
                "@accordproject/cicero-cli": "^0.24.0",
                "@accordproject/cicero-core": "^0.24.0",
            },
            "keywords": [
                clause_type.replace("_", "-"),
                "accord-project",
                "smart-contract",
                "legal-tech",
            ],
        }

    def generate_typescript_logic(
        self, clause_type: str, extractions: Dict[str, Any], template_id: str
    ) -> str:
        """Generate TypeScript logic for modern Accord Project templates"""

        namespace = f"org.accordproject.{template_id.replace('-', '').lower()}"

        # Extract variable names for logic
        variable_names = [
            var
            for var in extractions.keys()
            if var not in {"generatedBy", "generatedAt", "contractState"}
        ]

        return f"""/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * TypeScript logic for {clause_type} template
 * Generated from original legal text with variable extraction
 * Uses unified Double type system for all numeric values
 */

/**
 * Initialize the template
 * @param data - the template data  
 * @returns the initialized template
 */
export function init(data: any): any {{
    // Initialization logic - validate template data
    return data;
}}

/**
 * Main execution function
 * @param data - the template data
 * @param request - the request
 * @returns the response
 */
export function main(data: any, request: any): any {{
    // Template logic for {clause_type}
    
    // Template variables available (all numeric values are Double type):
    {self._generate_variable_comments(variable_names)}
    
    // Basic response structure
    const response = {{
        $class: "{namespace}.MyResponse",
        output: `Template executed successfully for clause type: {clause_type}`,
    }};
    
    return response;
}}
"""

    def _generate_variable_comments(self, variable_names: List[str]) -> str:
        """Generate TypeScript comments for variables"""

        if not variable_names:
            return "    // No variables extracted"

        comments = []
        for var_name in variable_names[:5]:  # Limit to first 5
            comments.append(f"    // const {var_name} = data.{var_name};")

        if len(variable_names) > 5:
            comments.append(f"    // ... and {len(variable_names) - 5} more variables")

        return "\n".join(comments)

    def generate_readme_md(
        self, clause_type: str, template_id: str, extractions: Dict, original_text: str
    ) -> str:
        """Generate README.md documentation for the template."""

        template_info = self.clause_to_template.get(clause_type, {})
        priority = self.get_template_priority(clause_type)
        coverage = self.calculate_coverage_score(extractions, clause_type)

        # Determine if original text was preserved
        text_source = (
            "Preserved from original legal text"
            if original_text
            else "Generated template"
        )

        return f"""# {template_id.replace('-', ' ').title()}

## Overview
{template_info.get('description', f'Accord Project template for {clause_type}')}

## Template Information
- **Clause Type**: {clause_type}
- **Priority**: {priority}
- **Variable Coverage**: {coverage:.1%}
- **Text Source**: {text_source}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Data Types**: Uses unified Double system for all numeric values

## Variables
| Variable | Type | Description |
|----------|------|-------------|
{self._generate_variable_table(clause_type, extractions)}

## Usage

### Generate Contract
```bash
# Use the template engine to generate contract from template + data
npm run generate
```

### Test
```bash
npm test
```

### Validate
```bash
npm run validate
```

## Files
- `text/grammar.tem.md` - Template grammar with {{{{variableName}}}} placeholders
- `model/model.cto` - Concerto data model defining template structure
- `request.json` - Sample template data
- `state.json` - Initial contract state
- `sample.md` - Sample contract instance
- `package.json` - NPM package configuration
- `logic/logic.ts` - TypeScript business logic

## Original Legal Text
{self._format_original_text_for_readme(original_text)}

## Generated by
Template Generator - Updated with Unified Double Type System for Accord Project Compliance
"""

    def _format_original_text_for_readme(self, original_text: str) -> str:
        """Format original text for README display."""

        if not original_text:
            return "No original text available - template generated from variables."

        # Truncate if too long
        if len(original_text) > 500:
            return f"```\n{original_text[:500]}...\n```"
        else:
            return f"```\n{original_text}\n```"

    def _generate_variable_table(self, clause_type: str, extractions: Dict) -> str:
        """Generate markdown table of variables for README."""

        schema = self.variable_targets.get(clause_type, {})
        rows = []

        for var_name, value in extractions.items():
            if var_name in schema and self._is_usable_value(value):
                var_config = schema[var_name]
                var_type = var_config.get("type", "String")
                description = var_config.get("description", "Extracted variable")
                rows.append(f"| {var_name} | {var_type} | {description} |")

        return "\n".join(rows) if rows else "| No variables | - | - |"

    def _is_usable_value(self, value: Any) -> bool:
        """Check if a value is usable for template generation."""

        if value is None:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        if isinstance(value, bool):
            return True  # Both True and False are usable

        if isinstance(value, (int, float)):
            return True  # All numbers are usable (including 0 and 0.0)

        if isinstance(value, list):
            return len(value) > 0 and any(self._is_usable_value(item) for item in value)

        if isinstance(value, dict):
            return any(self._is_usable_value(v) for v in value.values())

        return bool(value)  # Fallback for other types

    def load_extraction_data(self, extraction_file: str) -> List[Dict]:
        """Load variable extraction data with smart auto-discovery."""

        logger.info(f"Looking for extraction file: '{extraction_file}'")

        # Smart path resolution with auto-discovery
        possible_locations = [
            Path(extraction_file),  # Direct path
            self.project_root / extraction_file,  # Relative to project root
            self.project_root / "outputs" / extraction_file,  # In outputs
            self.project_root
            / "outputs"
            / "extracted_variables"
            / extraction_file,  # In extracted_variables
        ]

        # If just a partial name provided, try auto-discovery
        if (
            not Path(extraction_file).is_absolute()
            and "/" not in extraction_file
            and "\\" not in extraction_file
        ):
            # Look for files matching pattern in common locations
            search_locations = [
                self.project_root / "outputs" / "extracted_variables",
                self.project_root / "outputs",
                self.project_root,
            ]

            logger.info(f"Auto-discovery mode for partial name: '{extraction_file}'")

            for search_dir in search_locations:
                logger.debug(f"Searching in: {search_dir}")
                if search_dir.exists():
                    # Try exact match first
                    exact_match = search_dir / extraction_file
                    if exact_match.exists():
                        logger.info(f"Found exact match: {exact_match}")
                        possible_locations.insert(0, exact_match)
                        continue

                    # Try pattern matching for partial names
                    patterns = [
                        f"*{extraction_file}*",  # Contains the name
                        f"extracted_variables*{extraction_file}*",  # Prefixed patterns
                        f"*{extraction_file}*.json",  # With extension
                        "extracted_variables_default_*.json",  # Default pattern
                        "extracted_variables_*.json",  # Any extraction file
                    ]

                    for pattern in patterns:
                        matches = list(search_dir.glob(pattern))
                        logger.debug(
                            f"Pattern '{pattern}' found {len(matches)} matches"
                        )
                        if matches:
                            # Sort by modification time (newest first)
                            matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            logger.info(
                                f"Auto-discovered extraction file: {matches[0]} (pattern: {pattern})"
                            )
                            possible_locations.insert(0, matches[0])
                            break
                else:
                    logger.debug(f"Directory does not exist: {search_dir}")

        # Find the first valid path
        valid_path = None
        for path in possible_locations:
            logger.debug(f"Checking path: {path}")
            if path.exists():
                valid_path = path
                logger.info(f"Using file: {valid_path}")
                break

        if not valid_path:
            # Show helpful information about what was searched
            searched_dirs = set(
                p.parent for p in possible_locations if p.parent.exists()
            )
            available_files = []
            for search_dir in searched_dirs:
                if search_dir.exists():
                    json_files = list(search_dir.glob("*.json"))
                    if json_files:
                        available_files.extend(
                            [f"{search_dir.name}/{f.name}" for f in json_files]
                        )

            logger.error(f"Available files found: {available_files}")

            raise FileNotFoundError(
                f"Extraction file not found: '{extraction_file}'\n"
                f"Searched in: {', '.join(str(p.parent) for p in possible_locations[:4])}\n"
                f"Available JSON files: {available_files[:10]}\n"
                f"Try using just the filename or a more specific pattern."
            )

        logger.info(f"Loading extraction data from: {valid_path}")

        extractions = []

        if valid_path.suffix == ".jsonl":
            # Handle JSONL format (one JSON object per line)
            with open(valid_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            extractions.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Skipping invalid JSON on line {line_num}: {e}"
                            )
        else:
            # Handle JSON format - multiple possible structures
            data = self._load_json_file(valid_path)

            logger.info(f"Loaded raw data type: {type(data)}")
            if isinstance(data, dict):
                logger.info(f"Data keys: {list(data.keys())}")

            if isinstance(data, dict):
                if "extractions" in data:
                    # Variable extractor output format
                    extractions = data["extractions"]
                    metadata = data.get("metadata", {})
                    logger.info(
                        f"Found 'extractions' key with {len(extractions)} items"
                    )
                    logger.info(
                        f"Loaded data with metadata: mode={metadata.get('mode', 'unknown')}, total_extractions={metadata.get('total_extractions', len(extractions))}"
                    )
                elif "results" in data:
                    # Alternative format
                    extractions = data["results"]
                    logger.info(f"Found 'results' key with {len(extractions)} items")
                else:
                    # Single extraction format
                    extractions = [data]
                    logger.info(f"Treating as single extraction")
            elif isinstance(data, list):
                # Direct list format
                extractions = data
                logger.info(f"Found direct list with {len(extractions)} items")
            else:
                raise ValueError(f"Unsupported extraction data format in {valid_path}")

        logger.info(
            f"Loaded {len(extractions)} extraction records from {valid_path.name}"
        )

        # Debug: Show sample extraction
        if extractions:
            logger.info(f"Sample extraction keys: {list(extractions[0].keys())}")

        return extractions

    def group_extractions_by_contract(
        self, extractions: List[Dict]
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """Group extractions by contract and clause type with enhanced validation."""

        logger.info("Grouping extractions by contract and clause type...")
        logger.info(f"Processing {len(extractions)} total extractions")

        # Debug: Show sample extraction structure
        if extractions:
            logger.info(f"Sample extraction structure: {list(extractions[0].keys())}")

        grouped = defaultdict(lambda: defaultdict(list))
        skipped_no_mapping = 0
        skipped_invalid = 0

        for idx, extraction in enumerate(extractions):
            # Validate required fields
            contract_id = extraction.get("contract_id", "").strip()
            clause_type = extraction.get("clause_type", "").strip()

            logger.debug(
                f"Processing extraction {idx}: contract_id='{contract_id}', clause_type='{clause_type}'"
            )

            if not contract_id or not clause_type:
                skipped_invalid += 1
                logger.warning(
                    f"Skipping extraction {idx}: missing contract_id='{contract_id}' or clause_type='{clause_type}'"
                )
                continue

            # Validate that we have mapping for this clause type
            if clause_type not in self.clause_to_template:
                skipped_no_mapping += 1
                logger.warning(
                    f"No template mapping for clause type: '{clause_type}' (available: {list(self.clause_to_template.keys())[:5]}...)"
                )
                continue

            # Validate extraction content
            extractions_dict = extraction.get("extractions", {})
            original_text = extraction.get("original_text", "").strip()

            if not extractions_dict and not original_text:
                skipped_invalid += 1
                logger.warning(f"Skipping extraction {idx}: no content or text")
                continue

            grouped[contract_id][clause_type].append(extraction)
            logger.debug(f"✓ Added extraction {idx} to {contract_id}/{clause_type}")

        self.stats.clauses_skipped_no_mapping = skipped_no_mapping

        logger.info(
            f"Grouped into {len(grouped)} contracts with {sum(len(clauses) for clauses in grouped.values())} clause types"
        )
        if skipped_no_mapping:
            logger.warning(
                f"Skipped {skipped_no_mapping} clauses with no template mapping"
            )
        if skipped_invalid:
            logger.warning(f"Skipped {skipped_invalid} invalid extractions")

        return dict(grouped)

    def calculate_coverage_score(
        self, extractions: Dict[str, Any], clause_type: str
    ) -> float:
        """Calculate variable coverage score for quality assessment."""

        schema = self.variable_targets.get(clause_type, {})
        if not schema:
            return 0.0

        total_vars = len(schema)
        covered_vars = sum(
            1 for var in schema.keys() if extractions.get(var) is not None
        )

        return covered_vars / total_vars if total_vars > 0 else 0.0

    def get_template_priority(self, clause_type: str) -> str:
        """Get priority level for a clause type."""

        for priority, clause_list in self.template_priorities.items():
            if clause_type in clause_list:
                return priority
        return "unknown"

    def generate_templates_for_contract(
        self,
        contract_id: str,
        clause_data: Dict[str, List[Dict]],
        force_regenerate: bool = False,
    ) -> int:
        """Generate all clause templates for a single contract."""

        logger.info(f"Processing contract: {contract_id}")
        self.stats.contracts_processed += 1

        # Create contract directory with safe naming
        contract_slug = slugify(contract_id, max_length=100)
        contract_dir = self.output_dir / contract_slug

        templates_generated = 0
        clauses_processed = 0

        # Sort clause types by priority for consistent processing
        sorted_clauses = self._sort_clauses_by_priority(clause_data.keys())

        for clause_type in sorted_clauses:
            extractions_list = clause_data[clause_type]

            # Get template mapping info
            template_info = self.clause_to_template.get(clause_type)
            if not template_info:
                logger.debug(f"No template mapping for clause type: {clause_type}")
                continue

            template_id = template_info.get("template", clause_type).replace(
                "@1.0.0", ""
            )

            # Process each extraction for this clause type
            for idx, extraction in enumerate(extractions_list):
                clauses_processed += 1

                extractions_dict = extraction.get("extractions", {})
                original_text = extraction.get("original_text", "")

                # Enhanced validation
                if not self._validate_extraction_data(extractions_dict, original_text):
                    self.stats.clauses_skipped_null += 1
                    logger.debug(f"Skipping {clause_type}: no usable data")
                    continue

                # Check coverage threshold
                coverage = self.calculate_coverage_score(extractions_dict, clause_type)
                if coverage < self.min_coverage:
                    logger.debug(
                        f"Skipping {clause_type}: coverage {coverage:.1%} below threshold {self.min_coverage:.1%}"
                    )
                    continue

                # Create clause template directory
                clause_dir_name = (
                    f"{clause_type}-{idx+1:03}"
                    if len(extractions_list) > 1
                    else clause_type
                )
                clause_dir = contract_dir / clause_dir_name

                # Skip if exists and not forcing
                if clause_dir.exists() and not force_regenerate:
                    logger.debug(f"Template exists: {clause_dir_name}")
                    continue

                # Create directory structure
                (clause_dir / "text").mkdir(parents=True, exist_ok=True)
                (clause_dir / "model").mkdir(parents=True, exist_ok=True)
                (clause_dir / "logic").mkdir(parents=True, exist_ok=True)

                try:
                    # Generate all template files
                    self._generate_template_files(
                        clause_dir,
                        clause_type,
                        template_id,
                        extractions_dict,
                        original_text,
                    )

                    templates_generated += 1
                    priority_indicator = self.get_template_priority(clause_type)[
                        0
                    ].upper()
                    text_indicator = (
                        "O" if original_text else "G"
                    )  # Original vs Generated
                    logger.info(
                        f"✓ [{priority_indicator}][{text_indicator}] {contract_slug}/{clause_dir_name} (coverage: {coverage:.1%})"
                    )

                except Exception as e:
                    logger.error(f"Failed to generate {clause_type}: {e}")
                    continue

        if templates_generated > 0:
            self.stats.contracts_successful += 1

        logger.info(
            f"Contract {contract_id}: {templates_generated}/{clauses_processed} templates generated"
        )
        return templates_generated

    def _validate_extraction_data(self, extractions: Dict, original_text: str) -> bool:
        """Validate that extraction data contains usable values OR original text exists."""

        # If we have original text, that's sufficient
        if original_text and original_text.strip():
            return True

        # Otherwise, we need usable extracted data
        if not extractions:
            return False

        usable_count = 0
        for var_name, value in extractions.items():
            if self._is_usable_value(value):
                usable_count += 1

        # Require at least one usable value
        return usable_count > 0

    def _sort_clauses_by_priority(self, clause_types: List[str]) -> List[str]:
        """Sort clause types by priority for consistent processing order."""

        priority_map = {}
        for clause_type in clause_types:
            priority = self.get_template_priority(clause_type)
            priority_order = (
                self.priority_order.index(priority)
                if priority in self.priority_order
                else 999
            )
            priority_map[clause_type] = priority_order

        return sorted(clause_types, key=lambda x: priority_map[x])

    def _generate_template_files(
        self,
        clause_dir: Path,
        clause_type: str,
        template_id: str,
        extractions: Dict,
        original_text: str,
    ):
        """Generate all files for a single clause template."""

        # Generate template grammar (text/grammar.tem.md) - PRESERVE ORIGINAL TEXT
        template_grammar = self.generate_template_grammar(
            original_text, clause_type, extractions
        )
        (clause_dir / "text" / "grammar.tem.md").write_text(
            template_grammar, encoding="utf-8"
        )

        # Generate sample contract (sample.md at root level) - USE ORIGINAL TEXT
        sample_contract = self.generate_sample_contract(
            original_text, clause_type, extractions
        )
        (clause_dir / "sample.md").write_text(sample_contract, encoding="utf-8")

        # Generate Concerto model (model/model.cto)
        model_cto = self.generate_model_cto(clause_type, extractions, template_id)
        (clause_dir / "model" / "model.cto").write_text(model_cto, encoding="utf-8")

        # Generate TypeScript logic (logic/logic.ts)
        typescript_logic = self.generate_typescript_logic(
            clause_type, extractions, template_id
        )
        (clause_dir / "logic" / "logic.ts").write_text(
            typescript_logic, encoding="utf-8"
        )

        # Generate template data (request.json)
        template_data = self.generate_template_data_json(
            clause_type, extractions, template_id
        )
        (clause_dir / "request.json").write_text(
            json.dumps(template_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Generate state.json for contract state management
        state_data = self.generate_state_json(clause_type, template_id)
        (clause_dir / "state.json").write_text(
            json.dumps(state_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Generate package.json
        package_json = self.generate_package_json(template_id, clause_type)
        (clause_dir / "package.json").write_text(
            json.dumps(package_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Generate README.md for documentation
        readme_content = self.generate_readme_md(
            clause_type, template_id, extractions, original_text
        )
        (clause_dir / "README.md").write_text(readme_content, encoding="utf-8")

    def generate_all_templates(
        self,
        extraction_file: str,
        max_contracts: Optional[int] = None,
        force_regenerate: bool = False,
        specific_template: Optional[str] = None,
        priority_filter: Optional[List[str]] = None,
    ) -> Dict:
        """Main method to generate all templates with proper CiceroMark syntax and unified Double system."""

        self.stats.start_time = datetime.now()
        logger.info("Starting template generation process...")
        try:
            # Load and validate extraction data
            extractions = self.load_extraction_data(extraction_file)
            if not extractions:
                raise ValueError("No extraction data found")

            # Group by contract and clause type
            grouped_data = self.group_extractions_by_contract(extractions)
            if not grouped_data:
                raise ValueError("No valid extractions found after grouping")

            # Apply filters
            filtered_data = self._apply_filters(
                grouped_data, max_contracts, specific_template, priority_filter
            )

            logger.info(f"Processing {len(filtered_data)} contracts...")

            # Generate templates for each contract
            for contract_id in filtered_data:
                clause_data = filtered_data[contract_id]

                try:
                    templates_count = self.generate_templates_for_contract(
                        contract_id, clause_data, force_regenerate
                    )
                    self.stats.templates_generated += templates_count

                except Exception as e:
                    logger.error(f"Failed to process contract {contract_id}: {e}")
                    continue

            self.stats.end_time = datetime.now()

            # Generate summary report
            summary = self._generate_summary_report()
            self._save_generation_report(summary)

            logger.info("Template generation complete!")
            self._log_final_summary(summary)

            return summary

        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            raise

    def _apply_filters(
        self,
        grouped_data: Dict,
        max_contracts: Optional[int],
        specific_template: Optional[str],
        priority_filter: Optional[List[str]],
    ) -> Dict:
        """Apply various filters to the grouped data."""

        filtered_data = grouped_data.copy()

        # Filter by specific template
        if specific_template:
            new_filtered = {}
            for contract_id, clause_data in filtered_data.items():
                filtered_clauses = {
                    clause_type: extractions
                    for clause_type, extractions in clause_data.items()
                    if self.clause_to_template.get(clause_type, {})
                    .get("template", "")
                    .startswith(specific_template)
                }
                if filtered_clauses:
                    new_filtered[contract_id] = filtered_clauses
            filtered_data = new_filtered
            logger.info(
                f"🔍 Filtered to template '{specific_template}': {len(filtered_data)} contracts"
            )

        # Filter by priority
        if priority_filter:
            new_filtered = {}
            for contract_id, clause_data in filtered_data.items():
                filtered_clauses = {
                    clause_type: extractions
                    for clause_type, extractions in clause_data.items()
                    if self.get_template_priority(clause_type) in priority_filter
                }
                if filtered_clauses:
                    new_filtered[contract_id] = filtered_clauses
            filtered_data = new_filtered
            logger.info(
                f"Filtered to priorities {priority_filter}: {len(filtered_data)} contracts"
            )

        # Limit number of contracts
        if max_contracts and max_contracts < len(filtered_data):
            contract_ids = list(filtered_data.keys())[:max_contracts]
            filtered_data = {cid: filtered_data[cid] for cid in contract_ids}
            logger.info(f"Limited to {max_contracts} contracts")

        return filtered_data

    def _generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report."""

        base_stats = self.stats.to_dict()

        return {
            "generation_metadata": {
                "mode": self.mode,
                "timestamp": datetime.now().isoformat(),
                "output_directory": str(self.output_dir),
                "quality_threshold": self.min_coverage,
                "approach": "Updated CiceroMark syntax with unified Double type system for Accord Project compliance",
            },
            "statistics": base_stats,
            "quality_metrics": {
                "generation_rate_per_minute": base_stats["generation_rate"] * 60,
                "success_rate_percentage": base_stats["success_rate"] * 100,
                "average_templates_per_contract": base_stats["templates_generated"]
                / max(base_stats["contracts_successful"], 1),
                "original_text_preservation_rate": (
                    base_stats["original_text_preserved"]
                    / max(base_stats["templates_generated"], 1)
                )
                * 100,
                "fallback_generation_rate": (
                    base_stats["fallback_templates_generated"]
                    / max(base_stats["templates_generated"], 1)
                )
                * 100,
                "double_replacement_rate": (
                    base_stats["double_replacements"]
                    / max(base_stats["templates_generated"], 1)
                )
                * 100,
            },
            "configuration": {
                "total_clause_types": len(self.clause_to_template),
                "priority_levels": list(self.template_priorities.keys()),
                "variable_targets_loaded": len(self.variable_targets),
                "unified_double_system": True,
            },
        }

    def _save_generation_report(self, summary: Dict):
        """Save generation report to outputs directory."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = (
            self.output_dir / f"generation_report_{self.mode}_{timestamp}.json"
        )

        self._save_json_file(report_file, summary)
        logger.info(f"Report saved: {report_file}")

    def _log_final_summary(self, summary: Dict):
        """Log final summary statistics."""

        stats = summary["statistics"]
        quality = summary["quality_metrics"]

        logger.info(f"Generation Summary:")
        logger.info(
            f"Contracts: {stats['contracts_successful']}/{stats['contracts_processed']} ({quality['success_rate_percentage']:.1f}%)"
        )
        logger.info(f"Templates: {stats['templates_generated']} generated")
        logger.info(
            f"Original Text Preserved: {stats['original_text_preserved']} ({quality['original_text_preservation_rate']:.1f}%)"
        )
        logger.info(
            f"Double Replacements: {stats['double_replacements']} ({quality['double_replacement_rate']:.1f}%)"
        )
        logger.info(
            f"Fallback Templates: {stats['fallback_templates_generated']} ({quality['fallback_generation_rate']:.1f}%)"
        )
        logger.info(
            f"Rate: {quality['generation_rate_per_minute']:.1f} templates/minute"
        )
        logger.info(f"Output: {summary['generation_metadata']['output_directory']}")

    def _load_json_file(self, path: Path, default: Any = None) -> Any:
        """Load JSON file with enhanced error handling."""

        if not path.exists():
            return default if default is not None else {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return default if default is not None else {}

    def _save_json_file(self, path: Path, data: Any):
        """Save JSON file with enhanced error handling."""

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {path}: {e}")


def main():
    """Updated command-line interface with comprehensive options and unified Double system."""

    parser = argparse.ArgumentParser(
        description="UPDATED Template Generator - Unified Double Type System for Accord Project Compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation with unified Double type system
  python -m src.generation.template_generator -e outputs/extracted_variables/default_20250115.json
  
  # Production run with unified Double system
  python -m src.generation.template_generator -e data.json -m 50 --force --mode production
  
  # Filter by template type
  python -m src.generation.template_generator -e data.json -t parties --debug
  
  # Filter by priority
  python -m src.generation.template_generator -e data.json --priority essential high
  
  # Custom output location
  python -m src.generation.template_generator -e data.json -o custom_templates

UPDATED VERSION - Unified Double Type System:
  ✓ Double data type handles BOTH decimals AND integers (no Integer type)
  ✓ Comprehensive numeric pattern recognition (1%, 110%, one percent, five(5) percent, integers, decimals)
  ✓ Proper CiceroMark syntax: {{variableName}}
  ✓ Processes ALL parties correctly
  ✓ Maintains authentic legal language
  ✓ Contract state management included
  ✓ TypeScript logic generation
  ✓ Proper Concerto model structure with valid Accord Project types only
  ✓ Full Accord Project compliance
        """,
    )

    # Required arguments
    parser.add_argument(
        "--extraction-file",
        "-e",
        required=True,
        help="Path to variable extraction file (JSON or JSONL)",
    )

    # Processing options
    parser.add_argument(
        "--max-contracts", "-m", type=int, help="Maximum number of contracts to process"
    )

    parser.add_argument(
        "--mode",
        choices=["default", "production", "test", "debug"],
        default="default",
        help="Generation mode (affects logging and validation)",
    )

    # Filtering options
    parser.add_argument(
        "--template", "-t", help="Generate only templates matching this ID"
    )

    parser.add_argument(
        "--priority",
        nargs="+",
        choices=["essential", "high", "medium", "low"],
        help="Filter by template priority levels",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs/generated_templates",
        help="Output directory path (default: outputs/generated_templates)",
    )

    parser.add_argument(
        "--config-dir",
        "-c",
        default="config",
        help="Configuration directory path (default: config)",
    )

    # Behavior options
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regeneration of existing templates",
    )

    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = (
        logging.DEBUG
        if args.debug
        else (logging.INFO if args.verbose else logging.INFO)
    )
    logging.getLogger().setLevel(log_level)

    print("UPDATED Template Generator for Accord Project Smart Legal Contracts")
    print("=" * 70)
    print("UPDATED VERSION - UNIFIED DOUBLE TYPE SYSTEM")
    print("✓ Double data type handles BOTH decimals AND integers (no Integer type)")
    print(
        "✓ Comprehensive numeric pattern recognition (1%, 110%, one percent, five(5) percent, integers, decimals)"
    )
    print("✓ Proper CiceroMark syntax: {{variableName}}")
    print("✓ Processes ALL parties correctly")
    print("✓ Maintains authentic legal language")
    print("✓ Contract state management included")
    print("✓ TypeScript logic generation")
    print("✓ Proper Concerto model structure with valid Accord Project types only")
    print("✓ Full Accord Project compliance")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print()

    try:
        # Initialize generator
        generator = TemplateGenerator(
            mode=args.mode, config_dir=args.config_dir, output_dir=args.output_dir
        )

        # Generate templates
        summary = generator.generate_all_templates(
            extraction_file=args.extraction_file,
            max_contracts=args.max_contracts,
            force_regenerate=args.force,
            specific_template=args.template,
            priority_filter=args.priority,
        )

        # Print final summary
        print("\n" + "=" * 70)
        print("Generation Complete")
        print("=" * 70)

        stats = summary["statistics"]
        quality = summary["quality_metrics"]

        print(f"Results:")
        print(
            f"• Contracts Processed: {stats['contracts_successful']}/{stats['contracts_processed']}"
        )
        print(f"• Templates Generated: {stats['templates_generated']}")
        print(
            f"• Original Text Preserved: {stats['original_text_preserved']} ({quality['original_text_preservation_rate']:.1f}%)"
        )
        print(
            f"• Double Replacements: {stats['double_replacements']} ({quality['double_replacement_rate']:.1f}%)"
        )
        print(
            f"• Fallback Templates: {stats['fallback_templates_generated']} ({quality['fallback_generation_rate']:.1f}%)"
        )
        print(f"• Success Rate: {quality['success_rate_percentage']:.1f}%")
        print(
            f"• Generation Rate: {quality['generation_rate_per_minute']:.1f} templates/min"
        )
        print(f"• Duration: {stats['duration_seconds']:.1f} seconds")
        print(f"• Unified Double System: ✓ Enabled")
        print(
            f"\nOutput Directory: {summary['generation_metadata']['output_directory']}"
        )

        # Legend for indicators
        print(f"\nLegend:")
        print(f"• [E][O] = Essential priority, Original text preserved")
        print(f"• [H][G] = High priority, Generated fallback template")
        print(f"• [M][O] = Medium priority, Original text preserved")

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Template generation failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
