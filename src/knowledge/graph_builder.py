"""
Legal Contract Knowledge Graph Builder
Builds Neo4j knowledge graph from extracted legal contract data with LLM integration
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from hashlib import sha256

try:
    from ollama import chat

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama package not installed. LLM features disabled.")

# Import slugify with fallback
try:
    from slugify import slugify

    SLUGIFY_AVAILABLE = True
except ImportError:
    SLUGIFY_AVAILABLE = False
    print(
        "Warning: python-slugify package not installed. Using fallback slugification."
    )

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class LegalKnowledgeGraphBuilder:
    """Builds a knowledge graph from legal contract extraction data."""

    def __init__(
        self,
        graph_config_path: str = "config/graph_config.json",
        extraction_config_path: str = "config/knowledge_extraction.json",
    ):
        """Initialize the graph builder."""

        self.graph_client = Neo4jClient(graph_config_path)
        self.extraction_config = self._load_extraction_config(extraction_config_path)

        # Stats tracking
        self.stats = {
            "contracts_processed": 0,
            "parties_created": 0,
            "clauses_created": 0,
            "variables_created": 0,
            "templates_created": 0,
            "relationships_created": 0,
            "llm_calls": 0,
            "errors": 0,
        }

        logger.info("Legal Knowledge Graph Builder initialized")

    def _load_extraction_config(self, config_path: str) -> Dict:
        """Load extraction configuration."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Extraction config not found: {config_path}")

        with open(config_file, "r") as f:
            return json.load(f)

    def _fallback_slugify(self, text: str, max_length: int = 100) -> str:
        """Fallback slugification when python-slugify is not available."""
        # Convert to lowercase and replace spaces/special chars with hyphens
        text = re.sub(r"[^\w\s-]", "", text.lower())
        text = re.sub(r"[-\s]+", "-", text)
        return text[:max_length].strip("-")

    def build_graph_from_extractions(
        self,
        extraction_file: Optional[str] = None,
        max_contracts: Optional[int] = None,
        clear_existing: bool = False,
    ) -> Dict:
        """Main method to build the knowledge graph from extraction data."""

        logger.info(
            f"Building knowledge graph from {extraction_file or 'auto-detected file'}"
        )
        start_time = datetime.now()

        # Clear existing graph if requested
        if clear_existing:
            logger.info("Clearing existing graph...")
            self.graph_client.clear_graph(confirm=True)

        # Create indexes
        self.graph_client.create_indexes()

        # Load extraction data
        extractions = self._load_extraction_data(extraction_file)

        if max_contracts:
            # Group by contract and limit
            contract_groups = self._group_by_contract(extractions)
            limited_contracts = list(contract_groups.keys())[:max_contracts]
            extractions = []
            for contract_id in limited_contracts:
                extractions.extend(contract_groups[contract_id])
            logger.info(
                f"Limited to {max_contracts} contracts ({len(extractions)} extractions)"
            )

        # Process extractions
        processed_contracts = self._process_extractions(extractions)

        # Create similarity relationships if available
        self._create_similarity_relationships(extractions)

        # Process generated templates (only for the contracts we processed)
        self._process_templates(processed_contracts)

        # Generate final stats
        graph_stats = self.graph_client.get_stats()
        end_time = datetime.now()

        results = {
            "processing_stats": self.stats,
            "graph_stats": graph_stats,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "extractions_processed": len(extractions),
        }

        logger.info(f"Graph building complete: {results}")
        return results

    def _load_extraction_data(
        self, extraction_file: Optional[str] = None
    ) -> List[Dict]:
        """Load extraction data from file with auto-detection."""

        if extraction_file:
            # Specific file provided
            file_path = self._find_extraction_file(extraction_file)
        else:
            # Auto-detect latest extraction file
            file_path = self._auto_detect_extraction_file()

        logger.info(f"Loading extractions from: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different extraction file formats
        if isinstance(data, dict) and "extractions" in data:
            extractions = data["extractions"]
            metadata = data.get("metadata", {})
            logger.info(
                f"Loaded {len(extractions)} extractions (mode: {metadata.get('mode', 'unknown')})"
            )
        elif isinstance(data, list):
            extractions = data
            logger.info(f"Loaded {len(extractions)} extractions (direct list)")
        else:
            raise ValueError(f"Unsupported extraction file format: {type(data)}")

        return extractions

    def _find_extraction_file(self, extraction_file: str) -> Path:
        """Find extraction file in common locations."""

        # Try direct path first
        file_path = Path(extraction_file)
        if file_path.exists():
            return file_path

        # Try common locations
        possible_paths = [
            Path("outputs/extracted_variables") / extraction_file,
            Path("outputs") / extraction_file,
            Path(extraction_file),
        ]

        # If just a partial name, try pattern matching
        if not any(
            file_path.suffix for file_path in possible_paths if file_path.exists()
        ):
            search_dirs = [Path("outputs/extracted_variables"), Path("outputs")]

            for search_dir in search_dirs:
                if search_dir.exists():
                    # Try pattern matching
                    patterns = [
                        f"*{extraction_file}*",
                        f"extracted_variables*{extraction_file}*",
                        f"*{extraction_file}*.json",
                    ]

                    for pattern in patterns:
                        matches = list(search_dir.glob(pattern))
                        if matches:
                            # Return newest file
                            matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            logger.info(
                                f"Auto-discovered: {matches[0]} (pattern: {pattern})"
                            )
                            return matches[0]

        # Try the possible paths
        for path in possible_paths:
            if path.exists():
                return path

        raise FileNotFoundError(f"Extraction file not found: {extraction_file}")

    def _auto_detect_extraction_file(self) -> Path:
        """Auto-detect the latest extraction file."""

        search_dir = Path("outputs/extracted_variables")
        if not search_dir.exists():
            search_dir = Path("outputs")

        if not search_dir.exists():
            raise FileNotFoundError("No outputs directory found")

        # Look for extraction files
        patterns = ["extracted_variables_*.json", "*extraction*.json", "*.json"]

        all_files = []
        for pattern in patterns:
            matches = list(search_dir.glob(pattern))
            all_files.extend(matches)

        if not all_files:
            raise FileNotFoundError(f"No extraction files found in {search_dir}")

        # Sort by modification time (newest first)
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        logger.info(f"Auto-detected latest extraction file: {all_files[0]}")
        return all_files[0]

    def _group_by_contract(self, extractions: List[Dict]) -> Dict[str, List[Dict]]:
        """Group extractions by contract ID."""
        groups = defaultdict(list)
        for extraction in extractions:
            contract_id = extraction.get("contract_id", "unknown")
            groups[contract_id].append(extraction)
        return dict(groups)

    def _process_extractions(self, extractions: List[Dict]) -> List[str]:
        """Process all extractions and create graph nodes and relationships. Returns list of processed contract IDs."""

        logger.info(f"Processing {len(extractions)} extractions...")

        # Group by contract for efficient processing
        contract_groups = self._group_by_contract(extractions)
        processed_contract_ids = []

        for contract_id, contract_extractions in contract_groups.items():
            try:
                self._process_contract(contract_id, contract_extractions)
                processed_contract_ids.append(contract_id)
                self.stats["contracts_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process contract {contract_id}: {e}")
                self.stats["errors"] += 1

        return processed_contract_ids

    def _process_contract(self, contract_id: str, extractions: List[Dict]):
        """Process a single contract and its extractions."""

        logger.debug(f"Processing contract: {contract_id}")

        # Create contract node
        contract_title = self._generate_contract_title(contract_id, extractions)
        contract_data = {
            "contract_id": contract_id,
            "title": contract_title,
            "created_at": datetime.now().isoformat(),
        }

        self.graph_client.create_contract_node(contract_data)

        # Track parties for this contract
        contract_parties = set()

        # Process each clause extraction
        for extraction in extractions:
            clause_id = self._generate_clause_id(contract_id, extraction)

            # Create clause node
            clause_data = self._prepare_clause_data(clause_id, extraction)
            self.graph_client.create_clause_node(clause_data)
            self.stats["clauses_created"] += 1

            # Create contract -> clause relationship
            self.graph_client.create_relationship(
                contract_id, clause_id, "CONTAINS", "Contract", "Clause"
            )
            self.stats["relationships_created"] += 1

            # Process variables for this clause
            variables_data = extraction.get("extractions", {})
            for var_name, var_value in variables_data.items():
                if self._is_valid_variable(var_value):
                    variable_data = self._prepare_variable_data(
                        var_name, var_value, extraction
                    )
                    var_id = self.graph_client.create_variable_node(variable_data)
                    self.stats["variables_created"] += 1

                    # Create clause -> variable relationship
                    self.graph_client.create_relationship(
                        clause_id, var_name, "HAS_VARIABLE", "Clause", "Variable"
                    )
                    self.stats["relationships_created"] += 1

                    # Handle party variables specially
                    if var_name == "parties" and isinstance(var_value, list):
                        for party_data in var_value:
                            party_id = self._process_party(party_data, extraction)
                            if party_id:
                                contract_parties.add(party_id)

        # Create contract -> party relationships
        for party_id in contract_parties:
            role = self._detect_party_role(party_id, extractions)
            self.graph_client.create_relationship(
                contract_id, party_id, "INVOLVES", "Contract", "Party", {"role": role}
            )
            self.stats["relationships_created"] += 1

    def _generate_contract_title(
        self, contract_id: str, extractions: List[Dict]
    ) -> str:
        """Generate a descriptive title for the contract - IMPROVED party extraction."""

        if not self.extraction_config["extraction_rules"]["contracts"][
            "title_extraction"
        ]["enabled"]:
            return contract_id.replace("_", " ").title()

        # Collect clause types and parties for title generation
        clause_types = list(set(e.get("clause_type", "") for e in extractions))
        parties = []

        # IMPROVED: Better party extraction logic
        for extraction in extractions:
            if extraction.get("clause_type") == "parties":
                party_list = extraction.get("extractions", {}).get("parties", [])
                for party in party_list:
                    if isinstance(party, dict):
                        party_name = party.get("partyId", "").strip()
                        if party_name and len(party_name) > 1:  # Valid party name
                            parties.append(party_name)
                    elif isinstance(party, str) and party.strip():
                        parties.append(party.strip())

        # Remove duplicates while preserving order
        seen = set()
        unique_parties = []
        for party in parties:
            if party.lower() not in seen:
                seen.add(party.lower())
                unique_parties.append(party)
        parties = unique_parties

        # Use LLM to generate title if available
        if OLLAMA_AVAILABLE and self.extraction_config["llm_integration"]["enabled"]:
            return self._llm_generate_title(contract_id, clause_types, parties)

        # Fallback: improved simple title generation
        return self._generate_safe_fallback_title(contract_id, clause_types, parties)

    def _generate_clause_id(self, contract_id: str, extraction: Dict) -> str:
        """Generate unique clause ID."""
        clause_type = extraction.get("clause_type", "unknown")
        text_hash = sha256(extraction.get("original_text", "").encode()).hexdigest()[:8]
        return f"{contract_id}_{clause_type}_{text_hash}"

    def _prepare_clause_data(self, clause_id: str, extraction: Dict) -> Dict:
        """Prepare clause data for node creation."""

        original_text = extraction.get("original_text", "")
        if (
            len(original_text)
            > self.extraction_config["extraction_rules"]["clauses"]["max_text_length"]
        ):
            original_text = original_text[:2000] + "..."

        # Calculate coverage score
        extractions_dict = extraction.get("extractions", {})
        total_vars = len(extractions_dict)
        valid_vars = sum(
            1 for v in extractions_dict.values() if self._is_valid_variable(v)
        )
        coverage_score = valid_vars / total_vars if total_vars > 0 else 0.0

        return {
            "clause_id": clause_id,
            "clause_type": extraction.get("clause_type", "unknown"),
            "original_text": original_text,
            "coverage_score": coverage_score,
        }

    def _prepare_variable_data(
        self, var_name: str, var_value: Any, extraction: Dict
    ) -> Dict:
        """Prepare variable data for node creation."""

        # Determine variable type
        var_type = self._detect_variable_type(var_value)

        # Format value for storage
        formatted_value = self._format_variable_value(var_value, var_type)

        return {
            "var_name": var_name,
            "var_type": var_type,
            "value": str(formatted_value)[:500],  # Limit length
            "extracted_value": (
                json.dumps(var_value)
                if isinstance(var_value, (dict, list))
                else str(var_value)
            ),
        }

    def _process_party(self, party_data: Dict, extraction: Dict) -> Optional[str]:
        """Process a party and create party node."""

        if not isinstance(party_data, dict):
            return None

        party_name = party_data.get("partyId", "").strip()
        if not party_name:
            return None

        party_id = self._generate_party_id(party_name)
        entity_type = self._detect_entity_type(party_name)

        party_node_data = {
            "party_id": party_id,
            "name": party_name,
            "entity_type": entity_type,
        }

        self.graph_client.create_party_node(party_node_data)
        self.stats["parties_created"] += 1

        return party_id

    def _generate_party_id(self, party_name: str) -> str:
        """Generate consistent party ID from name."""
        # Clean and normalize the name for ID
        cleaned = re.sub(r"[^\w\s]", "", party_name.lower())
        return "_".join(cleaned.split())

    def _detect_entity_type(self, party_name: str) -> str:
        """Detect entity type based on party name."""

        name_lower = party_name.lower()
        patterns = self.extraction_config["extraction_rules"]["parties"][
            "entity_type_detection"
        ]["patterns"]

        for entity_type, indicators in patterns.items():
            if any(indicator in name_lower for indicator in indicators):
                return entity_type

        return "unknown"

    def _detect_variable_type(self, var_value: Any) -> str:
        """Detect variable type from value."""

        if isinstance(var_value, bool):
            return "boolean"
        elif isinstance(var_value, (int, float)):
            return "numeric"
        elif isinstance(var_value, str):
            return "text"
        elif isinstance(var_value, list):
            return "entity_list"
        elif isinstance(var_value, dict):
            # Check for specific Accord Project types
            class_name = var_value.get("$class", "")
            if "MonetaryAmount" in class_name:
                return "monetary"
            elif "Duration" in class_name:
                return "temporal"
            else:
                return "structured"
        else:
            return "unknown"

    def _format_variable_value(self, var_value: Any, var_type: str) -> str:
        """Format variable value for display."""

        if var_type == "monetary" and isinstance(var_value, dict):
            amount = var_value.get("doubleValue", 0)
            currency = var_value.get("currencyCode", "USD")
            return f"{amount} {currency}"

        elif var_type == "temporal" and isinstance(var_value, dict):
            amount = var_value.get("amount", 0)
            unit = var_value.get("unit", "days")
            return f"{amount} {unit}"

        elif var_type == "entity_list" and isinstance(var_value, list):
            names = []
            for item in var_value:
                if isinstance(item, dict):
                    names.append(item.get("partyId", str(item)))
                else:
                    names.append(str(item))
            return ", ".join(names)

        else:
            return str(var_value)

    def _is_valid_variable(self, var_value: Any) -> bool:
        """Check if variable value is valid for graph creation."""

        if var_value is None:
            return False

        if isinstance(var_value, str):
            return bool(var_value.strip())

        if isinstance(var_value, list):
            return len(var_value) > 0

        if isinstance(var_value, dict):
            return any(v for v in var_value.values() if v is not None)

        return True

    def _detect_party_role(self, party_id: str, extractions: List[Dict]) -> str:
        """Detect party role using LLM if available."""

        if (
            not OLLAMA_AVAILABLE
            or not self.extraction_config["llm_integration"]["enabled"]
            or not self.extraction_config["llm_integration"]["tasks"][
                "party_role_detection"
            ]["enabled"]
        ):
            return "party"

        # Find party name and context
        party_name = party_id.replace("_", " ").title()
        context_text = ""

        for extraction in extractions:
            if extraction.get("clause_type") == "parties":
                context_text = extraction.get("original_text", "")[:500]
                break

        if not context_text:
            return "party"

        # Query LLM for role detection
        prompt = self.extraction_config["llm_integration"]["tasks"][
            "party_role_detection"
        ]["prompt"]
        formatted_prompt = prompt.format(
            party_name=party_name, clause_text=context_text
        )

        try:
            response = chat(
                model=self.extraction_config["llm_integration"]["model"],
                messages=[{"role": "user", "content": formatted_prompt}],
                options=self.extraction_config["llm_integration"]["options"],
            )

            role = response["message"]["content"].strip().lower()
            self.stats["llm_calls"] += 1

            # Validate role
            valid_roles = [
                "client",
                "vendor",
                "licensor",
                "licensee",
                "buyer",
                "seller",
                "other",
            ]
            return role if role in valid_roles else "party"

        except Exception as e:
            logger.warning(f"LLM role detection failed: {e}")
            return "party"

    def _llm_generate_title(
        self, contract_id: str, clause_types: List[str], parties: List[str]
    ) -> str:
        """Generate contract title using LLM - FIXED to prevent party hallucination."""

        try:
            # Filter out empty/invalid parties
            valid_parties = [
                p.strip() for p in parties if p and p.strip() and len(p.strip()) > 1
            ]

            # Create context based on what we actually have
            if len(valid_parties) >= 3:
                # Three or more parties - use first three with "Multi-Party"
                party_context = f"Parties: {', '.join(valid_parties[:3])}"
                prompt_template = "Generate ONLY a contract title. Use '[Party1] & [Party2] & [Party3] Agreement' or '[Party1] & Others [Agreement Type]'. Max 50 characters. Use ONLY the exact party names provided. No explanations."
            elif len(valid_parties) == 2:
                # Two parties - standard format
                party_context = f"Parties: {', '.join(valid_parties)}"
                prompt_template = "Generate ONLY a contract title. Format: '[Party1] & [Party2] [Agreement Type]' or '[Party1] & [Party2] Agreement'. Max 50 characters. Use ONLY the exact party names provided. No explanations."
            elif len(valid_parties) == 1:
                # Only one party - be explicit about this
                party_context = f"Single party: {valid_parties[0]}"
                prompt_template = "Generate ONLY a contract title. Format: '[Party] [Agreement Type]' or '[Party] Agreement'. Max 50 characters. Use ONLY the exact party name provided. Do not invent additional parties. No explanations."
            else:
                # No parties - focus on clause types
                party_context = "No parties identified"
                prompt_template = "Generate ONLY a contract title based on clause types. Format: '[Agreement Type] Agreement' or '[Main Topic] Contract'. Max 50 characters. Do not invent party names. No explanations."

            # Build context with clause types
            clause_context = (
                f"Clause types: {', '.join(clause_types[:5])}"
                if clause_types
                else "Mixed clauses"
            )
            full_context = f"{party_context}. {clause_context}"

            # Create the full prompt
            full_prompt = f"{prompt_template} Context: {full_context}"

            response = chat(
                model=self.extraction_config["llm_integration"]["model"],
                messages=[{"role": "user", "content": full_prompt}],
                options=self.extraction_config["llm_integration"]["options"],
            )

            raw_title = response["message"]["content"].strip()
            self.stats["llm_calls"] += 1

            # Clean the LLM output - remove common fluff
            title = self._clean_llm_title_output(raw_title)

            # Additional validation - check if LLM hallucinated parties
            if len(valid_parties) <= 1:
                # Remove common hallucinated party patterns
                hallucination_patterns = [
                    r"\b[A-Z][a-z]+ & [A-Z][a-z]+\b",  # "Acme & XYZ" pattern
                    r"\b[A-Z]+ & [A-Z]+\b",  # "ABC & DEF" pattern
                    r"\bCompany A & Company B\b",
                    r"\bParty A & Party B\b",
                    r"\bClient & Vendor\b",
                ]

                for pattern in hallucination_patterns:
                    if re.search(pattern, title, re.IGNORECASE):
                        logger.warning(
                            f"Detected hallucinated parties in title: {title}"
                        )
                        # Fallback to safe title
                        return self._generate_safe_fallback_title(
                            contract_id, clause_types, valid_parties
                        )

            return title[:60]  # Slightly longer limit for 3+ parties

        except Exception as e:
            logger.warning(f"LLM title generation failed: {e}")
            return self._generate_safe_fallback_title(
                contract_id, clause_types, parties
            )

    def _clean_llm_title_output(self, raw_title: str) -> str:
        """Clean LLM output to remove common fluff and explanations."""

        title = raw_title.strip()

        # Remove common LLM fluff patterns
        fluff_patterns = [
            r"^(Here is|Here\'s|The title is|Title:|Contract title:)\s*",
            r"^(Based on|Given|Considering)\s+.*?,\s*",
            r"\s*(would be|could be|should be)\s*:?\s*",
            r"\s*\(.*?\)\s*",  # Remove parenthetical notes
            r"\s*\.+$",  # Remove trailing dots
            r'^"(.*)"$',  # Remove quotes if wrapping entire title
            r"^\s*[\-\*]\s*",  # Remove bullet points
        ]

        for pattern in fluff_patterns:
            title = re.sub(
                pattern,
                r"\1" if "(" in pattern and ")" in pattern else "",
                title,
                flags=re.IGNORECASE,
            ).strip()

        # Split on common separators and take the first clean part
        separators = [".", ":", "!", "?", "\n", "because", "since", "as it", "which"]
        for sep in separators:
            if sep in title:
                parts = title.split(sep)
                if parts[0].strip():
                    title = parts[0].strip()
                    break

        # Final cleanup
        title = re.sub(r"[^\w\s&\-]", "", title)  # Keep only alphanumeric, spaces, &, -
        title = re.sub(r"\s+", " ", title)  # Normalize whitespace

        return title.strip()

    def _generate_safe_fallback_title(
        self, contract_id: str, clause_types: List[str], parties: List[str]
    ) -> str:
        """Generate a safe fallback title without hallucination."""

        # Filter valid parties
        valid_parties = [
            p.strip() for p in parties if p and p.strip() and len(p.strip()) > 1
        ]

        if len(valid_parties) >= 3:
            # Three or more parties - use first two with "& Others"
            return f"{valid_parties[0]} & {valid_parties[1]} & Others Agreement"[:60]
        elif len(valid_parties) == 2:
            # Two parties - standard format
            return f"{valid_parties[0]} & {valid_parties[1]} Agreement"[:60]
        elif len(valid_parties) == 1:
            # Single party
            main_clause_type = self._get_primary_clause_type(clause_types)
            if main_clause_type:
                clause_name = main_clause_type.replace("_", " ").title()
                return f"{valid_parties[0]} {clause_name}"[:60]
            else:
                return f"{valid_parties[0]} Agreement"[:60]
        else:
            # No parties - use clause types or contract ID
            main_clause_type = self._get_primary_clause_type(clause_types)
            if main_clause_type:
                clause_name = main_clause_type.replace("_", " ").title()
                return f"{clause_name} Agreement"[:60]
            else:
                # Last resort - clean up contract ID
                clean_id = contract_id.replace("_", " ").replace("-", " ").title()
                return f"{clean_id} Contract"[:60]

    def _get_primary_clause_type(self, clause_types: List[str]) -> str:
        """Get the most important clause type for title generation."""

        # Priority order for title generation
        priority_types = [
            "license_grant",
            "non_compete",
            "revenue_profit_sharing",
            "termination_convenience",
            "governing_law",
            "parties",
            "effective_date",
        ]

        # Find first priority type that exists
        for priority_type in priority_types:
            if priority_type in clause_types:
                return priority_type

        # If no priority types, return the first clause type
        return clause_types[0] if clause_types else ""

    def _create_similarity_relationships(self, extractions: List[Dict]):
        """Create similarity relationships between clauses."""

        if not self.extraction_config["similarity_processing"]["enabled"]:
            return

        logger.info("Creating similarity relationships...")

        # Group clauses by type for similarity comparison
        clauses_by_type = defaultdict(list)
        for extraction in extractions:
            clause_type = extraction.get("clause_type", "unknown")
            clause_id = self._generate_clause_id(
                extraction.get("contract_id", ""), extraction
            )
            clauses_by_type[clause_type].append(
                {
                    "clause_id": clause_id,
                    "text": extraction.get("original_text", ""),
                    "extraction": extraction,
                }
            )

        # Create similarity relationships within each clause type
        threshold = self.extraction_config["similarity_processing"]["threshold"]
        similarity_data = []

        for clause_type, clauses in clauses_by_type.items():
            if len(clauses) < 2:
                continue

            # Simple similarity based on text overlap
            for i, clause1 in enumerate(clauses):
                for j, clause2 in enumerate(clauses[i + 1 :], i + 1):
                    similarity = self._calculate_text_similarity(
                        clause1["text"], clause2["text"]
                    )

                    if similarity >= threshold:
                        similarity_data.append(
                            {
                                "clause1_id": clause1["clause_id"],
                                "clause2_id": clause2["clause_id"],
                                "score": similarity,
                            }
                        )

        if similarity_data:
            self.graph_client.create_similarity_relationships(similarity_data)
            self.stats["relationships_created"] += len(similarity_data)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation (placeholder for actual similarity matrix)."""

        if not text1 or not text2:
            return 0.0

        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _process_templates(self, processed_contract_ids: List[str]):
        """Process generated templates ONLY for the contracts we actually processed - FIXED VERSION."""

        logger.info(
            f"Processing templates for {len(processed_contract_ids)} contracts..."
        )

        # Find template directories
        template_base_dir = Path("outputs/generated_templates")
        if not template_base_dir.exists():
            logger.warning("No generated templates directory found")
            return

        templates_processed = 0

        # Create contract slug mapping for matching with proper slugify handling
        contract_slug_mapping = {}
        for contract_id in processed_contract_ids:
            if SLUGIFY_AVAILABLE:
                contract_slug = slugify(contract_id, max_length=100)
            else:
                contract_slug = self._fallback_slugify(contract_id, max_length=100)

            contract_slug_mapping[contract_slug] = contract_id
            # Also add variations for better matching
            contract_slug_mapping[contract_id] = contract_id
            contract_slug_mapping[contract_id.lower()] = contract_id
            contract_slug_mapping[contract_id.replace("_", "-")] = contract_id

        # Only process template directories that match our processed contracts
        for contract_dir in template_base_dir.iterdir():
            if not contract_dir.is_dir():
                continue

            # Better matching logic
            matched_contract_id = None

            # Try exact match first
            if contract_dir.name in contract_slug_mapping:
                matched_contract_id = contract_slug_mapping[contract_dir.name]
            else:
                # Try fuzzy matching
                for slug, contract_id in contract_slug_mapping.items():
                    if (
                        contract_dir.name in slug
                        or slug in contract_dir.name
                        or contract_dir.name.replace("-", "_") == slug.replace("-", "_")
                    ):
                        matched_contract_id = contract_id
                        break

            if not matched_contract_id:
                logger.debug(
                    f"Skipping template directory (no match): {contract_dir.name}"
                )
                continue

            logger.info(
                f"Processing templates for contract: {contract_dir.name} -> {matched_contract_id}"
            )

            # Each contract directory contains clause template directories
            for clause_dir in contract_dir.iterdir():
                if not clause_dir.is_dir():
                    continue

                try:
                    template_data = self._extract_template_metadata(
                        contract_dir.name, clause_dir
                    )
                    if template_data:
                        # Create template node
                        template_id = self.graph_client.create_template_node(
                            template_data
                        )
                        if template_id:
                            self.stats["templates_created"] += 1

                            # Create relationships with the matched contract ID
                            self._create_template_relationships_fixed(
                                template_data, matched_contract_id, clause_dir
                            )

                            templates_processed += 1
                            logger.debug(
                                f"✓ Processed template: {template_data['template_id']}"
                            )

                except Exception as e:
                    logger.warning(f"Failed to process template {clause_dir}: {e}")
                    self.stats["errors"] += 1

        logger.info(
            f"Processed {templates_processed} templates for {len(processed_contract_ids)} contracts"
        )

    def _extract_template_metadata(
        self, contract_slug: str, clause_dir: Path
    ) -> Optional[Dict]:
        """Extract metadata from a template directory."""

        # Check for required files
        required_files = {
            "package": clause_dir / "package.json",
            "model": clause_dir / "model" / "model.cto",
            "grammar": clause_dir / "text" / "grammar.tem.md",
            "readme": clause_dir / "README.md",
        }

        # Verify essential files exist
        if not all(
            f.exists() for f in [required_files["package"], required_files["grammar"]]
        ):
            logger.debug(f"Template incomplete: {clause_dir}")
            return None

        # Extract from package.json
        try:
            with open(required_files["package"], "r", encoding="utf-8") as f:
                package_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read package.json from {clause_dir}: {e}")
            return None

        # Extract clause type from directory name
        clause_type = self._extract_clause_type_from_template_path(clause_dir.name)

        # Check if original text was preserved
        original_text_preserved = self._check_original_text_preservation(
            required_files["grammar"]
        )

        # Calculate coverage score from README if available
        coverage_score = self._extract_coverage_from_readme(required_files["readme"])

        return {
            "template_id": f"{contract_slug}_{clause_dir.name}",
            "template_name": package_data.get("displayName", clause_dir.name),
            "priority": self._detect_template_priority(clause_type),
            "generated_at": datetime.fromtimestamp(
                clause_dir.stat().st_mtime
            ).isoformat(),
            "coverage_score": coverage_score,
            "clause_type": clause_type,
            "original_text_preserved": original_text_preserved,
            "file_path": str(clause_dir).replace("\\", "/"),  # Fix Windows path issues
        }

    def _extract_clause_type_from_template_path(self, dir_name: str) -> str:
        """Extract clause type from template directory name."""
        # Handle numbered directories like "parties-001" or just "parties"
        if "-" in dir_name and dir_name.split("-")[-1].isdigit():
            return dir_name.rsplit("-", 1)[0]
        return dir_name

    def _check_original_text_preservation(self, grammar_file: Path) -> bool:
        """Check if template preserved original legal text."""
        if not grammar_file.exists():
            return False

        try:
            with open(grammar_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for CiceroMark variables - indicates original text was templated
            variable_count = len(re.findall(r"\{\{[^}]+\}\}", content))

            # Check for substantial text content (not just variables)
            text_without_variables = re.sub(r"\{\{[^}]+\}\}", "", content)
            substantial_text = len(text_without_variables.strip()) > 100

            return variable_count > 0 and substantial_text

        except Exception as e:
            logger.debug(f"Failed to check text preservation: {e}")
            return False

    def _extract_coverage_from_readme(self, readme_file: Path) -> float:
        """Extract coverage score from README if available."""
        if not readme_file.exists():
            return 0.0

        try:
            with open(readme_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for coverage percentage in README
            coverage_match = re.search(
                r"coverage[:\s]*([0-9.]+)%", content, re.IGNORECASE
            )
            if coverage_match:
                return float(coverage_match.group(1)) / 100.0

            # Look for variable coverage format
            var_coverage_match = re.search(
                r"variable\s+coverage[:\s]*([0-9.]+)%", content, re.IGNORECASE
            )
            if var_coverage_match:
                return float(var_coverage_match.group(1)) / 100.0

        except Exception as e:
            logger.debug(f"Failed to extract coverage from README: {e}")

        return 0.0

    def _detect_template_priority(self, clause_type: str) -> str:
        """Detect template priority based on clause type."""
        # Use the extraction config priority mapping if available
        priority_mapping = self.extraction_config.get("template_priorities", {})

        for priority, clause_types in priority_mapping.items():
            if clause_type in clause_types:
                return priority

        # Default priority classification
        essential_types = [
            "parties",
            "governing_law",
            "effective_date",
            "expiration_date",
        ]
        high_types = ["termination_convenience", "revenue_profit_sharing", "liability"]

        if clause_type in essential_types:
            return "essential"
        elif clause_type in high_types:
            return "high"
        else:
            return "medium"

    def _create_template_relationships_fixed(
        self, template_data: Dict, contract_id: str, clause_dir: Path
    ):
        """Create relationships between templates and other nodes - FIXED VERSION."""

        template_id = template_data["template_id"]
        clause_type = template_data["clause_type"]

        try:
            # Direct contract relationship using the matched contract_id
            self.graph_client.create_relationship(
                template_id,
                contract_id,
                "BASED_ON",
                "Template",
                "Contract",
                {"template_type": "accord_project"},
            )
            self.stats["relationships_created"] += 1
            logger.debug(
                f"✓ Created Template->Contract: {template_id} -> {contract_id}"
            )

            # Better clause matching
            clause_id = self._find_matching_clause_fixed(contract_id, clause_type)

            if clause_id:
                # Create Clause -> Template relationship
                self.graph_client.create_relationship(
                    clause_id,
                    template_id,
                    "GENERATES",
                    "Clause",
                    "Template",
                    {
                        "generated_at": template_data["generated_at"],
                        "coverage_score": template_data["coverage_score"],
                    },
                )
                self.stats["relationships_created"] += 1
                logger.debug(
                    f"✓ Created Clause->Template: {clause_id} -> {template_id}"
                )

                # Create Template -> Variable relationships
                self._link_template_to_variables(template_id, clause_id)
            else:
                logger.warning(
                    f"No matching clause found for {clause_type} in contract {contract_id}"
                )

        except Exception as e:
            logger.error(
                f"Failed to create relationships for template {template_id}: {e}"
            )

    def _find_matching_clause_fixed(
        self, contract_id: str, clause_type: str
    ) -> Optional[str]:
        """Find matching clause with better matching logic."""

        # Try exact match first
        clause_query = """
        MATCH (c:Contract {contract_id: $contract_id})-[:CONTAINS]->(cl:Clause)
        WHERE cl.clause_type = $clause_type
        RETURN cl.clause_id
        LIMIT 1
        """

        result = self.graph_client.execute_query(
            clause_query, {"contract_id": contract_id, "clause_type": clause_type}
        )

        if result:
            return result[0]["cl.clause_id"]

        # Try variations of clause type
        clause_variations = [
            clause_type.replace("-", "_"),
            clause_type.replace("_", "-"),
            clause_type.lower(),
            clause_type.replace("-", "_").lower(),
            clause_type.replace("_", "-").lower(),
        ]

        for variation in clause_variations:
            result = self.graph_client.execute_query(
                clause_query, {"contract_id": contract_id, "clause_type": variation}
            )
            if result:
                logger.debug(
                    f"Matched clause type variation: {clause_type} -> {variation}"
                )
                return result[0]["cl.clause_id"]

        # Try partial matching as last resort
        partial_query = """
        MATCH (c:Contract {contract_id: $contract_id})-[:CONTAINS]->(cl:Clause)
        WHERE toLower(cl.clause_type) CONTAINS toLower($clause_type)
           OR toLower($clause_type) CONTAINS toLower(cl.clause_type)
        RETURN cl.clause_id, cl.clause_type
        ORDER BY length(cl.clause_type)
        LIMIT 1
        """

        result = self.graph_client.execute_query(
            partial_query, {"contract_id": contract_id, "clause_type": clause_type}
        )

        if result:
            logger.debug(
                f"Partial clause match: {clause_type} -> {result[0]['cl.clause_type']}"
            )
            return result[0]["cl.clause_id"]

        return None

    def _link_template_to_variables(self, template_id: str, clause_id: str):
        """Link template to the variables it uses."""

        # Find all variables for this clause
        variable_query = """
        MATCH (cl:Clause {clause_id: $clause_id})-[:HAS_VARIABLE]->(v:Variable)
        RETURN v.var_name
        """

        variables = self.graph_client.execute_query(
            variable_query, {"clause_id": clause_id}
        )

        for var_result in variables:
            var_name = var_result["v.var_name"]

            # Create Template -> Variable relationship
            self.graph_client.create_relationship(
                template_id,
                var_name,
                "USES_VARIABLES",
                "Template",
                "Variable",
                {"variable_usage": "template_placeholder"},
            )
            self.stats["relationships_created"] += 1

    def get_graph_summary(self) -> Dict:
        """Get summary of the created graph."""

        graph_stats = self.graph_client.get_stats()
        sample_queries = self.graph_client.run_sample_queries()

        return {
            "processing_stats": self.stats,
            "graph_stats": graph_stats,
            "sample_data": sample_queries,
        }

    def close(self):
        """Close the graph client connection."""
        self.graph_client.close()


def main():
    """Command line interface for building the knowledge graph."""

    import argparse

    parser = argparse.ArgumentParser(description="Build Legal Contract Knowledge Graph")
    parser.add_argument(
        "--extraction-file",
        "-e",
        help="Path to extraction data file (auto-detects latest if not provided)",
    )
    parser.add_argument(
        "--max-contracts", "-m", type=int, help="Maximum number of contracts to process"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing graph before building"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Build the graph
        builder = LegalKnowledgeGraphBuilder()

        results = builder.build_graph_from_extractions(
            extraction_file=args.extraction_file,  # Can be None for auto-detection
            max_contracts=args.max_contracts,
            clear_existing=args.clear,
        )

        print("\n" + "=" * 50)
        print("KNOWLEDGE GRAPH BUILD COMPLETE")
        print("=" * 50)
        print(
            f"Contracts processed: {results['processing_stats']['contracts_processed']}"
        )
        print(f"Parties created: {results['processing_stats']['parties_created']}")
        print(f"Clauses created: {results['processing_stats']['clauses_created']}")
        print(f"Variables created: {results['processing_stats']['variables_created']}")
        print(f"Templates created: {results['processing_stats']['templates_created']}")
        print(
            f"Relationships created: {results['processing_stats']['relationships_created']}"
        )
        print(f"LLM calls: {results['processing_stats']['llm_calls']}")
        print(f"Duration: {results['duration_seconds']:.1f} seconds")

        print(f"\nGraph Statistics:")
        for stat_name, count in results["graph_stats"].items():
            print(f"  {stat_name}: {count}")

        # Get sample data
        summary = builder.get_graph_summary()
        sample_data = summary["sample_data"]

        if sample_data.get("top_parties"):
            print(f"\nTop Parties by Contract Count:")
            for party in sample_data["top_parties"][:3]:
                print(
                    f"  {party.get('p.name', 'Unknown')}: {party.get('contract_count', 0)} contracts"
                )

        builder.close()

    except Exception as e:
        logger.error(f"Graph building failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
