"""
Intelligent State Machine Generator for Smart Legal Contracts
Uses LLMs and NLP to intelligently extract states and transitions from contract text
Enhanced with robust JSON parsing, FSM validation, and improved state classification
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import sys
from hashlib import sha256
from ollama import chat
import spacy

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContractState:
    """Represents a contract lifecycle state"""

    name: str
    description: str
    triggers: List[str]
    source_clause_types: List[str]
    confidence: float
    extracted_evidence: str
    is_terminal: bool = False
    is_parallel: bool = False


@dataclass
class StateTransition:
    """Represents a transition between states"""

    source_state: str
    target_state: str
    trigger_event: str
    conditions: List[str]
    source_text: str
    confidence: float
    reasoning: str


class RobustJSONParser:
    """Enhanced JSON parser with multiple fallback strategies"""

    def __init__(self):
        self.parsing_stats = {
            "successful_parses": 0,
            "failed_parses": 0,
            "fallback_used": 0,
            "manual_extraction_used": 0,
        }

    def parse_llm_response(
        self, response: str, expected_structure: str = "states"
    ) -> Dict:
        """Parse LLM response with progressive fallback strategies"""

        if not response or not response.strip():
            logger.warning("Empty LLM response received")
            self.parsing_stats["failed_parses"] += 1
            return self._generate_empty_response(expected_structure)

        # Strategy 1: Progressive JSON cleaning
        for attempt in range(3):
            try:
                if attempt == 0:
                    cleaned = self._basic_json_cleaning(response)
                elif attempt == 1:
                    cleaned = self._aggressive_json_cleaning(response)
                else:
                    cleaned = self._extract_json_core(response)

                parsed = json.loads(cleaned)

                if self._validate_structure(parsed, expected_structure):
                    self.parsing_stats["successful_parses"] += 1
                    return parsed

            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing attempt {attempt + 1} failed: {e}")
                continue
            except Exception as e:
                logger.debug(f"Unexpected error in attempt {attempt + 1}: {e}")
                continue

        # Strategy 2: Manual field extraction
        logger.warning("JSON parsing failed, attempting manual extraction")
        manual_result = self._manual_field_extraction(response, expected_structure)

        if manual_result and self._validate_structure(
            manual_result, expected_structure
        ):
            self.parsing_stats["manual_extraction_used"] += 1
            return manual_result

        # Strategy 3: Fallback response
        logger.warning("All parsing strategies failed, using fallback")
        self.parsing_stats["fallback_used"] += 1
        return self._generate_fallback_response(response, expected_structure)

    def _basic_json_cleaning(self, response: str) -> str:
        """Basic JSON cleaning operations"""

        # Remove markdown code blocks
        response = re.sub(
            r"^```json\s*", "", response, flags=re.IGNORECASE | re.MULTILINE
        )
        response = re.sub(r"\s*```\s*$", "", response, flags=re.MULTILINE)

        # Extract JSON boundaries
        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            response = response[json_start:json_end]

        # Basic fixes
        response = re.sub(r",(\s*[}\]])", r"\1", response)  # Remove trailing commas
        response = re.sub(r"(\w+)(\s*:)", r'"\1"\2', response)  # Quote property names

        return response.strip()

    def _aggressive_json_cleaning(self, response: str) -> str:
        """More aggressive JSON cleaning"""

        response = self._basic_json_cleaning(response)

        # Fix escape sequences
        response = re.sub(r'\\(?!["\\/bfnrt])', r"\\\\", response)

        # Fix unescaped quotes within strings
        response = re.sub(r'(?<!\\)"(?=[^"]*"[^"]*:)', r'\\"', response)

        # Fix single quotes to double quotes (carefully)
        response = re.sub(r"'([^']*)'(\s*[:}])", r'"\1"\2', response)

        # Remove problematic characters
        response = re.sub(r"[^\x20-\x7E]", "", response)

        # Fix boolean and null values
        response = re.sub(r":\s*true\b", ": true", response)
        response = re.sub(r":\s*false\b", ": false", response)
        response = re.sub(r":\s*null\b", ": null", response)

        return response

    def _extract_json_core(self, response: str) -> str:
        """Extract core JSON structure by rebuilding"""

        # Try to find and extract the main array or object
        if '"states"' in response:
            return self._rebuild_states_json(response)
        elif '"transitions"' in response:
            return self._rebuild_transitions_json(response)

        # Fallback to basic cleaning
        return self._basic_json_cleaning(response)

    def _rebuild_states_json(self, response: str) -> str:
        """Rebuild states JSON structure"""

        states = []

        # Extract individual state objects
        state_pattern = r'\{[^{}]*"name"[^{}]*\}'
        state_matches = re.findall(state_pattern, response)

        for state_match in state_matches:
            try:
                # Extract key-value pairs
                name = self._extract_quoted_value(state_match, "name")
                description = self._extract_quoted_value(state_match, "description")
                confidence = self._extract_numeric_value(state_match, "confidence")
                evidence = self._extract_quoted_value(state_match, "evidence")
                is_terminal = self._extract_boolean_value(state_match, "is_terminal")

                if name:  # Only add if we have at least a name
                    state_obj = {
                        "name": name,
                        "description": description or "Contract state",
                        "confidence": confidence or 0.7,
                        "evidence": evidence or "Extracted from contract analysis",
                        "is_terminal": is_terminal or False,
                    }
                    states.append(state_obj)

            except Exception as e:
                logger.debug(f"Failed to parse state object: {e}")
                continue

        return json.dumps({"states": states})

    def _rebuild_transitions_json(self, response: str) -> str:
        """Rebuild transitions JSON structure"""

        transitions = []

        # Extract transition objects
        transition_pattern = r'\{[^{}]*"source_state"[^{}]*\}'
        transition_matches = re.findall(transition_pattern, response)

        for trans_match in transition_matches:
            try:
                source_state = self._extract_quoted_value(trans_match, "source_state")
                target_state = self._extract_quoted_value(trans_match, "target_state")
                trigger_event = self._extract_quoted_value(trans_match, "trigger_event")
                confidence = self._extract_numeric_value(trans_match, "confidence")
                reasoning = self._extract_quoted_value(trans_match, "reasoning")

                if source_state and target_state:
                    trans_obj = {
                        "source_state": source_state,
                        "target_state": target_state,
                        "trigger_event": trigger_event
                        or f"{source_state.upper()}_TO_{target_state.upper()}",
                        "confidence": confidence or 0.7,
                        "reasoning": reasoning or "LLM-determined transition",
                    }
                    transitions.append(trans_obj)

            except Exception as e:
                logger.debug(f"Failed to parse transition object: {e}")
                continue

        return json.dumps({"transitions": transitions})

    def _extract_quoted_value(self, text: str, key: str) -> Optional[str]:
        """Extract quoted string value for a given key"""
        pattern = rf'"{key}"\s*:\s*"([^"]*)"'
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _extract_numeric_value(self, text: str, key: str) -> Optional[float]:
        """Extract numeric value for a given key"""
        pattern = rf'"{key}"\s*:\s*([0-9.]+)'
        match = re.search(pattern, text)
        try:
            return float(match.group(1)) if match else None
        except ValueError:
            return None

    def _extract_boolean_value(self, text: str, key: str) -> Optional[bool]:
        """Extract boolean value for a given key"""
        pattern = rf'"{key}"\s*:\s*(true|false)'
        match = re.search(pattern, text)
        return match.group(1) == "true" if match else None

    def _manual_field_extraction(self, response: str, expected_structure: str) -> Dict:
        """Manual field extraction using regex patterns"""

        if expected_structure == "states":
            return self._extract_states_manually(response)
        elif expected_structure == "transitions":
            return self._extract_transitions_manually(response)

        return {}

    def _extract_states_manually(self, response: str) -> Dict:
        """Extract states using regex patterns"""

        states = []

        # Pattern for state information
        patterns = [
            r'"name"\s*:\s*"([^"]+)".*?"description"\s*:\s*"([^"]+)".*?"confidence"\s*:\s*([0-9.]+)',
            r"name[:\s]*([A-Za-z_]+).*?description[:\s]*([^,}\n]+).*?confidence[:\s]*([0-9.]+)",
            r"([a-z_]+)\s*state.*?confidence[:\s]*([0-9.]+)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)

            for match in matches:
                try:
                    if len(match.groups()) >= 3:
                        name, description, confidence = match.groups()[:3]
                    elif len(match.groups()) == 2:
                        name, confidence = match.groups()
                        description = f"{name} state"
                    else:
                        continue

                    # Clean extracted values
                    name = re.sub(r"[^a-z_]", "", name.lower())
                    confidence = float(confidence) if confidence else 0.7

                    if name:
                        states.append(
                            {
                                "name": name,
                                "description": description.strip().strip("\"'"),
                                "confidence": min(max(confidence, 0.0), 1.0),
                                "evidence": "Extracted from LLM response",
                                "is_terminal": "terminated" in name
                                or "expired" in name,
                            }
                        )

                except (ValueError, AttributeError) as e:
                    logger.debug(f"Failed to parse manual extraction: {e}")
                    continue

            if states:  # If we found states with this pattern, break
                break

        return {"states": states} if states else {}

    def _extract_transitions_manually(self, response: str) -> Dict:
        """Extract transitions using regex patterns"""

        transitions = []

        # Look for transition patterns
        patterns = [
            r"([a-z_]+)\s*(?:->|to|transitions?\s+to)\s*([a-z_]+)",
            r'"source_state"\s*:\s*"([^"]+)".*?"target_state"\s*:\s*"([^"]+)"',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)

            for match in matches:
                try:
                    source, target = match.groups()
                    source = re.sub(r"[^a-z_]", "", source.lower())
                    target = re.sub(r"[^a-z_]", "", target.lower())

                    if source and target and source != target:
                        transitions.append(
                            {
                                "source_state": source,
                                "target_state": target,
                                "trigger_event": f"{source.upper()}_TO_{target.upper()}",
                                "confidence": 0.6,
                                "reasoning": "Extracted from manual parsing",
                            }
                        )

                except Exception as e:
                    logger.debug(f"Failed to parse manual transition: {e}")
                    continue

        return {"transitions": transitions} if transitions else {}

    def _validate_structure(self, parsed: Dict, expected_structure: str) -> bool:
        """Validate parsed structure"""

        if not isinstance(parsed, dict):
            return False

        if expected_structure == "states":
            return (
                "states" in parsed
                and isinstance(parsed["states"], list)
                and len(parsed["states"]) > 0
                and all(
                    isinstance(state, dict) and "name" in state
                    for state in parsed["states"]
                )
            )
        elif expected_structure == "transitions":
            return (
                "transitions" in parsed
                and isinstance(parsed["transitions"], list)
                and all(
                    isinstance(trans, dict)
                    and "source_state" in trans
                    and "target_state" in trans
                    for trans in parsed["transitions"]
                )
            )

        return True

    def _generate_empty_response(self, expected_structure: str) -> Dict:
        """Generate empty response structure"""
        if expected_structure == "states":
            return {"states": []}
        elif expected_structure == "transitions":
            return {"transitions": []}
        return {}

    def _generate_fallback_response(
        self, response: str, expected_structure: str
    ) -> Dict:
        """Generate fallback response based on common contract states"""

        if expected_structure == "states":
            # Generate reasonable default states based on response content
            fallback_states = []
            common_states = [
                ("draft", "Contract is in draft form"),
                ("under_review", "Contract is being reviewed"),
                ("approved", "Contract has been approved"),
                ("signed", "Contract has been signed"),
                ("effective", "Contract is effective"),
                ("active", "Contract is actively operating"),
                ("terminated", "Contract has been terminated"),
            ]

            response_lower = response.lower()
            for state_name, description in common_states:
                if (
                    state_name.replace("_", " ") in response_lower
                    or state_name in response_lower
                    or any(word in response_lower for word in state_name.split("_"))
                ):

                    fallback_states.append(
                        {
                            "name": state_name,
                            "description": description,
                            "confidence": 0.5,
                            "evidence": "Generated from fallback analysis",
                            "is_terminal": state_name in ["terminated", "expired"],
                        }
                    )

            return {
                "states": (
                    fallback_states
                    if fallback_states
                    else [
                        {
                            "name": "active",
                            "description": "Contract is in active state",
                            "confidence": 0.5,
                            "evidence": "Default fallback state",
                            "is_terminal": False,
                        }
                    ]
                )
            }

        elif expected_structure == "transitions":
            return {"transitions": []}

        return {}

    def get_parsing_statistics(self) -> Dict:
        """Get parsing performance statistics"""
        total_attempts = sum(self.parsing_stats.values())
        if total_attempts == 0:
            return self.parsing_stats

        return {
            **self.parsing_stats,
            "success_rate": self.parsing_stats["successful_parses"] / total_attempts,
            "fallback_rate": self.parsing_stats["fallback_used"] / total_attempts,
            "manual_extraction_rate": self.parsing_stats["manual_extraction_used"]
            / total_attempts,
        }


class IntelligentStateMachineGenerator:
    """Extract contract states using LLMs and intelligent analysis"""

    def __init__(self, config_path: str = "config/state_machine_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Setup caching for LLM responses
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.llm_cache_path = self.cache_dir / "state_extraction_cache.json"
        self.llm_cache = self._load_cache()

        # Initialize enhanced JSON parser
        self.json_parser = RobustJSONParser()

        # Initialize NLP pipeline
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model not found. Install with: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

        # FSM validation statistics
        self.fsm_validation_stats = {
            "total_fsms_validated": 0,
            "unreachable_states_found": 0,
            "non_deterministic_transitions": 0,
            "orphaned_states_found": 0,
            "validation_warnings": 0,
        }

        logger.info(
            "Initialized Intelligent State Machine Generator with enhanced JSON parsing and FSM validation"
        )

    def _load_config(self) -> Dict:
        """Load state machine configuration"""
        if not self.config_path.exists():
            # Create default config if not found
            default_config = {
                "xstate_templates": {
                    "basic_contract": {
                        "id": "contract-state-machine",
                        "initial": "draft",
                        "context": {
                            "contractId": "",
                            "parties": [],
                            "effectiveDate": "",
                            "expirationDate": "",
                        },
                        "states": {},
                    }
                }
            }
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            return default_config

        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_cache(self) -> Dict:
        """Load LLM response cache"""
        if self.llm_cache_path.exists():
            try:
                with open(self.llm_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Save LLM response cache"""
        try:
            with open(self.llm_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.llm_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

    def _get_structured_prompt(self, content: str, task_type: str) -> str:
        """Generate structured prompt for reliable JSON output"""

        if task_type == "states":
            format_example = """
{
 "states": [
   {
     "name": "draft",
     "confidence": 0.9,
     "evidence": "text excerpt showing this state exists",
     "description": "Contract is in draft form",
     "is_terminal": false
   }
 ]
}"""

            return f"""You are a legal contract analysis expert. You must respond with ONLY valid JSON in the exact format specified.

Analyze this contract text and identify the key lifecycle states this contract goes through.

Consider the ENTIRE contract lifecycle from creation to termination. Look for:
- Formation states (draft, negotiation, approval, signing)
- Operational states (effective, active, performing)
- Change states (amendment, renewal, suspension)  
- Termination states (notice period, terminating, terminated, post-termination)

For each state you identify, provide:
1. State name (concise, lowercase_with_underscores)
2. Confidence (0.0-1.0)
3. Evidence (specific text excerpt that indicates this state)
4. Description (what this state means)
5. Is it terminal (true/false)

CONTRACT TEXT:
{content}

CRITICAL REQUIREMENTS:
- Response must be valid JSON only
- No markdown, no explanations, no extra text
- Use double quotes for all strings
- No trailing commas
- No line breaks within string values
- No unescaped characters

Expected format: {format_example}

JSON Response:"""

        elif task_type == "transitions":
            format_example = """
{
 "transitions": [
   {
     "source_state": "draft",
     "target_state": "under_review", 
     "trigger_event": "SUBMIT_FOR_REVIEW",
     "confidence": 0.9,
     "reasoning": "Draft contracts typically go to review next"
   }
 ]
}"""

            return f"""You are a legal contract analysis expert. You must respond with ONLY valid JSON.

Given these contract lifecycle states, determine the logical transitions between them.

{content}

For each logical transition, provide:
1. Source state name
2. Target state name  
3. Trigger event (what causes this transition)
4. Confidence (0.0-1.0)
5. Reasoning (why this transition makes sense)

Consider standard contract flow: draft → review → approval → signing → effectiveness → active operation → renewal/termination

CRITICAL REQUIREMENTS:
- Response must be valid JSON only
- No markdown, no explanations, no extra text
- Use double quotes for all strings
- No trailing commas
- No unescaped characters

Expected format: {format_example}

JSON Response:"""

        return content

    def _get_llm_options(self) -> Dict:
        """Get optimized options for JSON generation"""
        return {
            "temperature": 0.1,
            "seed": 42,
            "num_predict": 1000,
            "top_p": 0.9,
            "stop": ["}]}", "}\n\n", "```", "\n\n\n"],
        }

    def _query_llm_with_robust_parsing(
        self, prompt: str, cache_key: str, task_type: str
    ) -> Dict:
        """Query LLM with robust JSON parsing"""

        if cache_key in self.llm_cache:
            cached_response = self.llm_cache[cache_key]
            if isinstance(cached_response, dict):
                return cached_response
            else:
                # Re-parse cached string response
                parsed = self.json_parser.parse_llm_response(cached_response, task_type)
                self.llm_cache[cache_key] = parsed  # Update cache with parsed version
                return parsed

        try:
            response = chat(
                model="llama3",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal expert. Always respond with valid JSON only. Never use markdown or explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options=self._get_llm_options(),
            )

            raw_response = response["message"]["content"].strip()

            # Use robust JSON parser
            parsed_result = self.json_parser.parse_llm_response(raw_response, task_type)

            # Cache the parsed result
            self.llm_cache[cache_key] = parsed_result
            self._save_cache()

            return parsed_result

        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return self.json_parser._generate_empty_response(task_type)

    def extract_contract_states(
        self, extractions: List[Dict]
    ) -> Dict[str, List[ContractState]]:
        """Extract states using intelligent analysis"""
        logger.info(f"Analyzing {len(extractions)} extractions for contract states")

        contract_states = defaultdict(list)

        # Group extractions by contract for holistic analysis
        contracts = defaultdict(list)
        for extraction in extractions:
            contract_id = extraction.get("contract_id", "unknown")
            contracts[contract_id].append(extraction)

        for contract_id, contract_extractions in contracts.items():
            logger.info(f"Analyzing contract lifecycle for: {contract_id[:50]}...")

            # Analyze entire contract context for states
            contract_states[contract_id] = self._analyze_contract_lifecycle(
                contract_id, contract_extractions
            )

        logger.info(f"Extracted states for {len(contract_states)} contracts")
        return dict(contract_states)

    def _analyze_contract_lifecycle(
        self, contract_id: str, extractions: List[Dict]
    ) -> List[ContractState]:
        """Analyze entire contract for lifecycle states using LLM"""

        # Combine all contract text for holistic analysis
        contract_context = self._build_contract_context(extractions)

        if not contract_context.strip():
            return []

        # Create cache key
        context_hash = sha256(contract_context.encode()).hexdigest()[:16]
        cache_key = f"lifecycle_{context_hash}"

        # Generate structured prompt
        prompt = self._get_structured_prompt(contract_context[:3000], "states")

        # Query LLM with robust parsing
        response_data = self._query_llm_with_robust_parsing(prompt, cache_key, "states")

        # Convert to ContractState objects
        states = self._convert_to_contract_states(response_data, extractions)

        return states

    def _convert_to_contract_states(
        self, response_data: Dict, extractions: List[Dict]
    ) -> List[ContractState]:
        """Convert parsed response to ContractState objects with enhanced classification"""

        states = []

        if not response_data or "states" not in response_data:
            return states

        for state_data in response_data["states"]:
            if not isinstance(state_data, dict) or "name" not in state_data:
                continue

            source_clause_types = self._identify_source_clauses(
                state_data.get("evidence", ""), extractions
            )

            triggers = self._extract_triggers_from_evidence(
                state_data.get("evidence", ""), state_data.get("name", "")
            )

            state = ContractState(
                name=state_data.get("name", "unknown").lower().replace(" ", "_"),
                description=state_data.get("description", ""),
                triggers=triggers,
                source_clause_types=source_clause_types,
                confidence=float(state_data.get("confidence", 0.5)),
                extracted_evidence=state_data.get("evidence", ""),
                is_terminal=state_data.get("is_terminal", False),
            )

            # Enhanced state classification
            state_category = self._classify_state_category(
                state.name, state.description
            )
            state.source_clause_types.append(f"category_{state_category}")

            # Enhanced terminal state detection
            if self._is_enhanced_terminal_state(state.name, state.description):
                state.is_terminal = True

            states.append(state)

        return states

    def _classify_state_category(self, state_name: str, description: str) -> str:
        """Classify state into logical categories"""

        formation_indicators = ["draft", "review", "negotiat", "approv", "sign"]
        operational_indicators = [
            "effective",
            "active",
            "perform",
            "execut",
            "operational",
        ]
        modification_indicators = ["amend", "renew", "suspend", "modif", "chang"]
        termination_indicators = ["terminat", "expir", "end", "complet", "final"]

        combined_text = f"{state_name} {description}".lower()

        if any(indicator in combined_text for indicator in formation_indicators):
            return "formation"
        elif any(indicator in combined_text for indicator in operational_indicators):
            return "operational"
        elif any(indicator in combined_text for indicator in modification_indicators):
            return "modification"
        elif any(indicator in combined_text for indicator in termination_indicators):
            return "termination"
        else:
            return "operational"  # Default category

    def _is_enhanced_terminal_state(self, state_name: str, description: str) -> bool:
        """Enhanced terminal state detection"""

        terminal_indicators = [
            "terminated",
            "expired",
            "completed",
            "finished",
            "ended",
            "post_termination",
            "final",
            "closed",
            "concluded",
        ]

        combined_text = f"{state_name} {description}".lower()
        return any(indicator in combined_text for indicator in terminal_indicators)

    def _build_contract_context(self, extractions: List[Dict]) -> str:
        """Build comprehensive contract context from all extractions"""
        context_parts = []

        # Prioritize certain clause types that are most indicative of lifecycle
        priority_clause_types = [
            "effective_date",
            "expiration_date",
            "agreement_date",
            "termination_convenience",
            "renewal_term",
            "notice_period_renewal",
        ]

        # Add high-priority clauses first
        for clause_type in priority_clause_types:
            for extraction in extractions:
                if (
                    extraction.get("clause_type") == clause_type
                    and extraction.get("original_text", "").strip()
                ):
                    context_parts.append(
                        f"[{clause_type}] {extraction['original_text']}"
                    )

        # Add other clauses
        for extraction in extractions:
            clause_type = extraction.get("clause_type", "")
            if (
                clause_type not in priority_clause_types
                and extraction.get("original_text", "").strip()
            ):
                context_parts.append(f"[{clause_type}] {extraction['original_text']}")

        return "\n\n".join(context_parts)

    def _identify_source_clauses(
        self, evidence: str, extractions: List[Dict]
    ) -> List[str]:
        """Identify which clause types contributed to a state"""
        source_clauses = []

        for extraction in extractions:
            original_text = extraction.get("original_text", "")
            clause_type = extraction.get("clause_type", "")

            # Check if evidence appears in this clause's text
            if (
                evidence
                and original_text
                and any(
                    word in original_text.lower()
                    for word in evidence.lower().split()[:3]
                )
            ):
                source_clauses.append(clause_type)

        return list(set(source_clauses))  # Remove duplicates

    def _extract_triggers_from_evidence(
        self, evidence: str, state_name: str
    ) -> List[str]:
        """Extract trigger phrases from evidence text"""
        if not evidence:
            return []

        # Use spaCy for intelligent phrase extraction if available
        if self.nlp:
            doc = self.nlp(evidence)

            # Extract verb phrases and key actions
            triggers = []
            for token in doc:
                if token.pos_ == "VERB" and not token.is_stop:
                    # Get the verb and its immediate context
                    verb_phrase = str(token.lemma_)
                    if token.head != token:
                        verb_phrase = f"{token.head.text} {verb_phrase}"
                    triggers.append(verb_phrase)

            return triggers[:3]  # Limit to top 3

        # Simple fallback: extract key action words
        action_words = []
        words = evidence.lower().split()
        action_indicators = [
            "sign",
            "execute",
            "terminate",
            "expire",
            "renew",
            "activate",
            "effective",
        ]

        for word in words:
            if any(indicator in word for indicator in action_indicators):
                action_words.append(word)

        return action_words[:3]

    def identify_transitions(
        self, contract_states: Dict[str, List[ContractState]]
    ) -> Dict[str, List[StateTransition]]:
        """Intelligently identify state transitions using LLM analysis"""
        logger.info("Analyzing state transitions...")

        contract_transitions = {}

        for contract_id, states in contract_states.items():
            if len(states) < 2:  # Need at least 2 states for transitions
                contract_transitions[contract_id] = []
                continue

            logger.info(f"Analyzing transitions for contract: {contract_id[:50]}...")

            # Use LLM to intelligently determine transitions
            transitions = self._analyze_state_transitions(contract_id, states)
            contract_transitions[contract_id] = transitions

        logger.info(f"Identified transitions for {len(contract_transitions)} contracts")
        return contract_transitions

    def _analyze_state_transitions(
        self, contract_id: str, states: List[ContractState]
    ) -> List[StateTransition]:
        """Use LLM to analyze logical state transitions"""

        # Create state list for LLM
        state_list = [
            {
                "name": s.name,
                "description": s.description,
                "evidence": s.extracted_evidence,
            }
            for s in states
        ]

        # Create cache key
        states_hash = sha256(
            json.dumps(state_list, sort_keys=True).encode()
        ).hexdigest()[:16]
        cache_key = f"transitions_{states_hash}"

        # Generate structured prompt for transitions
        states_content = json.dumps(state_list, indent=2)
        prompt = self._get_structured_prompt(
            f"States identified:\n{states_content}", "transitions"
        )

        # Query LLM with robust parsing
        response_data = self._query_llm_with_robust_parsing(
            prompt, cache_key, "transitions"
        )

        # Convert to StateTransition objects
        transitions = self._convert_to_state_transitions(response_data, states)

        return transitions

    def _convert_to_state_transitions(
        self, response_data: Dict, states: List[ContractState]
    ) -> List[StateTransition]:
        """Convert parsed response to StateTransition objects with enhanced validation"""

        transitions = []

        if not response_data or "transitions" not in response_data:
            return self._generate_default_transitions(states)

        state_names = [s.name for s in states]

        for trans_data in response_data["transitions"]:
            if not isinstance(trans_data, dict):
                continue

            source = trans_data.get("source_state", "")
            target = trans_data.get("target_state", "")

            # Validate that both states exist and transition is valid
            if source in state_names and target in state_names:
                # Enhanced business logic validation
                if self._validate_transition_business_logic(source, target, states):
                    transition = StateTransition(
                        source_state=source,
                        target_state=target,
                        trigger_event=trans_data.get(
                            "trigger_event", f"{source.upper()}_TO_{target.upper()}"
                        ),
                        conditions=[],
                        source_text="",
                        confidence=float(trans_data.get("confidence", 0.7)),
                        reasoning=trans_data.get(
                            "reasoning", "LLM-determined transition"
                        ),
                    )
                    transitions.append(transition)
                else:
                    logger.debug(
                        f"Invalid business logic for transition {source} -> {target}"
                    )

        # If no valid transitions extracted, use fallback
        if not transitions:
            transitions = self._generate_default_transitions(states)

        return transitions

    def _validate_transition_business_logic(
        self, source: str, target: str, states: List[ContractState]
    ) -> bool:
        """Validate transition follows business logic rules"""

        # Get state categories
        source_category = self._get_state_category(source, states)
        target_category = self._get_state_category(target, states)

        # Business logic rules
        invalid_transitions = [
            ("termination", "formation"),  # Can't go from terminated back to formation
            (
                "termination",
                "operational",
            ),  # Can't go from terminated back to operational
        ]

        # Check for invalid transitions
        if (source_category, target_category) in invalid_transitions:
            return False

        # Don't allow self-transitions (state to itself)
        if source == target:
            return False

        # Check for terminal state violations
        source_state = next((s for s in states if s.name == source), None)
        if source_state and source_state.is_terminal and target != source:
            # Terminal states should not have outgoing transitions (except self-loops)
            return False

        return True

    def _get_state_category(self, state_name: str, states: List[ContractState]) -> str:
        """Get category for a specific state"""
        state = next((s for s in states if s.name == state_name), None)
        if state:
            # Extract category from source_clause_types
            for clause_type in state.source_clause_types:
                if clause_type.startswith("category_"):
                    return clause_type.replace("category_", "")
        return "operational"  # Default

    def _generate_default_transitions(
        self, states: List[ContractState]
    ) -> List[StateTransition]:
        """Generate sensible default transitions if LLM parsing fails"""
        transitions = []
        state_names = [s.name for s in states]

        # Common transition patterns
        common_flows = [
            ("draft", "under_review"),
            ("under_review", "approved"),
            ("approved", "signed"),
            ("signed", "effective"),
            ("effective", "active"),
            ("active", "terminated"),
            ("active", "expired"),
            ("active", "renewal_notice_period"),
            ("renewal_notice_period", "active"),
            ("terminated", "post_termination_obligations"),
        ]

        for source, target in common_flows:
            if source in state_names and target in state_names:
                # Apply business logic validation even for defaults
                if self._validate_transition_business_logic(source, target, states):
                    transition = StateTransition(
                        source_state=source,
                        target_state=target,
                        trigger_event=f"{source.upper()}_TO_{target.upper()}",
                        conditions=[],
                        source_text="",
                        confidence=0.6,
                        reasoning="Default transition pattern",
                    )
                    transitions.append(transition)

        return transitions

    def _validate_fsm_properties(
        self, states_dict: Dict, transitions: List[StateTransition]
    ) -> Dict:
        """Validate formal FSM properties and return validation results"""

        self.fsm_validation_stats["total_fsms_validated"] += 1
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "reachability_analysis": {},
            "determinism_check": True,
        }

        state_names = set(states_dict.keys())

        # 1. Reachability Analysis
        reachable_states = self._analyze_reachability(states_dict, transitions)
        unreachable_states = state_names - reachable_states

        if unreachable_states:
            self.fsm_validation_stats["unreachable_states_found"] += len(
                unreachable_states
            )
            validation_results["warnings"].append(
                f"Unreachable states detected: {', '.join(unreachable_states)}"
            )
            self.fsm_validation_stats["validation_warnings"] += 1

        validation_results["reachability_analysis"] = {
            "reachable_states": list(reachable_states),
            "unreachable_states": list(unreachable_states),
            "reachability_percentage": len(reachable_states) / len(state_names) * 100,
        }

        # 2. Determinism Check
        non_deterministic = self._check_determinism(transitions)
        if non_deterministic:
            self.fsm_validation_stats["non_deterministic_transitions"] += len(
                non_deterministic
            )
            validation_results["determinism_check"] = False
            validation_results["warnings"].append(
                f"Non-deterministic transitions found: {non_deterministic}"
            )
            self.fsm_validation_stats["validation_warnings"] += 1

        # 3. Orphaned States Check
        orphaned_states = self._find_orphaned_states(states_dict, transitions)
        if orphaned_states:
            self.fsm_validation_stats["orphaned_states_found"] += len(orphaned_states)
            validation_results["warnings"].append(
                f"Orphaned states (no incoming/outgoing transitions): {', '.join(orphaned_states)}"
            )
            self.fsm_validation_stats["validation_warnings"] += 1

        # 4. Terminal State Validation
        terminal_issues = self._validate_terminal_states(states_dict, transitions)
        if terminal_issues:
            validation_results["warnings"].extend(terminal_issues)
            self.fsm_validation_stats["validation_warnings"] += len(terminal_issues)

        # Overall validation status
        if validation_results["warnings"] or validation_results["errors"]:
            validation_results["is_valid"] = len(validation_results["errors"]) == 0

        return validation_results

    def _analyze_reachability(
        self, states_dict: Dict, transitions: List[StateTransition]
    ) -> Set[str]:
        """Analyze which states are reachable from initial state"""

        # Build adjacency list
        graph = defaultdict(list)
        for transition in transitions:
            graph[transition.source_state].append(transition.target_state)

        # Find initial state (assume first state if not specified)
        initial_state = next(iter(states_dict.keys())) if states_dict else None
        if not initial_state:
            return set()

        # BFS to find all reachable states
        reachable = set()
        queue = [initial_state]
        reachable.add(initial_state)

        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        return reachable

    def _check_determinism(self, transitions: List[StateTransition]) -> List[str]:
        """Check for non-deterministic transitions (same source + trigger -> multiple targets)"""

        transition_map = defaultdict(list)
        non_deterministic = []

        for transition in transitions:
            key = (transition.source_state, transition.trigger_event)
            transition_map[key].append(transition.target_state)

        for (source, trigger), targets in transition_map.items():
            if len(set(targets)) > 1:  # Multiple different targets
                non_deterministic.append(f"{source}--{trigger}-->{targets}")

        return non_deterministic

    def _find_orphaned_states(
        self, states_dict: Dict, transitions: List[StateTransition]
    ) -> List[str]:
        """Find states with no incoming or outgoing transitions"""

        states_with_incoming = set()
        states_with_outgoing = set()

        for transition in transitions:
            states_with_outgoing.add(transition.source_state)
            states_with_incoming.add(transition.target_state)

        all_states = set(states_dict.keys())
        connected_states = states_with_incoming | states_with_outgoing
        orphaned = all_states - connected_states

        return list(orphaned)

    def _validate_terminal_states(
        self, states_dict: Dict, transitions: List[StateTransition]
    ) -> List[str]:
        """Validate terminal state properties"""

        issues = []

        # Find states marked as terminal
        terminal_states = set()
        for state_name, state_config in states_dict.items():
            if state_config.get("type") == "final":
                terminal_states.add(state_name)

        # Check if terminal states have outgoing transitions
        for transition in transitions:
            if transition.source_state in terminal_states:
                issues.append(
                    f"Terminal state '{transition.source_state}' has outgoing transition to '{transition.target_state}'"
                )

        return issues

    def generate_xstate_json(
        self,
        contract_id: str,
        states: List[ContractState],
        transitions: List[StateTransition],
    ) -> Dict:
        """Generate XState JSON with intelligent state machine structure and FSM validation"""

        # Start with base template
        xstate_config = self.config["xstate_templates"]["basic_contract"].copy()
        xstate_config["id"] = (
            f"contract-{contract_id.replace('_', '-').replace(' ', '-')[:50]}"
        )

        # Update context with contract-specific info
        xstate_config["context"]["contractId"] = contract_id

        # Build states object
        states_dict = {}
        state_names = [s.name for s in states]

        # Intelligently determine initial state
        initial_candidates = ["draft", "signed", "effective", "active"]
        initial_state = None
        for candidate in initial_candidates:
            if candidate in state_names:
                initial_state = candidate
                break

        if initial_state:
            xstate_config["initial"] = initial_state
        elif state_names:
            xstate_config["initial"] = state_names[0]

        # Build state configurations
        for state in states:
            state_config = {
                "meta": {
                    "description": state.description,
                    "confidence": state.confidence,
                    "evidence": state.extracted_evidence,
                    "source_clause_types": state.source_clause_types,
                    "triggers": state.triggers,
                }
            }

            # Add transitions for this state
            state_transitions = {}
            for transition in transitions:
                if transition.source_state == state.name:
                    state_transitions[transition.trigger_event] = {
                        "target": transition.target_state,
                        "meta": {
                            "confidence": transition.confidence,
                            "reasoning": transition.reasoning,
                        },
                    }

            if state_transitions:
                state_config["on"] = state_transitions

            # Mark terminal states
            if state.is_terminal:
                state_config["type"] = "final"

            states_dict[state.name] = state_config

        xstate_config["states"] = states_dict

        # Enhanced FSM validation before final return
        validation_results = self._validate_fsm_properties(states_dict, transitions)

        # Add generation metadata with validation results
        xstate_config["meta"] = {
            "generated_by": "Intelligent State Machine Generator",
            "generation_method": "LLM-based semantic analysis with robust JSON parsing and FSM validation",
            "contract_id": contract_id,
            "total_states": len(states),
            "total_transitions": len(transitions),
            "avg_state_confidence": (
                sum(s.confidence for s in states) / len(states) if states else 0
            ),
            "parsing_statistics": self.json_parser.get_parsing_statistics(),
            "fsm_validation": validation_results,
            "fsm_validation_summary": {
                "is_valid_fsm": validation_results["is_valid"],
                "reachability_percentage": validation_results[
                    "reachability_analysis"
                ].get("reachability_percentage", 0),
                "is_deterministic": validation_results["determinism_check"],
                "validation_warnings_count": len(validation_results["warnings"]),
            },
        }

        # Log validation results
        if validation_results["warnings"]:
            logger.warning(
                f"FSM validation warnings for {contract_id}: {validation_results['warnings']}"
            )
        else:
            logger.info(f"FSM validation passed for {contract_id}")

        return xstate_config

    def create_state_diagram(
        self,
        contract_id: str,
        states: List[ContractState],
        transitions: List[StateTransition],
        format: str = "mermaid",
    ) -> str:
        """Generate intelligent state diagrams with confidence indicators"""

        if format == "mermaid":
            return self._generate_intelligent_mermaid_diagram(
                contract_id, states, transitions
            )
        elif format == "graphviz":
            return self._generate_intelligent_graphviz_diagram(
                contract_id, states, transitions
            )
        else:
            raise ValueError(f"Unsupported diagram format: {format}")

    def _generate_intelligent_mermaid_diagram(
        self,
        contract_id: str,
        states: List[ContractState],
        transitions: List[StateTransition],
    ) -> str:
        """Generate Mermaid diagram with confidence indicators"""

        lines = [
            "```mermaid",
            "stateDiagram-v2",
            f"    title: Contract Lifecycle - {contract_id[:30]}",
            "",
        ]

        # Add states with confidence indicators
        for state in states:
            state_display = state.name.replace("_", " ").title()
            confidence_pct = f"({state.confidence:.0%})"

            if state.is_terminal:
                lines.append(f"    {state.name} --> [*]")

            lines.append(
                f'    state "{state_display} {confidence_pct}" as {state.name}'
            )

        lines.append("")

        # Add transitions with confidence
        for transition in transitions:
            trigger_display = transition.trigger_event.replace("_", " ").title()
            confidence_pct = f"({transition.confidence:.0%})"

            lines.append(
                f"    {transition.source_state} --> {transition.target_state} : {trigger_display} {confidence_pct}"
            )

        lines.append("```")

        return "\n".join(lines)

    def _generate_intelligent_graphviz_diagram(
        self,
        contract_id: str,
        states: List[ContractState],
        transitions: List[StateTransition],
    ) -> str:
        """Generate Graphviz diagram with confidence-based styling"""

        lines = [
            f"digraph contract_{contract_id.replace('-', '_').replace(' ', '_')[:30]} {{",
            f'    label="Contract Lifecycle - {contract_id[:30]}";',
            '    labelloc="t";',
            "    rankdir=LR;",
            "    node [fontname=Arial];",
            "",
        ]

        # Add states with confidence-based styling
        for state in states:
            color = (
                "red"
                if state.confidence < 0.5
                else "orange" if state.confidence < 0.8 else "green"
            )
            shape = "doublecircle" if state.is_terminal else "circle"

            state_name_display = state.name.replace("_", "\\n")
            confidence_display = f"({state.confidence:.0%})"
            state_label = f"{state_name_display}\\n{confidence_display}"

            lines.append(
                f'    {state.name} [shape={shape}, color={color}, style=filled, fillcolor=white, label="{state_label}"];'
            )

        lines.append("")

        # Add transitions with confidence-based styling
        for transition in transitions:
            color = (
                "red"
                if transition.confidence < 0.5
                else "orange" if transition.confidence < 0.8 else "green"
            )
            style = "dashed" if transition.confidence < 0.7 else "solid"

            trigger_display = transition.trigger_event.replace("_", "\\n")
            confidence_display = f"({transition.confidence:.0%})"
            label = f"{trigger_display}\\n{confidence_display}"

            lines.append(
                f'    {transition.source_state} -> {transition.target_state} [label="{label}", color={color}, style={style}];'
            )

        lines.append("}")
        return "\n".join(lines)

    def generate_stately_ai_format(self, xstate_config: Dict) -> Dict:
        """Generate enhanced Stately.ai format with intelligence metadata"""

        stately_config = xstate_config.copy()

        # Enhanced Stately.ai metadata
        stately_config["meta"].update(
            {
                "stately": {
                    "version": "2023-12-01",
                    "layout": "horizontal",
                    "ai_generated": True,
                    "generation_method": "LLM semantic analysis with robust JSON parsing and FSM validation",
                    "zoom": 1.0,
                    "pan": {"x": 0, "y": 0},
                }
            }
        )

        return stately_config

    def process_contract_extractions(self, extractions: List[Dict]) -> Dict:
        """Main intelligent processing method"""

        logger.info(f"Starting analysis of {len(extractions)} extractions")

        # Extract states using LLM intelligence
        contract_states = self.extract_contract_states(extractions)

        # Identify transitions using LLM reasoning
        contract_transitions = self.identify_transitions(contract_states)

        # Generate enhanced outputs
        results = {}

        for contract_id in contract_states:
            if contract_id not in contract_transitions:
                continue

            states = contract_states[contract_id]
            transitions = contract_transitions[contract_id]

            if not states:
                continue

            # Generate XState JSON with intelligence metadata and FSM validation
            xstate_json = self.generate_xstate_json(contract_id, states, transitions)

            # Generate intelligent diagrams
            mermaid_diagram = self.create_state_diagram(
                contract_id, states, transitions, "mermaid"
            )
            graphviz_diagram = self.create_state_diagram(
                contract_id, states, transitions, "graphviz"
            )

            # Generate Stately.ai format
            stately_format = self.generate_stately_ai_format(xstate_json)

            results[contract_id] = {
                "states": [asdict(s) for s in states],
                "transitions": [asdict(t) for t in transitions],
                "xstate_json": xstate_json,
                "mermaid_diagram": mermaid_diagram,
                "graphviz_diagram": graphviz_diagram,
                "stately_ai_format": stately_format,
                "intelligence_metrics": {
                    "avg_state_confidence": sum(s.confidence for s in states)
                    / len(states),
                    "avg_transition_confidence": (
                        sum(t.confidence for t in transitions) / len(transitions)
                        if transitions
                        else 0
                    ),
                    "high_confidence_states": len(
                        [s for s in states if s.confidence > 0.8]
                    ),
                    "total_evidence_extracted": len(
                        [s for s in states if s.extracted_evidence]
                    ),
                    "state_categories": {
                        "formation": len(
                            [
                                s
                                for s in states
                                if "category_formation" in s.source_clause_types
                            ]
                        ),
                        "operational": len(
                            [
                                s
                                for s in states
                                if "category_operational" in s.source_clause_types
                            ]
                        ),
                        "modification": len(
                            [
                                s
                                for s in states
                                if "category_modification" in s.source_clause_types
                            ]
                        ),
                        "termination": len(
                            [
                                s
                                for s in states
                                if "category_termination" in s.source_clause_types
                            ]
                        ),
                    },
                    "fsm_validation_summary": xstate_json.get("meta", {}).get(
                        "fsm_validation_summary", {}
                    ),
                },
            }

        # Log parsing and validation statistics
        parsing_stats = self.json_parser.get_parsing_statistics()
        logger.info(f"JSON Parsing Statistics: {parsing_stats}")
        logger.info(f"FSM Validation Statistics: {self.fsm_validation_stats}")

        logger.info(f"Generated state machines for {len(results)} contracts")
        return results

    def save_state_machines(
        self, results: Dict, output_dir: str = "outputs/state_machines"
    ):
        """Save intelligent state machines with enhanced metadata"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for contract_id, data in results.items():
            contract_dir = (
                output_path / contract_id.replace("_", "-").replace(" ", "-")[:50]
            )
            contract_dir.mkdir(exist_ok=True)

            # Save XState JSON
            with open(contract_dir / "state_machine.json", "w", encoding="utf-8") as f:
                json.dump(data["xstate_json"], f, indent=2, ensure_ascii=False)

            # Save diagrams
            with open(contract_dir / "diagram.mermaid", "w", encoding="utf-8") as f:
                f.write(data["mermaid_diagram"])

            with open(contract_dir / "diagram.dot", "w", encoding="utf-8") as f:
                f.write(data["graphviz_diagram"])

            # Save Stately.ai format
            with open(contract_dir / "stately_ai.json", "w", encoding="utf-8") as f:
                json.dump(data["stately_ai_format"], f, indent=2, ensure_ascii=False)

            # Save enhanced summary with intelligence metrics
            summary = {
                "contract_id": contract_id,
                "total_states": len(data["states"]),
                "total_transitions": len(data["transitions"]),
                "generation_method": "LLM-based intelligent analysis with robust JSON parsing and FSM validation",
                "intelligence_metrics": data["intelligence_metrics"],
                "parsing_statistics": self.json_parser.get_parsing_statistics(),
                "fsm_validation_statistics": self.fsm_validation_stats,
                "states": data["states"],
                "transitions": data["transitions"],
            }

            with open(contract_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # Save README with enhanced explanation
            parsing_stats = self.json_parser.get_parsing_statistics()
            fsm_validation = data["intelligence_metrics"]["fsm_validation_summary"]

            readme = f"""# Contract State Machine

Contract ID: {contract_id}

## Generation Method
This state machine was generated using LLM-based semantic analysis with enhanced JSON parsing and formal FSM validation.

## Intelligence Metrics
- Average State Confidence: {data['intelligence_metrics']['avg_state_confidence']:.0%}
- Average Transition Confidence: {data['intelligence_metrics']['avg_transition_confidence']:.0%}
- High Confidence States: {data['intelligence_metrics']['high_confidence_states']}/{len(data['states'])}
- Evidence Extracted: {data['intelligence_metrics']['total_evidence_extracted']} states with textual evidence

## State Classification
- Formation States: {data['intelligence_metrics']['state_categories']['formation']}
- Operational States: {data['intelligence_metrics']['state_categories']['operational']}  
- Modification States: {data['intelligence_metrics']['state_categories']['modification']}
- Termination States: {data['intelligence_metrics']['state_categories']['termination']}

## FSM Validation Results
- Valid FSM: {fsm_validation.get('is_valid_fsm', 'Unknown')}
- Reachability: {fsm_validation.get('reachability_percentage', 0):.1f}% of states reachable
- Deterministic: {fsm_validation.get('is_deterministic', 'Unknown')}
- Validation Warnings: {fsm_validation.get('validation_warnings_count', 0)}

## JSON Parsing Performance
- Success Rate: {parsing_stats.get('success_rate', 0):.1%}
- Fallback Usage: {parsing_stats.get('fallback_rate', 0):.1%}
- Manual Extraction: {parsing_stats.get('manual_extraction_rate', 0):.1%}

## Files
- `state_machine.json` - XState configuration with confidence metadata and FSM validation
- `diagram.mermaid` - Mermaid diagram with confidence indicators
- `diagram.dot` - Graphviz diagram with confidence-based styling
- `stately_ai.json` - Stately.ai compatible format
- `summary.json` - Complete analysis summary with validation metrics

## Usage
Import `state_machine.json` into XState or upload to Stately.ai for visualization and execution.
The FSM validation ensures the state machine follows formal computational theory principles.
"""

            with open(contract_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme)

        # Save global parsing and validation statistics
        parsing_stats_file = output_path / "parsing_statistics.json"
        with open(parsing_stats_file, "w", encoding="utf-8") as f:
            json.dump(self.json_parser.get_parsing_statistics(), f, indent=2)

        fsm_validation_stats_file = output_path / "fsm_validation_statistics.json"
        with open(fsm_validation_stats_file, "w", encoding="utf-8") as f:
            json.dump(self.fsm_validation_stats, f, indent=2)

        logger.info(f"Saved state machines to {output_path}")
        logger.info(f"JSON Parsing Statistics saved to {parsing_stats_file}")
        logger.info(f"FSM Validation Statistics saved to {fsm_validation_stats_file}")


if __name__ == "__main__":
    pass
