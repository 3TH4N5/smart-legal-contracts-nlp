"""
Workflow Utilities for State Machine Generation
Support functions for XState validation, visualization, and integration
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
import subprocess
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of XState validation"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class XStateValidator:
    """Validate XState JSON configurations"""

    def __init__(self):
        self.required_fields = ["id", "initial", "states"]
        self.optional_fields = ["context", "meta", "on"]

    def validate_xstate_config(self, config: Dict) -> ValidationResult:
        """Validate XState configuration structure"""

        errors = []
        warnings = []
        suggestions = []

        # Check required fields
        for field in self.required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate states structure
        if "states" in config:
            states = config["states"]
            if not isinstance(states, dict):
                errors.append("'states' must be an object/dictionary")
            else:
                # Check initial state exists
                initial_state = config.get("initial")
                if initial_state and initial_state not in states:
                    errors.append(
                        f"Initial state '{initial_state}' not found in states"
                    )

                # Validate individual states
                for state_name, state_config in states.items():
                    state_errors = self._validate_state_config(
                        state_name, state_config, states
                    )
                    errors.extend(state_errors)

        # Check for unreachable states
        unreachable_states = self._find_unreachable_states(config)
        if unreachable_states:
            warnings.extend(
                [f"Unreachable state: {state}" for state in unreachable_states]
            )

        # Check for missing transitions
        missing_transitions = self._suggest_missing_transitions(config)
        if missing_transitions:
            suggestions.extend(missing_transitions)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

    def _validate_state_config(
        self, state_name: str, state_config: Dict, all_states: Dict
    ) -> List[str]:
        """Validate individual state configuration"""
        errors = []

        if not isinstance(state_config, dict):
            errors.append(f"State '{state_name}' configuration must be an object")
            return errors

        # Check transitions reference valid states
        if "on" in state_config:
            transitions = state_config["on"]
            if isinstance(transitions, dict):
                for event, target in transitions.items():
                    if isinstance(target, str):
                        if target not in all_states:
                            errors.append(
                                f"State '{state_name}' references unknown target state: '{target}'"
                            )
                    elif isinstance(target, dict):
                        # Complex transition object
                        if "target" in target and target["target"] not in all_states:
                            errors.append(
                                f"State '{state_name}' references unknown target state: '{target['target']}'"
                            )

        return errors

    def _find_unreachable_states(self, config: Dict) -> List[str]:
        """Find states that cannot be reached from the initial state"""

        states = config.get("states", {})
        initial_state = config.get("initial")

        if not initial_state or not states:
            return []

        # BFS to find reachable states
        reachable = set()
        queue = [initial_state]

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue

            reachable.add(current)

            # Find transitions from current state
            state_config = states.get(current, {})
            transitions = state_config.get("on", {})

            for event, target in transitions.items():
                if isinstance(target, str):
                    if target not in reachable:
                        queue.append(target)
                elif isinstance(target, dict) and "target" in target:
                    target_state = target["target"]
                    if target_state not in reachable:
                        queue.append(target_state)

        # Return unreachable states
        all_states = set(states.keys())
        unreachable = all_states - reachable

        return list(unreachable)

    def _suggest_missing_transitions(self, config: Dict) -> List[str]:
        """Suggest potentially missing transitions"""
        suggestions = []

        states = config.get("states", {})

        # Check for states without outgoing transitions (except final states)
        for state_name, state_config in states.items():
            if state_config.get("type") == "final":
                continue

            transitions = state_config.get("on", {})
            if not transitions:
                suggestions.append(
                    f"State '{state_name}' has no outgoing transitions - consider adding some"
                )

        return suggestions


class DiagramGenerator:
    """Generate visual diagrams from state machine configurations"""

    def __init__(self):
        self.supported_formats = ["mermaid", "graphviz", "plantuml"]

    def generate_mermaid_state_diagram(
        self, config: Dict, title: Optional[str] = None
    ) -> str:
        """Generate Mermaid state diagram from XState config"""

        lines = ["```mermaid", "stateDiagram-v2"]

        if title:
            lines.append(f"    title: {title}")

        lines.append("")

        states = config.get("states", {})
        initial_state = config.get("initial")

        # Add initial state marker
        if initial_state:
            lines.append(f"    [*] --> {initial_state}")

        # Add states and transitions
        for state_name, state_config in states.items():
            # Add state definition
            if state_config.get("type") == "final":
                lines.append(f"    {state_name} --> [*]")

            # Add meta information as state note
            if "meta" in state_config:
                meta = state_config["meta"]
                if "description" in meta:
                    lines.append(f"    {state_name} : {meta['description']}")

            # Add transitions
            transitions = state_config.get("on", {})
            for event, target in transitions.items():
                if isinstance(target, str):
                    lines.append(f"    {state_name} --> {target} : {event}")
                elif isinstance(target, dict) and "target" in target:
                    lines.append(f"    {state_name} --> {target['target']} : {event}")

        lines.append("```")
        return "\n".join(lines)

    def generate_graphviz_diagram(
        self, config: Dict, title: Optional[str] = None
    ) -> str:
        """Generate Graphviz DOT diagram from XState config"""

        contract_id = config.get("id", "state_machine")
        lines = [f'digraph {contract_id.replace("-", "_")} {{']

        if title:
            lines.append(f'    label="{title}";')
            lines.append('    labelloc="t";')

        lines.extend(["    rankdir=LR;", "    node [shape=circle];", ""])

        states = config.get("states", {})
        initial_state = config.get("initial")

        # Add initial state
        if initial_state:
            lines.append("    start [shape=point];")
            lines.append(f"    start -> {initial_state};")

        # Add states
        for state_name, state_config in states.items():
            shape = "doublecircle" if state_config.get("type") == "final" else "circle"

            # Add description from meta
            label = state_name
            if "meta" in state_config and "description" in state_config["meta"]:
                desc = state_config["meta"]["description"][
                    :30
                ]  # Truncate long descriptions
                label = f"{state_name}\\n{desc}"

            lines.append(f'    {state_name} [shape={shape}, label="{label}"];')

        lines.append("")

        # Add transitions
        for state_name, state_config in states.items():
            transitions = state_config.get("on", {})
            for event, target in transitions.items():
                if isinstance(target, str):
                    lines.append(f'    {state_name} -> {target} [label="{event}"];')
                elif isinstance(target, dict) and "target" in target:
                    lines.append(
                        f'    {state_name} -> {target["target"]} [label="{event}"];'
                    )

        lines.append("}")
        return "\n".join(lines)

    def generate_plantuml_diagram(
        self, config: Dict, title: Optional[str] = None
    ) -> str:
        """Generate PlantUML state diagram from XState config"""

        lines = ["@startuml"]

        if title:
            lines.append(f"title {title}")

        lines.append("")

        states = config.get("states", {})
        initial_state = config.get("initial")

        # Add initial state
        if initial_state:
            lines.append(f"[*] --> {initial_state}")

        # Add states and transitions
        for state_name, state_config in states.items():
            # Add final state markers
            if state_config.get("type") == "final":
                lines.append(f"{state_name} --> [*]")

            # Add transitions
            transitions = state_config.get("on", {})
            for event, target in transitions.items():
                if isinstance(target, str):
                    lines.append(f"{state_name} --> {target} : {event}")
                elif isinstance(target, dict) and "target" in target:
                    lines.append(f"{state_name} --> {target['target']} : {event}")

        lines.append("@enduml")
        return "\n".join(lines)

    def render_graphviz_to_image(
        self, dot_content: str, output_path: str, format: str = "png"
    ) -> bool:
        """Render Graphviz DOT to image file (requires Graphviz installed)"""

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".dot", delete=False
            ) as f:
                f.write(dot_content)
                dot_file = f.name

            # Run dot command
            cmd = ["dot", f"-T{format}", dot_file, "-o", output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Clean up
            Path(dot_file).unlink()

            if result.returncode == 0:
                logger.info(f"Generated diagram: {output_path}")
                return True
            else:
                logger.error(f"Graphviz error: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.warning("Graphviz not installed - cannot generate image")
            return False
        except Exception as e:
            logger.error(f"Error generating diagram: {e}")
            return False


class StateMachineAnalyzer:
    """Analyze state machine properties and characteristics"""

    def analyze_state_machine(self, config: Dict) -> Dict:
        """Comprehensive analysis of state machine properties"""

        states = config.get("states", {})
        analysis = {
            "basic_properties": self._analyze_basic_properties(config),
            "state_analysis": self._analyze_states(states),
            "transition_analysis": self._analyze_transitions(states),
            "complexity_metrics": self._calculate_complexity_metrics(config),
            "reachability": self._analyze_reachability(config),
        }

        return analysis

    def _analyze_basic_properties(self, config: Dict) -> Dict:
        """Analyze basic state machine properties"""

        states = config.get("states", {})

        return {
            "total_states": len(states),
            "has_initial_state": "initial" in config,
            "initial_state": config.get("initial"),
            "has_context": "context" in config,
            "machine_id": config.get("id", "unknown"),
        }

    def _analyze_states(self, states: Dict) -> Dict:
        """Analyze state characteristics"""

        final_states = []
        compound_states = []
        simple_states = []
        parallel_states = []

        for state_name, state_config in states.items():
            if state_config.get("type") == "final":
                final_states.append(state_name)
            elif "states" in state_config:  # Compound state
                compound_states.append(state_name)
            elif state_config.get("type") == "parallel":
                parallel_states.append(state_name)
            else:
                simple_states.append(state_name)

        return {
            "final_states": final_states,
            "compound_states": compound_states,
            "simple_states": simple_states,
            "parallel_states": parallel_states,
            "final_state_count": len(final_states),
            "compound_state_count": len(compound_states),
            "simple_state_count": len(simple_states),
            "parallel_state_count": len(parallel_states),
        }

    def _analyze_transitions(self, states: Dict) -> Dict:
        """Analyze transition characteristics"""

        total_transitions = 0
        states_with_no_transitions = []
        transition_events = set()

        for state_name, state_config in states.items():
            transitions = state_config.get("on", {})

            if not transitions:
                states_with_no_transitions.append(state_name)
            else:
                total_transitions += len(transitions)
                transition_events.update(transitions.keys())

        return {
            "total_transitions": total_transitions,
            "unique_events": len(transition_events),
            "event_list": list(transition_events),
            "states_with_no_transitions": states_with_no_transitions,
            "average_transitions_per_state": total_transitions / max(len(states), 1),
        }

    def _calculate_complexity_metrics(self, config: Dict) -> Dict:
        """Calculate complexity metrics"""

        states = config.get("states", {})

        # Cyclomatic complexity approximation
        num_states = len(states)
        num_transitions = sum(
            len(state_config.get("on", {})) for state_config in states.values()
        )

        # Simple complexity score based on states and transitions
        complexity_score = num_states + num_transitions

        # Determine complexity level
        if complexity_score < 5:
            complexity_level = "simple"
        elif complexity_score < 15:
            complexity_level = "moderate"
        elif complexity_score < 30:
            complexity_level = "complex"
        else:
            complexity_level = "very_complex"

        return {
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "state_transition_ratio": num_transitions / max(num_states, 1),
            "branching_factor": self._calculate_branching_factor(states),
        }

    def _calculate_branching_factor(self, states: Dict) -> float:
        """Calculate average branching factor"""

        if not states:
            return 0.0

        total_outgoing = sum(
            len(state_config.get("on", {})) for state_config in states.values()
        )
        return total_outgoing / len(states)

    def _analyze_reachability(self, config: Dict) -> Dict:
        """Analyze state reachability"""

        states = config.get("states", {})
        initial_state = config.get("initial")

        if not initial_state:
            return {"reachable_states": [], "unreachable_states": list(states.keys())}

        # BFS to find all reachable states
        reachable = set()
        queue = [initial_state]

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue

            reachable.add(current)

            # Find transitions from current state
            state_config = states.get(current, {})
            transitions = state_config.get("on", {})

            for event, target in transitions.items():
                if isinstance(target, str):
                    if target not in reachable and target not in queue:
                        queue.append(target)
                elif isinstance(target, dict) and "target" in target:
                    target_state = target["target"]
                    if target_state not in reachable and target_state not in queue:
                        queue.append(target_state)

        unreachable = set(states.keys()) - reachable

        return {
            "reachable_states": list(reachable),
            "unreachable_states": list(unreachable),
            "reachability_percentage": len(reachable) / max(len(states), 1) * 100,
        }


class StatelyAIIntegration:
    """Integration utilities for Stately.ai platform"""

    def __init__(self):
        self.stately_api_version = "2023-12-01"

    def format_for_stately_editor(self, config: Dict) -> Dict:
        """Format XState config for Stately.ai editor"""

        stately_config = config.copy()

        # Add Stately.ai specific metadata
        stately_config["meta"] = stately_config.get("meta", {})
        stately_config["meta"].update(
            {
                "stately": {
                    "version": self.stately_api_version,
                    "layout": "horizontal",
                    "zoom": 1.0,
                    "pan": {"x": 0, "y": 0},
                }
            }
        )

        # Ensure all states have proper meta for visualization
        states = stately_config.get("states", {})
        for state_name, state_config in states.items():
            if "meta" not in state_config:
                state_config["meta"] = {}

            # Add default position if not specified
            if "stately" not in state_config["meta"]:
                state_config["meta"]["stately"] = {"position": {"x": 0, "y": 0}}

        return stately_config

    def generate_stately_url(self, config: Dict) -> str:
        """Generate URL for opening state machine in Stately.ai editor"""

        # Encode the config for URL
        import base64
        import urllib.parse

        config_json = json.dumps(config, separators=(",", ":"))
        encoded_config = base64.b64encode(config_json.encode()).decode()
        encoded_url = urllib.parse.quote(encoded_config)

        return f"https://stately.ai/registry/new?source=xstate&config={encoded_url}"


class WorkflowIntegrationUtils:
    """Utilities for integrating with workflow management systems"""

    def __init__(self):
        self.supported_systems = ["zapier", "n8n", "microsoft_flow", "camunda"]

    def convert_to_zapier_format(self, config: Dict) -> Dict:
        """Convert state machine to Zapier workflow format"""

        states = config.get("states", {})
        zapier_workflow = {
            "title": f"Contract Workflow - {config.get('id', 'Unknown')}",
            "steps": [],
        }

        step_counter = 1
        for state_name, state_config in states.items():
            step = {
                "id": step_counter,
                "title": state_name.replace("_", " ").title(),
                "type": "webhook" if state_config.get("type") != "final" else "filter",
                "meta": state_config.get("meta", {}),
            }

            # Add transitions as conditions
            transitions = state_config.get("on", {})
            if transitions:
                step["conditions"] = []
                for event, target in transitions.items():
                    condition = {
                        "trigger": event,
                        "next_step": (
                            target if isinstance(target, str) else target.get("target")
                        ),
                    }
                    step["conditions"].append(condition)

            zapier_workflow["steps"].append(step)
            step_counter += 1

        return zapier_workflow

    def convert_to_bpmn(self, config: Dict) -> str:
        """Convert state machine to BPMN XML format (simplified)"""

        states = config.get("states", {})
        initial_state = config.get("initial")

        bpmn_elements = []
        bpmn_elements.append('<?xml version="1.0" encoding="UTF-8"?>')
        bpmn_elements.append(
            '<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">'
        )
        bpmn_elements.append(
            f'  <process id="{config.get("id", "contract_process")}" isExecutable="true">'
        )

        # Add start event
        if initial_state:
            bpmn_elements.append('    <startEvent id="startEvent" name="Start">')
            bpmn_elements.append(f"      <outgoing>flow_to_{initial_state}</outgoing>")
            bpmn_elements.append("    </startEvent>")
            bpmn_elements.append(
                f'    <sequenceFlow id="flow_to_{initial_state}" sourceRef="startEvent" targetRef="{initial_state}"/>'
            )

        # Add tasks for each state
        for state_name, state_config in states.items():
            if state_config.get("type") == "final":
                bpmn_elements.append(
                    f'    <endEvent id="{state_name}" name="{state_name.replace("_", " ").title()}"/>'
                )
            else:
                bpmn_elements.append(
                    f'    <userTask id="{state_name}" name="{state_name.replace("_", " ").title()}">'
                )

                # Add outgoing flows for transitions
                transitions = state_config.get("on", {})
                for event, target in transitions.items():
                    target_state = (
                        target if isinstance(target, str) else target.get("target")
                    )
                    bpmn_elements.append(
                        f"      <outgoing>flow_{state_name}_to_{target_state}</outgoing>"
                    )

                bpmn_elements.append("    </userTask>")

                # Add sequence flows
                for event, target in transitions.items():
                    target_state = (
                        target if isinstance(target, str) else target.get("target")
                    )
                    bpmn_elements.append(
                        f'    <sequenceFlow id="flow_{state_name}_to_{target_state}" sourceRef="{state_name}" targetRef="{target_state}"/>'
                    )

        bpmn_elements.append("  </process>")
        bpmn_elements.append("</definitions>")

        return "\n".join(bpmn_elements)


class FileManager:
    """File management utilities for state machine outputs"""

    @staticmethod
    def save_xstate_config(config: Dict, output_path: Union[str, Path]) -> bool:
        """Save XState configuration to JSON file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved XState config: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving XState config: {e}")
            return False

    @staticmethod
    def save_diagram(content: str, output_path: Union[str, Path], format: str) -> bool:
        """Save diagram content to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Saved {format} diagram: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving {format} diagram: {e}")
            return False

    @staticmethod
    def create_workflow_package(
        config: Dict,
        diagrams: Dict[str, str],
        analysis: Dict,
        output_dir: Union[str, Path],
    ) -> bool:
        """Create complete workflow package with all outputs"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save main XState config
            FileManager.save_xstate_config(config, output_dir / "state_machine.json")

            # Save diagrams
            for format_name, content in diagrams.items():
                if format_name == "mermaid":
                    FileManager.save_diagram(
                        content, output_dir / "diagram.mermaid", format_name
                    )
                elif format_name == "graphviz":
                    FileManager.save_diagram(
                        content, output_dir / "diagram.dot", format_name
                    )
                elif format_name == "plantuml":
                    FileManager.save_diagram(
                        content, output_dir / "diagram.puml", format_name
                    )

            # Save analysis
            with open(output_dir / "analysis.json", "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            # Create README
            readme_content = f"""# Contract State Machine

## Overview
This package contains the generated state machine for contract lifecycle management.

## Files
- `state_machine.json` - XState configuration
- `diagram.mermaid` - Mermaid state diagram
- `diagram.dot` - Graphviz diagram
- `analysis.json` - State machine analysis

## Analysis Summary
- Total States: {analysis.get('basic_properties', {}).get('total_states', 0)}
- Total Transitions: {analysis.get('transition_analysis', {}).get('total_transitions', 0)}
- Complexity: {analysis.get('complexity_metrics', {}).get('complexity_level', 'unknown')}

## Usage
Import the `state_machine.json` into XState or Stately.ai for execution.

Generated by Smart Legal Contracts Pipeline
"""

            with open(output_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)

            logger.info(f"Created workflow package: {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Error creating workflow package: {e}")
            return False


# Main utility functions for easy import
def validate_state_machine(config: Dict) -> ValidationResult:
    """Convenience function for state machine validation"""
    validator = XStateValidator()
    return validator.validate_xstate_config(config)


def generate_diagram(
    config: Dict, format: str = "mermaid", title: Optional[str] = None
) -> str:
    """Convenience function for diagram generation"""
    generator = DiagramGenerator()

    if format == "mermaid":
        return generator.generate_mermaid_state_diagram(config, title)
    elif format == "graphviz":
        return generator.generate_graphviz_diagram(config, title)
    elif format == "plantuml":
        return generator.generate_plantuml_diagram(config, title)
    else:
        raise ValueError(f"Unsupported format: {format}")


def analyze_state_machine(config: Dict) -> Dict:
    """Convenience function for state machine analysis"""
    analyzer = StateMachineAnalyzer()
    return analyzer.analyze_state_machine(config)


def create_complete_workflow_package(
    config: Dict, output_dir: Union[str, Path], title: Optional[str] = None
) -> bool:
    """Create complete workflow package with validation, diagrams, and analysis"""

    try:
        # Validate configuration
        validation = validate_state_machine(config)
        if not validation.is_valid:
            logger.error(f"Invalid state machine: {validation.errors}")
            return False

        # Generate diagrams
        diagrams = {
            "mermaid": generate_diagram(config, "mermaid", title),
            "graphviz": generate_diagram(config, "graphviz", title),
            "plantuml": generate_diagram(config, "plantuml", title),
        }

        # Analyze state machine
        analysis = analyze_state_machine(config)

        # Add validation results to analysis
        analysis["validation"] = {
            "is_valid": validation.is_valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "suggestions": validation.suggestions,
        }

        # Create package
        return FileManager.create_workflow_package(
            config, diagrams, analysis, output_dir
        )

    except Exception as e:
        logger.error(f"Error creating workflow package: {e}")
        return False


if __name__ == "__main__":
    # Example usage and testing

    # Test XState config
    test_config = {
        "id": "test-contract",
        "initial": "draft",
        "states": {
            "draft": {"on": {"SUBMIT": "under_review"}},
            "under_review": {"on": {"APPROVE": "signed", "REJECT": "draft"}},
            "signed": {"on": {"ACTIVATE": "active"}},
            "active": {"on": {"TERMINATE": "terminated"}},
            "terminated": {"type": "final"},
        },
    }

    print("Testing Workflow Utils...")

    # Test validation
    validation = validate_state_machine(test_config)
    print(f"Validation: {'✓' if validation.is_valid else '✗'}")

    # Test diagram generation
    mermaid_diagram = generate_diagram(test_config, "mermaid", "Test Contract")
    print(f"Mermaid diagram generated: {len(mermaid_diagram)} characters")

    # Test analysis
    analysis = analyze_state_machine(test_config)
    print(f"Analysis completed: {analysis['basic_properties']['total_states']} states")

    # Test package creation
    success = create_complete_workflow_package(
        test_config, "test_output", "Test Contract"
    )
    print(f"Package creation: {'✓' if success else '✗'}")

    print("Testing complete!")
