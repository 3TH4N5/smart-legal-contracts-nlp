"""
Enhanced Pipeline - Integrates Variable Extraction, Template Generation, and State Machine Generation
Orchestrates the complete end-to-end pipeline: Contract Text → Variables → Templates → State Machines
Enhanced with improved metrics tracking and validation reporting
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import sys

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import your existing modules with better error handling
try:
    from src.extraction.variable_extractor import LegalVariableExtractor
except ImportError as e:
    print(f"Could not import variable_extractor: {e}")
    print("Make sure variable_extractor.py is in the project root or src/ directory")
    sys.exit(1)

try:
    from src.generation.template_generator import TemplateGenerator
except ImportError as e:
    print(f"Could not import template_generator: {e}")
    print("Make sure template_generator.py is in the project root or src/ directory")
    sys.exit(1)

try:
    from src.fsm.intelligent_state_generator import IntelligentStateMachineGenerator
except ImportError:
    try:
        from intelligent_state_generator import IntelligentStateMachineGenerator
    except ImportError as e:
        print(f"Could not import intelligent_state_generator: {e}")
        sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedSmartContractPipeline:
    """
    Complete pipeline that transforms contract text into:
    1. Extracted variables
    2. Accord Project templates
    3. Contract lifecycle state machines with FSM validation
    """

    def __init__(
        self,
        mode: str = "default",
        data_dir: str = "data",
        config_dir: str = "config",
        output_dir: str = "outputs",
    ):
        self.mode = mode
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)

        # Initialize components
        self.variable_extractor = LegalVariableExtractor(
            data_dir=str(self.data_dir), mode=mode
        )

        self.template_generator = TemplateGenerator(
            mode=mode,
            config_dir=str(self.config_dir),
            output_dir=str(self.output_dir / "generated_templates"),
        )

        self.state_machine_generator = IntelligentStateMachineGenerator(
            config_path=str(self.config_dir / "state_machine_config.json")
        )

        # Enhanced pipeline statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "contracts_processed": 0,
            "variables_extracted": 0,
            "templates_generated": 0,
            "state_machines_generated": 0,
            "total_states_extracted": 0,
            "total_transitions_identified": 0,
            "fsm_validation_passed": 0,
            "fsm_validation_warnings": 0,
            "high_confidence_states": 0,
            "state_categories": {
                "formation": 0,
                "operational": 0,
                "modification": 0,
                "termination": 0,
            },
        }

        logger.info("Enhanced Pipeline initialized with FSM validation capabilities")
        logger.info(
            "Components: Variable Extraction → Template Generation → State Machine Generation with Validation"
        )

    def run_complete_pipeline(
        self,
        max_contracts: Optional[int] = None,
        force_regenerate: bool = False,
        generate_templates: bool = True,
        generate_state_machines: bool = True,
    ) -> Dict:
        """Run the complete end-to-end pipeline with enhanced metrics"""

        self.stats["start_time"] = datetime.now()
        logger.info("Starting Enhanced Smart Contract Pipeline with FSM Validation")
        logger.info(f"Mode: {self.mode}, Max contracts: {max_contracts or 'unlimited'}")
        logger.info(
            f"Components: Templates={generate_templates}, StateMachines={generate_state_machines}"
        )

        try:
            # Step 1: Variable Extraction
            logger.info("STEP 1: Variable Extraction")

            variable_results = self.variable_extractor.extract_all_contracts(
                max_contracts
            )
            self.stats["contracts_processed"] = len(
                set(r["contract_id"] for r in variable_results)
            )
            self.stats["variables_extracted"] = len(variable_results)

            if not variable_results:
                raise ValueError("No variables extracted - cannot proceed")

            logger.info(
                f"Extracted {len(variable_results)} variable sets from {self.stats['contracts_processed']} contracts"
            )

            # Save variable extraction results
            variable_output_file = self.variable_extractor.save_results(
                variable_results,
                f"pipeline_variables_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            # Step 2: Template Generation (Optional)
            template_summary = {}
            if generate_templates:
                logger.info("STEP 2: Template Generation")

                # Convert variable results to format expected by template generator
                template_input_file = variable_output_file

                template_summary = self.template_generator.generate_all_templates(
                    extraction_file=template_input_file,
                    max_contracts=max_contracts,
                    force_regenerate=force_regenerate,
                )

                self.stats["templates_generated"] = template_summary.get(
                    "statistics", {}
                ).get("templates_generated", 0)
                logger.info(
                    f"Generated {self.stats['templates_generated']} Accord Project templates"
                )
            else:
                logger.info("Skipping template generation (disabled)")

            # Step 3: Enhanced State Machine Generation (Optional)
            state_machine_results = {}
            if generate_state_machines:
                logger.info(
                    "STEP 3: Enhanced State Machine Generation with FSM Validation"
                )

                state_machine_results = (
                    self.state_machine_generator.process_contract_extractions(
                        variable_results
                    )
                )

                # Save state machines
                self.state_machine_generator.save_state_machines(
                    state_machine_results, str(self.output_dir / "state_machines")
                )

                # Enhanced statistics collection
                self._collect_enhanced_state_machine_stats(state_machine_results)

                logger.info(
                    f"Generated {self.stats['state_machines_generated']} state machines with FSM validation"
                )
                logger.info(
                    f"Extracted {self.stats['total_states_extracted']} states, {self.stats['total_transitions_identified']} transitions"
                )
                logger.info(
                    f"FSM Validation: {self.stats['fsm_validation_passed']} passed, {self.stats['fsm_validation_warnings']} with warnings"
                )
            else:
                logger.info("Skipping state machine generation (disabled)")

            self.stats["end_time"] = datetime.now()

            # Generate comprehensive pipeline report
            pipeline_summary = self._generate_pipeline_summary(
                variable_results, template_summary, state_machine_results
            )

            # Save pipeline report
            self._save_pipeline_report(pipeline_summary)

            logger.info("Pipeline completed successfully")
            self._log_final_statistics(pipeline_summary)

            return pipeline_summary

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _collect_enhanced_state_machine_stats(self, state_machine_results: Dict):
        """Collect enhanced statistics from state machine generation"""

        self.stats["state_machines_generated"] = len(state_machine_results)

        for contract_id, data in state_machine_results.items():
            # Basic counts
            self.stats["total_states_extracted"] += len(data["states"])
            self.stats["total_transitions_identified"] += len(data["transitions"])

            # Intelligence metrics
            intelligence_metrics = data.get("intelligence_metrics", {})
            self.stats["high_confidence_states"] += intelligence_metrics.get(
                "high_confidence_states", 0
            )

            # State categories
            categories = intelligence_metrics.get("state_categories", {})
            for category, count in categories.items():
                if category in self.stats["state_categories"]:
                    self.stats["state_categories"][category] += count

            # FSM validation metrics
            fsm_validation = intelligence_metrics.get("fsm_validation_summary", {})
            if fsm_validation.get("is_valid_fsm", False):
                self.stats["fsm_validation_passed"] += 1
            if fsm_validation.get("validation_warnings_count", 0) > 0:
                self.stats["fsm_validation_warnings"] += 1

    def _generate_pipeline_summary(
        self,
        variable_results: List[Dict],
        template_summary: Dict,
        state_machine_results: Dict,
    ) -> Dict:
        """Generate comprehensive pipeline summary with enhanced metrics"""

        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        # Convert datetime objects to ISO strings for JSON serialization
        start_time_str = (
            self.stats["start_time"].isoformat() if self.stats["start_time"] else None
        )
        end_time_str = (
            self.stats["end_time"].isoformat() if self.stats["end_time"] else None
        )

        # Analyze variable extraction results
        variable_analysis = self._analyze_variable_results(variable_results)

        # Analyze state machine results with enhanced metrics
        state_machine_analysis = self._analyze_state_machine_results(
            state_machine_results
        )

        return {
            "pipeline_metadata": {
                "mode": self.mode,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "components_enabled": {
                    "variable_extraction": True,
                    "template_generation": bool(template_summary),
                    "state_machine_generation": bool(state_machine_results),
                    "fsm_validation": bool(state_machine_results),
                },
            },
            "statistics": {
                **self.stats,
                "start_time": start_time_str,  # JSON-serializable
                "end_time": end_time_str,  # JSON-serializable
                "duration_seconds": duration,
            },
            "variable_extraction": {
                "summary": variable_analysis,
                "success_rate": variable_analysis["extraction_success_rate"],
            },
            "template_generation": template_summary,
            "state_machine_generation": {
                "summary": state_machine_analysis,
                "contracts_with_state_machines": len(state_machine_results),
                "average_states_per_contract": state_machine_analysis[
                    "average_states_per_contract"
                ],
                "average_transitions_per_contract": state_machine_analysis[
                    "average_transitions_per_contract"
                ],
                "fsm_validation_metrics": {
                    "validation_pass_rate": (
                        self.stats["fsm_validation_passed"]
                        / max(len(state_machine_results), 1)
                    ),
                    "warning_rate": (
                        self.stats["fsm_validation_warnings"]
                        / max(len(state_machine_results), 1)
                    ),
                    "high_confidence_state_rate": (
                        self.stats["high_confidence_states"]
                        / max(self.stats["total_states_extracted"], 1)
                    ),
                },
            },
            "integration_metrics": {
                "end_to_end_success_rate": self._calculate_end_to_end_success_rate(
                    variable_results, template_summary, state_machine_results
                ),
                "contracts_with_complete_processing": self._count_complete_processing(
                    variable_results, template_summary, state_machine_results
                ),
                "most_common_clause_types": self._get_most_common_clause_types(
                    variable_results
                ),
                "most_common_states": self._get_most_common_states(
                    state_machine_results
                ),
                "state_category_distribution": self.stats["state_categories"],
            },
            "output_locations": {
                "variables": str(self.output_dir / "extracted_variables"),
                "templates": str(self.output_dir / "generated_templates"),
                "state_machines": str(self.output_dir / "state_machines"),
                "reports": str(self.output_dir / "reports"),
                "validation_stats": str(
                    self.output_dir
                    / "state_machines"
                    / "fsm_validation_statistics.json"
                ),
            },
        }

    def _analyze_variable_results(self, variable_results: List[Dict]) -> Dict:
        """Analyze variable extraction results"""

        if not variable_results:
            return {"extraction_success_rate": 0.0}

        total_variables = 0
        successful_variables = 0
        clause_type_counts = {}

        for result in variable_results:
            clause_type = result.get("clause_type", "unknown")
            clause_type_counts[clause_type] = clause_type_counts.get(clause_type, 0) + 1

            extractions = result.get("extractions", {})
            for var_name, value in extractions.items():
                total_variables += 1
                if value is not None:
                    successful_variables += 1

        return {
            "total_extractions": len(variable_results),
            "total_variables": total_variables,
            "successful_variables": successful_variables,
            "extraction_success_rate": successful_variables / max(total_variables, 1),
            "clause_type_distribution": clause_type_counts,
            "unique_clause_types": len(clause_type_counts),
            "unique_contracts": len(set(r["contract_id"] for r in variable_results)),
        }

    def _analyze_state_machine_results(self, state_machine_results: Dict) -> Dict:
        """Analyze state machine generation results with enhanced metrics"""

        if not state_machine_results:
            return {
                "average_states_per_contract": 0.0,
                "average_transitions_per_contract": 0.0,
                "average_confidence": 0.0,
                "fsm_validation_summary": {},
            }

        total_states = sum(
            len(data["states"]) for data in state_machine_results.values()
        )
        total_transitions = sum(
            len(data["transitions"]) for data in state_machine_results.values()
        )
        num_contracts = len(state_machine_results)

        # Enhanced metrics collection
        total_confidence = 0
        state_count = 0
        fsm_validation_passed = 0
        reachability_scores = []

        state_distribution = {}
        for contract_id, data in state_machine_results.items():
            # State confidence aggregation
            for state in data["states"]:
                total_confidence += state.get("confidence", 0)
                state_count += 1
                state_name = state["name"]
                state_distribution[state_name] = (
                    state_distribution.get(state_name, 0) + 1
                )

            # FSM validation metrics
            intelligence_metrics = data.get("intelligence_metrics", {})
            fsm_validation = intelligence_metrics.get("fsm_validation_summary", {})

            if fsm_validation.get("is_valid_fsm", False):
                fsm_validation_passed += 1

            reachability_pct = fsm_validation.get("reachability_percentage", 0)
            reachability_scores.append(reachability_pct)

        average_confidence = total_confidence / max(state_count, 1)
        average_reachability = sum(reachability_scores) / max(
            len(reachability_scores), 1
        )

        return {
            "total_contracts": num_contracts,
            "total_states": total_states,
            "total_transitions": total_transitions,
            "average_states_per_contract": total_states / max(num_contracts, 1),
            "average_transitions_per_contract": total_transitions
            / max(num_contracts, 1),
            "average_confidence": average_confidence,
            "state_distribution": state_distribution,
            "unique_state_types": len(state_distribution),
            "fsm_validation_summary": {
                "contracts_with_valid_fsm": fsm_validation_passed,
                "fsm_validation_rate": fsm_validation_passed / max(num_contracts, 1),
                "average_reachability_percentage": average_reachability,
            },
        }

    def _calculate_end_to_end_success_rate(
        self,
        variable_results: List[Dict],
        template_summary: Dict,
        state_machine_results: Dict,
    ) -> float:
        """Calculate end-to-end processing success rate"""

        if not variable_results:
            return 0.0

        unique_contracts = set(r["contract_id"] for r in variable_results)

        # Count contracts that completed all enabled steps
        successful_contracts = 0

        for contract_id in unique_contracts:
            has_variables = any(
                r["contract_id"] == contract_id for r in variable_results
            )
            has_templates = (
                not template_summary
                or template_summary.get("statistics", {}).get("contracts_successful", 0)
                > 0
            )
            has_state_machines = (
                not state_machine_results or contract_id in state_machine_results
            )

            if has_variables and has_templates and has_state_machines:
                successful_contracts += 1

        return successful_contracts / len(unique_contracts)

    def _count_complete_processing(
        self,
        variable_results: List[Dict],
        template_summary: Dict,
        state_machine_results: Dict,
    ) -> int:
        """Count contracts with complete processing"""

        unique_contracts = set(r["contract_id"] for r in variable_results)
        complete_count = 0

        for contract_id in unique_contracts:
            has_variables = any(
                r["contract_id"] == contract_id for r in variable_results
            )
            has_state_machines = contract_id in state_machine_results

            if has_variables and has_state_machines:
                complete_count += 1

        return complete_count

    def _get_most_common_clause_types(self, variable_results: List[Dict]) -> List[Dict]:
        """Get most common clause types"""

        clause_counts = {}
        for result in variable_results:
            clause_type = result.get("clause_type", "unknown")
            clause_counts[clause_type] = clause_counts.get(clause_type, 0) + 1

        return [
            {"clause_type": clause_type, "count": count}
            for clause_type, count in sorted(
                clause_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

    def _get_most_common_states(self, state_machine_results: Dict) -> List[Dict]:
        """Get most common contract states"""

        state_counts = {}
        for data in state_machine_results.values():
            for state in data["states"]:
                state_name = state["name"]
                state_counts[state_name] = state_counts.get(state_name, 0) + 1

        return [
            {"state": state, "count": count}
            for state, count in sorted(
                state_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

    def _save_pipeline_report(self, summary: Dict):
        """Save comprehensive pipeline report"""

        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = (
            reports_dir / f"enhanced_pipeline_report_{self.mode}_{timestamp}.json"
        )

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Pipeline report saved: {report_file}")

    def _log_final_statistics(self, summary: Dict):
        """Log final pipeline statistics with enhanced metrics"""

        stats = summary["statistics"]

        # Fix: Use the actual datetime objects, not the string versions
        start_time = self.stats["start_time"]  # This is the actual datetime object
        end_time = self.stats["end_time"]  # This is the actual datetime object

        if start_time and end_time:
            duration = end_time - start_time
            duration_minutes = duration.total_seconds() / 60
        else:
            duration_minutes = 0

        logger.info("Enhanced Pipeline Statistics:")
        logger.info(f"Duration: {duration_minutes:.1f} minutes")
        logger.info(f"Contracts: {stats['contracts_processed']}")
        logger.info(f"Variables: {stats['variables_extracted']}")
        logger.info(f"Templates: {stats['templates_generated']}")
        logger.info(f"State Machines: {stats['state_machines_generated']}")
        logger.info(f"States: {stats['total_states_extracted']}")
        logger.info(f"Transitions: {stats['total_transitions_identified']}")
        logger.info(f"High Confidence States: {stats['high_confidence_states']}")

        # Enhanced FSM validation statistics
        logger.info("FSM Validation Results:")
        logger.info(f"  Valid FSMs: {stats['fsm_validation_passed']}")
        logger.info(f"  FSMs with Warnings: {stats['fsm_validation_warnings']}")

        # State category distribution
        logger.info("State Category Distribution:")
        for category, count in stats["state_categories"].items():
            logger.info(f"  {category.title()}: {count}")

        integration = summary["integration_metrics"]
        logger.info(f"Success Rate: {integration['end_to_end_success_rate']:.1%}")
        logger.info(
            f"Complete Processing: {integration['contracts_with_complete_processing']} contracts"
        )

        # FSM validation metrics from state machine analysis
        fsm_metrics = summary["state_machine_generation"].get(
            "fsm_validation_metrics", {}
        )
        logger.info("FSM Quality Metrics:")
        logger.info(
            f"  Validation Pass Rate: {fsm_metrics.get('validation_pass_rate', 0):.1%}"
        )
        logger.info(f"  Warning Rate: {fsm_metrics.get('warning_rate', 0):.1%}")
        logger.info(
            f"  High Confidence State Rate: {fsm_metrics.get('high_confidence_state_rate', 0):.1%}"
        )

        logger.info("Output Locations:")
        for location_type, path in summary["output_locations"].items():
            logger.info(f"  {location_type}: {path}")


def main():
    """Command-line interface for enhanced pipeline"""

    # Set up logger for main function
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Enhanced Smart Contract Pipeline - Variables → Templates → State Machines with FSM Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline (all components)
  python -m src.fsm.enhanced_pipeline --mode default --max-contracts 10
  
  # Variables + State Machines only (skip templates)
  python -m src.fsm.enhanced_pipeline --mode production --no-templates
  
  # Variables + Templates only (skip state machines)
  python -m src.fsm.enhanced_pipeline --mode test --no-state-machines
  
  # Variables only
  python -m src.fsm.enhanced_pipeline --variables-only --max-contracts 5
        """,
    )

    # Processing options
    parser.add_argument(
        "--mode",
        "-m",
        choices=["default", "production", "test", "debug"],
        default="default",
        help="Processing mode (default: default)",
    )

    parser.add_argument(
        "--max-contracts", "-c", type=int, help="Maximum number of contracts to process"
    )

    # Component control
    parser.add_argument(
        "--no-templates", action="store_true", help="Skip template generation"
    )

    parser.add_argument(
        "--no-state-machines", action="store_true", help="Skip state machine generation"
    )

    parser.add_argument(
        "--variables-only",
        action="store_true",
        help="Only extract variables (skip templates and state machines)",
    )

    # Other options
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regeneration of existing outputs",
    )

    parser.add_argument(
        "--data-dir", "-d", default="data", help="Data directory path (default: data)"
    )

    parser.add_argument(
        "--config-dir",
        default="config",
        help="Configuration directory path (default: config)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs",
        help="Output directory path (default: outputs)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine what to generate
    if args.variables_only:
        generate_templates = False
        generate_state_machines = False
    else:
        generate_templates = not args.no_templates
        generate_state_machines = not args.no_state_machines

    print("Enhanced Smart Contract Pipeline with FSM Validation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Max contracts: {args.max_contracts or 'unlimited'}")
    print(f"Generate templates: {generate_templates}")
    print(f"Generate state machines: {generate_state_machines}")
    print(f"FSM validation: {generate_state_machines}")
    print(f"Force regeneration: {args.force}")
    print("=" * 60)

    try:
        # Initialize and run pipeline
        pipeline = EnhancedSmartContractPipeline(
            mode=args.mode,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
        )

        results = pipeline.run_complete_pipeline(
            max_contracts=args.max_contracts,
            force_regenerate=args.force,
            generate_templates=generate_templates,
            generate_state_machines=generate_state_machines,
        )

        print("\nPipeline completed successfully")
        print(f"Processed {results['statistics']['contracts_processed']} contracts")
        print(
            f"Generated {results['statistics']['state_machines_generated']} state machines"
        )
        print(
            f"FSM validation passed: {results['statistics']['fsm_validation_passed']}"
        )
        print(f"Results saved to: {args.output_dir}")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
