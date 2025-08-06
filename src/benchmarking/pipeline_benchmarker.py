"""
Pipeline Benchmarking System
Analyzes performance of the smart legal contract generation pipeline
Updated with dynamic contract limits matching the variable extractor pattern
"""

import json
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import sys

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.variable_extractor import LegalVariableExtractor
from src.generation.template_generator import TemplateGenerator
from src.fsm.intelligent_state_generator import IntelligentStateMachineGenerator
from src.fsm.enhanced_pipeline import EnhancedSmartContractPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessMonitor:
    """Monitor system resources during processing"""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0

    def start(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

    def update(self):
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)

    def stop(self):
        end_time = time.time()
        return {
            "duration_seconds": end_time - self.start_time,
            "start_memory_mb": self.start_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": self.peak_memory - self.start_memory,
        }


class ExtractionBenchmark:
    """Benchmarks variable extraction performance"""

    def analyze_results(self, extraction_results: List[Dict]) -> Dict:
        """Analyze extraction results for academic metrics"""

        if not extraction_results:
            return {"error": "No extraction results provided"}

        metrics = {
            "total_extractions": len(extraction_results),
            "unique_contracts": len(
                set(r.get("contract_id", "") for r in extraction_results)
            ),
            "clause_type_distribution": defaultdict(int),
            "variable_type_performance": defaultdict(
                lambda: {"total": 0, "successful": 0}
            ),
            "confidence_distribution": [],
            "error_patterns": defaultdict(int),
        }

        for result in extraction_results:
            # Clause type distribution
            clause_type = result.get("clause_type", "unknown")
            metrics["clause_type_distribution"][clause_type] += 1

            # Variable performance by type
            extractions = result.get("extractions", {})
            for var_name, value in extractions.items():
                var_type = self._infer_variable_type(var_name, value)
                metrics["variable_type_performance"][var_type]["total"] += 1

                if self._is_successful_extraction(value):
                    metrics["variable_type_performance"][var_type]["successful"] += 1
                else:
                    metrics["error_patterns"][f"{var_type}_null"] += 1

        # Calculate success rates
        for var_type, stats in metrics["variable_type_performance"].items():
            if stats["total"] > 0:
                stats["success_rate"] = stats["successful"] / stats["total"]
            else:
                stats["success_rate"] = 0.0

        return dict(metrics)

    def _infer_variable_type(self, var_name: str, value) -> str:
        """Infer variable type from name and value"""
        if value is None:
            return "unknown"
        elif isinstance(value, bool):
            return "Boolean"
        elif isinstance(value, (int, float)):
            return "Double"
        elif isinstance(value, str):
            return "String"
        elif isinstance(value, list):
            return "Array"
        elif isinstance(value, dict):
            if "$class" in value:
                class_name = value["$class"].split(".")[-1]
                return class_name
            return "Object"
        return "unknown"

    def _is_successful_extraction(self, value) -> bool:
        """Determine if extraction was successful"""
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, list) and len(value) == 0:
            return False
        return True


class TemplateBenchmark:
    """Benchmarks template generation quality"""

    def analyze_generation_stats(self, template_stats: Dict) -> Dict:
        """Analyze template generation statistics"""

        return {
            "generation_metrics": {
                "contracts_processed": template_stats.get("contracts_processed", 0),
                "contracts_successful": template_stats.get("contracts_successful", 0),
                "templates_generated": template_stats.get("templates_generated", 0),
                "success_rate": self._calculate_success_rate(template_stats),
                "original_text_preserved": template_stats.get(
                    "original_text_preserved", 0
                ),
                "fallback_templates": template_stats.get(
                    "fallback_templates_generated", 0
                ),
                "double_replacements": template_stats.get("double_replacements", 0),
            },
            "quality_metrics": {
                "preservation_rate": self._calculate_preservation_rate(template_stats),
                "replacement_accuracy": self._calculate_replacement_accuracy(
                    template_stats
                ),
            },
        }

    def _calculate_success_rate(self, stats: Dict) -> float:
        """Calculate overall success rate"""
        processed = stats.get("contracts_processed", 0)
        successful = stats.get("contracts_successful", 0)
        return successful / processed if processed > 0 else 0.0

    def _calculate_preservation_rate(self, stats: Dict) -> float:
        """Calculate original text preservation rate"""
        total = stats.get("templates_generated", 0)
        preserved = stats.get("original_text_preserved", 0)
        return preserved / total if total > 0 else 0.0

    def _calculate_replacement_accuracy(self, stats: Dict) -> float:
        """Calculate variable replacement accuracy"""
        total = stats.get("templates_generated", 0)
        replacements = stats.get("double_replacements", 0)
        return replacements / total if total > 0 else 0.0


class StateMachineBenchmark:
    """Benchmarks state machine generation"""

    def analyze_fsm_results(self, sm_results: Dict) -> Dict:
        """Analyze state machine generation results"""

        if not sm_results:
            return {"error": "No state machine results provided"}

        metrics = {
            "generation_summary": {
                "total_contracts": len(sm_results),
                "total_states": 0,
                "total_transitions": 0,
                "avg_states_per_contract": 0,
                "avg_transitions_per_contract": 0,
            },
            "validation_metrics": {
                "fsm_validation_passed": 0,
                "fsm_validation_warnings": 0,
                "average_confidence": 0,
                "high_confidence_states": 0,
            },
            "parsing_performance": {
                "json_parse_success": 0,
                "fallback_parsing_used": 0,
                "manual_extraction_used": 0,
            },
        }

        total_confidence = 0
        confidence_count = 0

        for contract_id, data in sm_results.items():
            # Basic counts
            states = data.get("states", [])
            transitions = data.get("transitions", [])

            metrics["generation_summary"]["total_states"] += len(states)
            metrics["generation_summary"]["total_transitions"] += len(transitions)

            # Confidence analysis
            for state in states:
                if isinstance(state, dict) and "confidence" in state:
                    confidence = state["confidence"]
                    total_confidence += confidence
                    confidence_count += 1

                    if confidence > 0.8:
                        metrics["validation_metrics"]["high_confidence_states"] += 1

            # FSM validation analysis
            intelligence_metrics = data.get("intelligence_metrics", {})
            fsm_validation = intelligence_metrics.get("fsm_validation_summary", {})

            if fsm_validation.get("is_valid_fsm", False):
                metrics["validation_metrics"]["fsm_validation_passed"] += 1

            if fsm_validation.get("validation_warnings_count", 0) > 0:
                metrics["validation_metrics"]["fsm_validation_warnings"] += 1

        # Calculate averages
        if len(sm_results) > 0:
            metrics["generation_summary"]["avg_states_per_contract"] = metrics[
                "generation_summary"
            ]["total_states"] / len(sm_results)
            metrics["generation_summary"]["avg_transitions_per_contract"] = metrics[
                "generation_summary"
            ]["total_transitions"] / len(sm_results)

        if confidence_count > 0:
            metrics["validation_metrics"]["average_confidence"] = (
                total_confidence / confidence_count
            )

        return metrics


class AcademicDataExporter:
    """Exports benchmarking data in academic formats"""

    def __init__(self, output_dir: str = "outputs/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_comprehensive_results(
        self, benchmark_data: Dict, timestamp: str = None
    ) -> Dict[str, Path]:
        """Export all benchmark results for academic analysis"""

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exported_files = {}

        # Export JSON summary
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        exported_files["summary"] = summary_file

        # Export CSV data for statistical analysis
        csv_file = self.output_dir / f"benchmark_metrics_{timestamp}.csv"
        self._export_csv_metrics(benchmark_data, csv_file)
        exported_files["csv_metrics"] = csv_file

        # Export academic summary
        academic_file = self.output_dir / f"academic_summary_{timestamp}.json"
        academic_summary = self._create_academic_summary(benchmark_data)
        with open(academic_file, "w", encoding="utf-8") as f:
            json.dump(academic_summary, f, indent=2, ensure_ascii=False)
        exported_files["academic_summary"] = academic_file

        return exported_files

    def _export_csv_metrics(self, data: Dict, csv_file: Path):
        """Export key metrics to CSV for statistical analysis"""
        import csv

        metrics_data = []

        # Extract key metrics for CSV
        if "extraction_benchmark" in data:
            ext_data = data["extraction_benchmark"]
            for var_type, stats in ext_data.get(
                "variable_type_performance", {}
            ).items():
                metrics_data.append(
                    {
                        "component": "extraction",
                        "metric_type": "success_rate",
                        "category": var_type,
                        "value": stats.get("success_rate", 0),
                        "count": stats.get("total", 0),
                    }
                )

        if "template_benchmark" in data:
            temp_data = data["template_benchmark"]
            gen_metrics = temp_data.get("generation_metrics", {})
            quality_metrics = temp_data.get("quality_metrics", {})

            for metric_name, value in gen_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_data.append(
                        {
                            "component": "template",
                            "metric_type": "generation",
                            "category": metric_name,
                            "value": value,
                            "count": 1,
                        }
                    )

            for metric_name, value in quality_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_data.append(
                        {
                            "component": "template",
                            "metric_type": "quality",
                            "category": metric_name,
                            "value": value,
                            "count": 1,
                        }
                    )

        if "state_machine_benchmark" in data:
            sm_data = data["state_machine_benchmark"]
            for category, metrics in sm_data.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            metrics_data.append(
                                {
                                    "component": "state_machine",
                                    "metric_type": category,
                                    "category": metric_name,
                                    "value": value,
                                    "count": 1,
                                }
                            )

        # Write CSV
        if metrics_data:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "component",
                        "metric_type",
                        "category",
                        "value",
                        "count",
                    ],
                )
                writer.writeheader()
                writer.writerows(metrics_data)

    def _create_academic_summary(self, data: Dict) -> Dict:
        """Create academic summary with key findings"""

        summary = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "pipeline_components": list(data.keys()),
                "total_processing_time": data.get("performance_metrics", {}).get(
                    "duration_seconds", 0
                ),
            },
            "key_findings": {},
            "statistical_summary": {},
        }

        # Extract key academic findings
        if "extraction_benchmark" in data:
            ext_data = data["extraction_benchmark"]
            summary["key_findings"]["extraction"] = {
                "total_extractions": ext_data.get("total_extractions", 0),
                "unique_contracts": ext_data.get("unique_contracts", 0),
                "clause_types_processed": len(
                    ext_data.get("clause_type_distribution", {})
                ),
                "overall_success_rate": self._calculate_overall_extraction_success(
                    ext_data
                ),
            }

        if "template_benchmark" in data:
            temp_data = data["template_benchmark"]
            summary["key_findings"]["template"] = {
                "generation_success_rate": temp_data.get("generation_metrics", {}).get(
                    "success_rate", 0
                ),
                "preservation_rate": temp_data.get("quality_metrics", {}).get(
                    "preservation_rate", 0
                ),
                "templates_generated": temp_data.get("generation_metrics", {}).get(
                    "templates_generated", 0
                ),
            }

        if "state_machine_benchmark" in data:
            sm_data = data["state_machine_benchmark"]
            if "error" not in sm_data:  # Only process if we have valid data
                summary["key_findings"]["state_machine"] = {
                    "contracts_processed": sm_data.get("generation_summary", {}).get(
                        "total_contracts", 0
                    ),
                    "fsm_validation_rate": sm_data.get("validation_metrics", {}).get(
                        "fsm_validation_rate", 0
                    ),
                    "average_confidence": sm_data.get("validation_metrics", {}).get(
                        "average_confidence", 0
                    ),
                    "total_states": sm_data.get("generation_summary", {}).get(
                        "total_states", 0
                    ),
                    "total_transitions": sm_data.get("generation_summary", {}).get(
                        "total_transitions", 0
                    ),
                }
            else:
                summary["key_findings"]["state_machine"] = {
                    "contracts_processed": 0,
                    "fsm_validation_rate": 0,
                    "average_confidence": 0,
                    "error": sm_data.get("error", "Unknown error"),
                }

        return summary

    def _calculate_overall_extraction_success(self, ext_data: Dict) -> float:
        """Calculate overall extraction success rate"""
        var_performance = ext_data.get("variable_type_performance", {})
        if not var_performance:
            return 0.0

        total_vars = sum(stats.get("total", 0) for stats in var_performance.values())
        successful_vars = sum(
            stats.get("successful", 0) for stats in var_performance.values()
        )

        return successful_vars / total_vars if total_vars > 0 else 0.0


class ComprehensivePipelineBenchmark:
    """Main benchmarking orchestrator for the entire pipeline with dynamic contract limits"""

    def __init__(self, mode: str = "default", output_dir: str = "outputs/benchmarks"):
        self.mode = mode.lower()

        # Dynamic contract limits matching variable extractor pattern
        self.contract_limits = {"default": 15, "production": 100, "test": 5, "debug": 1}

        # Validate mode
        if self.mode not in self.contract_limits:
            logger.warning(f"Unknown mode '{mode}', defaulting to 'default'")
            self.mode = "default"

        self.max_contracts = self.contract_limits[self.mode]

        self.monitor = ProcessMonitor()
        self.extraction_benchmark = ExtractionBenchmark()
        self.template_benchmark = TemplateBenchmark()
        self.state_machine_benchmark = StateMachineBenchmark()
        self.exporter = AcademicDataExporter(output_dir)

        logger.info(
            f"Initialized Pipeline Benchmarker (mode: {self.mode.upper()}, max contracts: {self.max_contracts})"
        )

    def run_full_benchmark(
        self, max_contracts: Optional[int] = None, mode: Optional[str] = None
    ) -> Dict[str, Path]:
        """Run comprehensive benchmark of the entire pipeline with dynamic contract limits"""

        # Override mode if provided
        if mode and mode.lower() != self.mode:
            self.mode = mode.lower()
            if self.mode in self.contract_limits:
                self.max_contracts = self.contract_limits[self.mode]
                logger.info(
                    f"Updated mode to {self.mode.upper()} (max contracts: {self.max_contracts})"
                )
            else:
                logger.warning(
                    f"Unknown mode '{mode}', keeping current mode: {self.mode}"
                )

        # Use provided max_contracts or fall back to mode default
        contracts_to_process = (
            max_contracts if max_contracts is not None else self.max_contracts
        )

        logger.info(
            f"Starting comprehensive benchmark (mode: {self.mode.upper()}, contracts: {contracts_to_process})"
        )
        self.monitor.start()

        try:
            # Initialize pipeline with same mode
            pipeline = EnhancedSmartContractPipeline(mode=self.mode)

            # Run pipeline with monitoring
            logger.info("Running pipeline...")
            pipeline_results = pipeline.run_complete_pipeline(
                max_contracts=contracts_to_process,
                force_regenerate=True,
                generate_templates=True,
                generate_state_machines=True,
            )

            self.monitor.update()

            # Debug: Log the structure of pipeline_results
            logger.info(f"Pipeline results structure: {list(pipeline_results.keys())}")
            for key, value in pipeline_results.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}: {list(value.keys())}")

            # Collect benchmark data
            benchmark_data = self._collect_comprehensive_metrics(
                pipeline, pipeline_results
            )
            benchmark_data["performance_metrics"] = self.monitor.stop()

            # Export results with mode-specific naming
            logger.info("Exporting benchmark results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            benchmark_data["benchmark_metadata"] = {
                "mode": self.mode,
                "max_contracts": contracts_to_process,
                "mode_max_contracts": self.max_contracts,  # Original mode limit
                "timestamp": timestamp,
                "duration_seconds": benchmark_data["performance_metrics"][
                    "duration_seconds"
                ],
                "contract_limits": self.contract_limits,  # Include all limits for reference
            }

            exported_files = self.exporter.export_comprehensive_results(
                benchmark_data,
                timestamp=f"{self.mode}_{contracts_to_process}contracts_{timestamp}",
            )

            logger.info("Benchmark complete!")
            for file_type, file_path in exported_files.items():
                logger.info(f"  {file_type}: {file_path}")

            return exported_files

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def _collect_comprehensive_metrics(self, pipeline, pipeline_results: Dict) -> Dict:
        """Collect metrics from all pipeline components"""

        benchmark_data = {
            "pipeline_metadata": {
                "timestamp": datetime.now().isoformat(),
                "mode": self.mode,
                "max_contracts": self.max_contracts,
                "contract_limits": self.contract_limits,
                "components_enabled": {
                    "variable_extraction": True,
                    "template_generation": "template_generation" in pipeline_results,
                    "state_machine_generation": "state_machine_generation"
                    in pipeline_results,
                },
            }
        }

        # Variable extraction benchmark - get the raw data that was used
        # The pipeline stores raw variable_results, we need to access it
        if hasattr(pipeline, "variable_extractor"):
            # Try to get recent extraction results from the pipeline run
            benchmark_data["extraction_benchmark"] = {
                "note": "Variable extraction analysis would need raw extraction data"
            }

        # Template generation benchmark
        if "template_generation" in pipeline_results:
            template_stats = pipeline_results["template_generation"].get(
                "statistics", {}
            )
            benchmark_data["template_benchmark"] = (
                self.template_benchmark.analyze_generation_stats(template_stats)
            )

        # State machine benchmark - work with the processed summary data your pipeline provides
        if "state_machine_generation" in pipeline_results:
            sm_section = pipeline_results["state_machine_generation"]

            # Your pipeline provides summary data, let's work with that
            if "summary" in sm_section:
                summary_data = sm_section["summary"]

                # Create benchmark metrics from your pipeline's summary format
                benchmark_data["state_machine_benchmark"] = {
                    "generation_summary": {
                        "total_contracts": summary_data.get("total_contracts", 0),
                        "total_states": summary_data.get("total_states", 0),
                        "total_transitions": summary_data.get("total_transitions", 0),
                        "avg_states_per_contract": summary_data.get(
                            "average_states_per_contract", 0
                        ),
                        "avg_transitions_per_contract": summary_data.get(
                            "average_transitions_per_contract", 0
                        ),
                    },
                    "validation_metrics": {
                        "average_confidence": summary_data.get("average_confidence", 0),
                        "fsm_validation_passed": summary_data.get(
                            "fsm_validation_summary", {}
                        ).get("contracts_with_valid_fsm", 0),
                        "fsm_validation_rate": summary_data.get(
                            "fsm_validation_summary", {}
                        ).get("fsm_validation_rate", 0),
                        "high_confidence_states": pipeline_results.get(
                            "statistics", {}
                        ).get("high_confidence_states", 0),
                    },
                    "parsing_performance": {
                        "contracts_processed": sm_section.get(
                            "contracts_with_state_machines", 0
                        )
                    },
                }
            else:
                logger.warning(
                    "State machine summary data not found in expected format"
                )
                benchmark_data["state_machine_benchmark"] = {
                    "error": "State machine summary not found"
                }

        return benchmark_data


def main():
    """CLI interface for benchmarking with dynamic contract limits"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark smart legal contract pipeline with dynamic contract limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dynamic Contract Limits by Mode:
  • test: 5 contracts (for quick testing)
  • default: 15 contracts (balanced testing)
  • production: 100 contracts (full analysis)
  • debug: 1 contract (minimal debugging)

Examples:
  # Use mode defaults
  python pipeline_benchmarker.py --mode test
  python pipeline_benchmarker.py --mode production
  
  # Override contract limits
  python pipeline_benchmarker.py --mode test --contracts 10
  python pipeline_benchmarker.py --mode production --contracts 25
        """,
    )

    parser.add_argument(
        "--contracts",
        "-c",
        type=int,
        help="Override max contracts to process (overrides mode default)",
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["test", "default", "production", "debug"],
        default="default",
        help="Processing mode with dynamic contract limits (default: default)",
    )

    parser.add_argument(
        "--output", "-o", default="outputs/benchmarks", help="Output directory"
    )

    args = parser.parse_args()

    # Initialize benchmarker with mode
    benchmarker = ComprehensivePipelineBenchmark(mode=args.mode, output_dir=args.output)

    # Determine actual contract limit
    if args.contracts:
        actual_contracts = args.contracts
        logger.info(
            f"Using override: {actual_contracts} contracts (mode {args.mode} default: {benchmarker.max_contracts})"
        )
    else:
        actual_contracts = benchmarker.max_contracts
        logger.info(f"Using mode default: {actual_contracts} contracts")

    # Run benchmark
    results = benchmarker.run_full_benchmark(max_contracts=args.contracts)

    print(f"\nBenchmark complete! Results saved to:")
    for file_type, path in results.items():
        print(f"  {file_type}: {path}")

    # Print quick summary with contract limits info
    print(f"\nQuick Summary:")
    print(f"  Mode: {args.mode} (limit: {benchmarker.max_contracts})")
    print(f"  Contracts processed: {actual_contracts}")
    print(f"  Override applied: {'Yes' if args.contracts else 'No'}")
    print(f"  Output: {args.output}")
    print(f"\nContract Limits by Mode:")
    for mode, limit in benchmarker.contract_limits.items():
        marker = " ← current" if mode == args.mode else ""
        print(f"  {mode}: {limit}{marker}")


if __name__ == "__main__":
    main()
