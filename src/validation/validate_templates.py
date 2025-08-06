"""
Accord Project Template Validation Script
Module: src.validation.validate_templates

Tests generated templates with cicero CLI commands:
- cicero compile (validate model and structure)
- cicero parse (parse sample text)
- cicero draft (generate text from data)

Usage:
    python -m src.validation.validate_templates --template-dir outputs/generated_templates/contract_name/clause_type
    python -m src.validation.validate_templates --scan-all  # Test all templates
"""

import subprocess
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationResult:
    """Result of template validation."""

    template_path: str
    test_name: str
    success: bool
    output: str
    error: str
    execution_time: float

    def to_dict(self) -> Dict:
        return {
            "template_path": self.template_path,
            "test_name": self.test_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
        }


class AccordProjectValidator:
    """Validates Accord Project templates using cicero CLI."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.outputs_dir = project_root / "outputs" / "generated_templates"
        self.results: List[ValidationResult] = []

        # Check if cicero is installed
        self._check_cicero_installation()

    def _check_cicero_installation(self):
        """Check if cicero CLI is installed and accessible."""
        try:
            result = subprocess.run(
                ["cicero", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=True,  # Windows compatibility
            )
            if result.returncode == 0:
                logger.info(f"Cicero CLI found: {result.stdout.strip()}")
            else:
                logger.error("Cicero CLI not found or not working")
                logger.error("Please install: npm install -g @accordproject/cicero-cli")
                sys.exit(1)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Cicero CLI not found: {e}")
            logger.error("Please install: npm install -g @accordproject/cicero-cli")
            sys.exit(1)

    def _run_cicero_command(
        self, command: List[str], cwd: Path, timeout: int = 30
    ) -> Tuple[bool, str, str, float]:
        """Run a cicero command and return results."""
        start_time = datetime.now()

        try:
            logger.debug(f"Running command: {' '.join(command)} in {cwd}")

            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,  # Windows compatibility
                env=dict(
                    os.environ, FORCE_COLOR="0"
                ),  # Disable colors for cleaner output
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            success = result.returncode == 0
            output = result.stdout.strip()
            error = result.stderr.strip()

            return success, output, error, execution_time

        except subprocess.TimeoutExpired:
            execution_time = (datetime.now() - start_time).total_seconds()
            return (
                False,
                "",
                f"Command timed out after {timeout} seconds",
                execution_time,
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return False, "", f"Command failed: {str(e)}", execution_time

    def validate_template_structure(self, template_path: Path) -> ValidationResult:
        """Validate that template has required files."""
        logger.info(f"Validating structure: {template_path.name}")

        required_files = [
            "text/grammar.tem.md",
            "model/model.cto",
            "request.json",
            "state.json",
            "package.json",
        ]

        optional_files = [
            "logic/logic.ts",
            "logic/logic.js",
            "logic/logic.ergo",
            "sample.md",
            "README.md",
        ]

        missing_files = []
        found_files = []

        for file_path in required_files:
            full_path = template_path / file_path
            if full_path.exists():
                found_files.append(file_path)
            else:
                missing_files.append(file_path)

        for file_path in optional_files:
            full_path = template_path / file_path
            if full_path.exists():
                found_files.append(f"{file_path} (optional)")

        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}"
            output = f"Found files: {', '.join(found_files)}"
            return ValidationResult(
                template_path=str(template_path),
                test_name="structure",
                success=False,
                output=output,
                error=error_msg,
                execution_time=0.0,
            )
        else:
            output = f"All required files found: {', '.join(found_files)}"
            return ValidationResult(
                template_path=str(template_path),
                test_name="structure",
                success=True,
                output=output,
                error="",
                execution_time=0.0,
            )

    def validate_compile(self, template_path: Path) -> ValidationResult:
        """Test cicero compile command."""
        logger.info(f"Testing compile: {template_path.name}")

        # FIXED: Added required --target parameter
        command = [
            "cicero",
            "compile",
            "--template",
            ".",
            "--target",
            "JSONSchema",
            "--warnings",
        ]
        success, output, error, exec_time = self._run_cicero_command(
            command, template_path
        )

        return ValidationResult(
            template_path=str(template_path),
            test_name="compile",
            success=success,
            output=output,
            error=error,
            execution_time=exec_time,
        )

    def validate_parse(self, template_path: Path) -> ValidationResult:
        """Test cicero parse command using sample.md."""
        logger.info(f"Testing parse: {template_path.name}")

        # Check if sample.md exists
        sample_file = template_path / "sample.md"
        if not sample_file.exists():
            return ValidationResult(
                template_path=str(template_path),
                test_name="parse",
                success=False,
                output="",
                error="sample.md file not found",
                execution_time=0.0,
            )

        command = [
            "cicero",
            "parse",
            "--template",
            ".",
            "--sample",
            "sample.md",
            "--warnings",
        ]
        success, output, error, exec_time = self._run_cicero_command(
            command, template_path
        )

        return ValidationResult(
            template_path=str(template_path),
            test_name="parse",
            success=success,
            output=output,
            error=error,
            execution_time=exec_time,
        )

    def validate_draft(self, template_path: Path) -> ValidationResult:
        """Test cicero draft command using request.json."""
        logger.info(f"Testing draft: {template_path.name}")

        # Check if request.json exists
        request_file = template_path / "request.json"
        if not request_file.exists():
            return ValidationResult(
                template_path=str(template_path),
                test_name="draft",
                success=False,
                output="",
                error="request.json file not found",
                execution_time=0.0,
            )

        command = [
            "cicero",
            "draft",
            "--template",
            ".",
            "--data",
            "request.json",
            "--warnings",
        ]
        success, output, error, exec_time = self._run_cicero_command(
            command, template_path
        )

        return ValidationResult(
            template_path=str(template_path),
            test_name="draft",
            success=success,
            output=output,
            error=error,
            execution_time=exec_time,
        )

    def validate_template(self, template_path: Path) -> List[ValidationResult]:
        """Run all validation tests on a single template."""
        logger.info(f"Validating template: {template_path}")

        if not template_path.exists():
            logger.error(f"Template path does not exist: {template_path}")
            return []

        if not template_path.is_dir():
            logger.error(f"Template path is not a directory: {template_path}")
            return []

        results = []

        # Test 1: Structure validation
        result = self.validate_template_structure(template_path)
        results.append(result)
        self.results.append(result)

        if not result.success:
            logger.error(f"Structure validation failed, skipping other tests")
            return results

        # Test 2: Compile validation
        result = self.validate_compile(template_path)
        results.append(result)
        self.results.append(result)

        # Test 3: Parse validation (only if compile succeeded)
        if result.success:
            result = self.validate_parse(template_path)
            results.append(result)
            self.results.append(result)
        else:
            logger.warning(f"Skipping parse test due to compile failure")

        # Test 4: Draft validation (only if compile succeeded)
        if results[1].success:  # Check compile result
            result = self.validate_draft(template_path)
            results.append(result)
            self.results.append(result)
        else:
            logger.warning(f"Skipping draft test due to compile failure")

        return results

    def find_all_templates(self) -> List[Path]:
        """Find all generated template directories."""
        logger.info(f"Scanning for templates in: {self.outputs_dir}")

        templates = []

        if not self.outputs_dir.exists():
            logger.error(f"Outputs directory not found: {self.outputs_dir}")
            return templates

        # Look for template directories (they should have the required structure)
        for contract_dir in self.outputs_dir.iterdir():
            if contract_dir.is_dir():
                for clause_dir in contract_dir.iterdir():
                    if clause_dir.is_dir():
                        # Check if it looks like a template directory
                        if (clause_dir / "text" / "grammar.tem.md").exists():
                            templates.append(clause_dir)

        logger.info(f"Found {len(templates)} template directories")
        return templates

    def validate_all_templates(self) -> Dict:
        """Validate all found templates."""
        logger.info("Starting validation of all templates...")

        templates = self.find_all_templates()

        if not templates:
            logger.error("No templates found to validate")
            return {"error": "No templates found"}

        total_results = []

        for template_path in templates:
            try:
                results = self.validate_template(template_path)
                total_results.extend(results)
            except Exception as e:
                logger.error(f"Error validating {template_path}: {e}")
                error_result = ValidationResult(
                    template_path=str(template_path),
                    test_name="validation",
                    success=False,
                    output="",
                    error=str(e),
                    execution_time=0.0,
                )
                total_results.append(error_result)
                self.results.append(error_result)

        return self._generate_summary_report()

    def _generate_summary_report(self) -> Dict:
        """Generate a comprehensive summary report."""
        if not self.results:
            return {"error": "No validation results"}

        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests

        # Group by test type
        test_types = {}
        for result in self.results:
            test_type = result.test_name
            if test_type not in test_types:
                test_types[test_type] = {"passed": 0, "failed": 0, "total": 0}

            test_types[test_type]["total"] += 1
            if result.success:
                test_types[test_type]["passed"] += 1
            else:
                test_types[test_type]["failed"] += 1

        # Group by template
        template_results = {}
        for result in self.results:
            template = result.template_path
            if template not in template_results:
                template_results[template] = {"passed": 0, "failed": 0, "total": 0}

            template_results[template]["total"] += 1
            if result.success:
                template_results[template]["passed"] += 1
            else:
                template_results[template]["failed"] += 1

        # Generate report
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (
                    (passed_tests / total_tests * 100) if total_tests > 0 else 0
                ),
                "total_templates": len(template_results),
                "templates_with_failures": sum(
                    1 for stats in template_results.values() if stats["failed"] > 0
                ),
            },
            "test_types": test_types,
            "template_results": template_results,
            "detailed_results": [result.to_dict() for result in self.results],
            "generated_at": datetime.now().isoformat(),
        }

        return report

    def save_report(self, report: Dict, output_file: Optional[str] = None):
        """Save validation report to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"validation_report_{timestamp}.json"

        output_path = self.project_root / "outputs" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to: {output_path}")

    def print_summary(self, report: Dict):
        """Print a formatted summary of validation results."""
        if "error" in report:
            print(f"ERROR: {report['error']}")
            return

        summary = report["summary"]

        print("\n" + "=" * 70)
        print("ACCORD PROJECT TEMPLATE VALIDATION RESULTS")
        print("=" * 70)

        print(f"Overall Results:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Templates Tested: {summary['total_templates']}")
        print(f"   Templates with Issues: {summary['templates_with_failures']}")

        print(f"\nTest Type Breakdown:")
        for test_type, stats in report["test_types"].items():
            success_rate = (
                (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            )
            print(
                f"   {test_type.title()}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)"
            )

        print(f"\nTemplate Results:")
        for template, stats in report["template_results"].items():
            template_name = Path(template).name
            status = "PASS" if stats["failed"] == 0 else "FAIL"
            print(
                f"   {status}: {template_name} - {stats['passed']}/{stats['total']} tests passed"
            )

        # Show failed tests details
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print(f"\nFailed Tests Details:")
            for result in failed_results:
                template_name = Path(result.template_path).name
                print(f"   {template_name} ({result.test_name}): {result.error}")

        print("\n" + "=" * 70)


def main():
    """Main entry point for template validation."""
    parser = argparse.ArgumentParser(
        description="Validate Accord Project templates using cicero CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate a specific template
    python -m src.validation.validate_templates --template outputs/generated_templates/contract1/clause1
    
    # Validate all generated templates
    python -m src.validation.validate_templates --scan-all
    
    # Validate with custom output
    python -m src.validation.validate_templates --scan-all --output my_report.json
    
    # Validate with verbose output
    python -m src.validation.validate_templates --scan-all --verbose
        """,
    )

    # Main options
    parser.add_argument(
        "--template", "-t", help="Path to specific template directory to validate"
    )

    parser.add_argument(
        "--scan-all",
        "-a",
        action="store_true",
        help="Scan and validate all templates in outputs directory",
    )

    parser.add_argument(
        "--output", "-o", help="Output file for detailed report (JSON format)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--no-save", action="store_true", help="Don't save report to file"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.template and not args.scan_all:
        parser.error("Must specify either --template or --scan-all")

    if args.template and args.scan_all:
        parser.error("Cannot specify both --template and --scan-all")

    # Initialize validator
    validator = AccordProjectValidator()

    try:
        if args.template:
            # Validate specific template
            template_path = Path(args.template)
            if not template_path.is_absolute():
                template_path = PROJECT_ROOT / template_path

            print(f"Validating template: {template_path}")
            validator.validate_template(template_path)
            report = validator._generate_summary_report()
        else:
            # Validate all templates
            print("Validating all templates...")
            report = validator.validate_all_templates()

        # Print summary
        validator.print_summary(report)

        # Save report
        if not args.no_save:
            validator.save_report(report, args.output)

        # Exit with appropriate code
        if "error" in report:
            sys.exit(1)
        elif report["summary"]["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
