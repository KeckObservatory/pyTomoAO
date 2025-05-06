import os
import subprocess
import argparse
import glob
from pathlib import Path
from typing import Dict, Tuple, List, Any

def count_file_lines(file_path: str) -> int:
    """Count the number of non-empty, non-comment lines in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        count = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                count += 1
        return count
    except (FileNotFoundError, IOError):
        return 0

def count_directory_lines(directory: str) -> int:
    """Recursively count lines in all Python files in a directory."""
    total_lines = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                total_lines += count_file_lines(file_path)
    
    return total_lines

def get_coverage_classification(coverage_percentage: float) -> Dict[str, Any]:
    """
    Classify coverage percentage with detailed information.
    
    Args:
        coverage_percentage: The coverage percentage as a float
        
    Returns:
        Dict: Detailed classification information
    """
    if coverage_percentage < 40:
        level = 'low'
        risk = 'high'
        recommendation = 'Urgent action needed'
    elif coverage_percentage < 75:
        level = 'medium'
        risk = 'moderate'
        recommendation = 'Improvements recommended'
    else:
        level = 'high'
        risk = 'low'
        recommendation = 'Good coverage, maintain or improve'
    
    return {
        'level': level,
        'risk': risk,
        'recommendation': recommendation,
        'score': coverage_percentage
    }

def run_pytest_coverage(test_dir: str, cov_target_dir: str, repo_dir: str = ".", verbose: bool = False) -> Dict[str, Any]:
    """
    Run pytest with coverage.

    Args:
        test_dir: The directory containing pytest tests.
        cov_target_dir: The directory whose coverage should be measured.
        repo_dir: The repository root directory. Defaults to current directory.
        verbose: Whether to print detailed debug information.

    Returns:
        Dict with coverage results.
    """
    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    current_pythonpath = env.get('PYTHONPATH', '')
    repo_path = os.path.abspath(os.path.expanduser(repo_dir))

    # Add repo directory to PYTHONPATH
    if current_pythonpath:
        env['PYTHONPATH'] = f"{repo_path}:{current_pythonpath}"
    else:
        env['PYTHONPATH'] = repo_path

    # Normalize paths
    test_path = os.path.abspath(os.path.join(repo_path, test_dir))
    cov_target_path = os.path.abspath(os.path.join(repo_path, cov_target_dir))
    cov_target_rel = os.path.relpath(cov_target_path, repo_path)

    if verbose:
        print(f"Repository path: {repo_path}")
        print(f"Test path: {test_path}")
        print(f"Coverage target path: {cov_target_path}")
        print(f"Coverage target relative: {cov_target_rel}")
        print(f"PYTHONPATH: {env['PYTHONPATH']}")

    # Prepare pytest command
    pytest_cmd = ["pytest", test_path]
    if verbose:
        pytest_cmd.append("-v")

    pytest_cmd.extend([
        f"--cov={cov_target_rel}",
        "--cov-report=term-missing"
    ])

    if verbose:
        print(f"Running from: {repo_path}")
        print(f"Running command: {' '.join(pytest_cmd)}")

    # Run pytest with coverage from the repository root
    try:
        result = subprocess.run(
            pytest_cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_path
        )

        if verbose:
            print("\n--- pytest stdout ---")
            print(result.stdout)
            print("--- pytest stderr ---")
            print(result.stderr)
            print("---------------------")

        # Parse coverage data from output
        coverage_data = {
            'coverage': 0.0,
            'statements': 0,
            'missing': 0,
            'passed': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr
        }

        # Extract coverage statistics from the TOTAL line
        coverage_found = False
        for line in result.stdout.split('\n'):
            if line.startswith("TOTAL") and "%" in line:
                if verbose:
                    print(f"Found TOTAL line: {line.strip()}")
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '%' in part:
                            coverage_data['statements'] = int(parts[i-2])
                            coverage_data['missing'] = int(parts[i-1])
                            coverage_data['coverage'] = float(part.rstrip('%'))
                            coverage_found = True
                            break
                except (IndexError, ValueError) as e:
                    if verbose:
                        print(f"Failed to parse total coverage data from line: {line} - {e}")
                break

        if not coverage_found and verbose:
             print("WARNING: Could not find or parse TOTAL coverage line in pytest output.")
             tests_collected = 0
             for line in result.stdout.split('\n'):
                 if "collected" in line.lower():
                     try:
                         tests_collected = int(line.split("collected")[1].split()[0])
                         break
                     except (IndexError, ValueError):
                         pass
             if tests_collected == 0:
                 print("No tests were collected! Check test naming and pytest configuration.")

        return coverage_data

    except subprocess.SubprocessError as e:
        print(f"Failed to run pytest.")
        print(f"Error: {e}")
        return {
            'coverage': 0.0,
            'statements': 0,
            'missing': 0,
            'passed': False,
            'output': '',
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Analyze pyTomoAO test coverage')
    parser.add_argument('--repo-dir', type=str, default=".",
                      help='Path to the repository root directory (default: .)')
    parser.add_argument('--test-dir', type=str, default="tests",
                      help='Directory containing tests relative to repo-dir (default: tests)')
    parser.add_argument('--source-dir', type=str, default="pyTomoAO",
                      help='Source directory for coverage analysis relative to repo-dir (default: pyTomoAO)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Show detailed output during analysis')
    parser.add_argument('--coverage-threshold', type=float, default=75.0,
                      help='Minimum required test coverage percentage (default: 75.0)')
    parser.add_argument('--line-count-threshold', type=int, default=150,
                      help='Only analyze if source directory has more than this many lines (default: 150)')
    parser.add_argument('--fail-on-problems', action='store_true',
                      help='Exit with error code if coverage is below threshold or tests fail')
    parser.add_argument('--debug', action='store_true',
                      help='Print extensive debugging information')
    args = parser.parse_args()

    if args.debug:
        args.verbose = True

    repo_dir_abs = os.path.abspath(os.path.expanduser(args.repo_dir))
    test_dir_abs = os.path.abspath(os.path.join(repo_dir_abs, args.test_dir))
    source_dir_abs = os.path.abspath(os.path.join(repo_dir_abs, args.source_dir))

    print(f"Repository Root: {repo_dir_abs}")
    print(f"Test Directory: {test_dir_abs}")
    print(f"Source Directory: {source_dir_abs}")

    line_count = count_directory_lines(source_dir_abs)
    print(f"Line count for {args.source_dir}: {line_count}")

    if line_count <= args.line_count_threshold:
        print(f"Skipping analysis: Source directory has {line_count} lines (threshold: {args.line_count_threshold})")
        exit(0)

    print(f"\nRunning pytest coverage for '{args.source_dir}'...")
    coverage_results = run_pytest_coverage(
        test_dir=args.test_dir,
        cov_target_dir=args.source_dir,
        repo_dir=args.repo_dir,
        verbose=args.verbose
    )

    classification = get_coverage_classification(coverage_results['coverage'])
    tests_passed = coverage_results['passed']
    coverage_pct = coverage_results['coverage']

    print("\n\n" + "=" * 80)
    print("                         TEST COVERAGE ANALYSIS SUMMARY                         ")
    print("=" * 80)
    print(f"{'Target':<20} | {'Lines':<7} | {'Coverage':<25} | {'Statements':<15} | {'Status'}")
    print("-" * 80)

    status = "✅ PASSED" if tests_passed else "❌ FAILED"
    is_problematic = False
    if coverage_pct < args.coverage_threshold:
        status += " (Low Coverage ⚠️)"
        is_problematic = True
    if not tests_passed:
        is_problematic = True

    coverage_str = f"{coverage_pct:.1f}% ({classification['level'].upper()})"
    statements_str = f"{coverage_results['statements'] - coverage_results['missing']}/{coverage_results['statements']}"

    print(f"{args.source_dir:<20} | {line_count:<7} | {coverage_str:<25} | {statements_str:<15} | {status}")
    print("-" * 80)

    if args.verbose:
        print("\nDetailed Analysis:")
        print("-" * 80)
        print(f"  Target Directory: {source_dir_abs}")
        print(f"  Coverage: {coverage_pct:.1f}% ({classification['level'].upper()})")
        print(f"  Statements: {statements_str} ({coverage_results['missing']} missing)")
        print(f"  Risk Level: {classification['risk']}")
        print(f"  Recommendation: {classification['recommendation']}")
        print(f"  Tests Passed: {'Yes' if tests_passed else 'No'}")
        print("-" * 80)
        if coverage_results['error']:
             print("\nErrors during execution:")
             print(coverage_results['error'])
             print("-" * 80)

    if is_problematic:
        print("\nProblem detected:")
        if not tests_passed:
            print(f"- Tests failed.")
        if coverage_pct < args.coverage_threshold:
            print(f"- Coverage ({coverage_pct:.1f}%) is below threshold ({args.coverage_threshold:.1f}%).")

        if args.fail_on_problems:
            print("\nExiting with error code due to problems found.")
            exit(1)
        else:
            print("\nWARNING: Problems found, but --fail-on-problems not set.")
    else:
        print("\nAnalysis complete. No problems detected based on threshold and test status.")

if __name__ == "__main__":
    main()
