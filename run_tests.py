import os
import subprocess
import sys


def run_tests(performance_only=False):
    coverage_file = ".coverage"
    if os.path.exists(coverage_file):
        try:
            os.remove(coverage_file)
            print(f"Successfully removed existing {coverage_file} file.")
        except OSError as e:
            print(f"Error removing {coverage_file}: {e}")
            print("Please ensure no other processes are using the file and try again.")
            sys.exit(1)

    command = [
        sys.executable, "-m", "pytest", "-v",
        "--cov=src", "--cov-report=html"
    ]
    if performance_only:
        command += ["-m", "performance"]

    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode == 0:
            print("All tests passed!")
        else:
            print(f"Tests failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("pytest is not installed or not found in your environment.")
        sys.exit(1)


if __name__ == "__main__":
    performance_only = "--performance" in sys.argv
    run_tests(performance_only)