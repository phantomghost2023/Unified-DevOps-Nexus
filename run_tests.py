import os
import subprocess
import sys

def run_tests():
    coverage_file = ".coverage"
    if os.path.exists(coverage_file):
        try:
            os.remove(coverage_file)
            print(f"Successfully removed existing {coverage_file} file.")
        except OSError as e:
            print(f"Error removing {coverage_file}: {e}")
            print("Please ensure no other processes are using the file and try again.")
            sys.exit(1)

    command = ["pytest", "-v", "-m", "performance", "--cov=src", "--cov-report=html", "--cov-fail-under=100"]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    sys.exit(result.returncode)

if __name__ == "__main__":
    run_tests()