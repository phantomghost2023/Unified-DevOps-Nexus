# fix_test_imports.py
import os
import re

def fix_imports_in_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # Replace 'from src.' with 'from core.'
    content_new = re.sub(r'from\s+src\.', 'from core.', content)
    # Replace 'import src.' with 'import core.'
    content_new = re.sub(r'import\s+src\.', 'import core.', content_new)
    if content != content_new:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content_new)
        print(f"Fixed imports in: {filepath}")

def walk_and_fix_tests(root="tests"):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                fix_imports_in_file(os.path.join(dirpath, filename))

if __name__ == "__main__":
    walk_and_fix_tests()
    print("All test imports fixed. Now run:\n  git add tests/\n  git commit -m \"fix: auto-fix all test imports\"\n  git push")