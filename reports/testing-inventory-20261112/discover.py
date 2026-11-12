"""Document test definitions without importing tests, Torch or pytest."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def inventory():
    modules = []
    tracked = (
        subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.py"])
        .decode()
        .split("\0")
    )
    paths = [ROOT / name for name in sorted(filter(None, tracked))]
    for path in paths:
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=str(path))
        definitions = []

        def scan(body, prefix="", definitions=definitions):
            for node in body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name.startswith("test"):
                    definitions.append(
                        {
                            "name": prefix + node.name,
                            "line": node.lineno,
                            "docstring": ast.get_docstring(node),
                            "decorators": [ast.unparse(d) for d in node.decorator_list],
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    scan(node.body, prefix + node.name + ".")

        scan(tree.body)
        if not definitions:
            continue
        modules.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "purpose_docstring": ast.get_docstring(tree),
                "definition_count": len(definitions),
                "command": "uv run --no-sync pytest -q " + path.relative_to(ROOT).as_posix(),
                "tests": definitions,
            }
        )
    return {
        "schema_version": 1,
        "source_snapshot": "cb507d1fbc7facec4ab86ced137a3e4885c39e24",
        "tracked_python_files_scanned": len(paths),
        "nonstandard_module_paths": [
            m["path"]
            for m in modules
            if not (m["path"].startswith("tests/") and Path(m["path"]).name.startswith("test_"))
        ],
        "method": "All git-tracked Python files parsed with stdlib AST; module/class test* definitions, no imports/collection/execution",
        "count_semantics": "Module/class test* definitions, including nonstandard file/class names; not parametrized pytest cases, collection claims or PASS counts",
        "modules": len(modules),
        "test_definitions": sum(m["definition_count"] for m in modules),
        "test_modules": modules,
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2, ensure_ascii=False))
