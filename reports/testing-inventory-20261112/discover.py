"""Document test definitions without importing tests, Torch or pytest."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def inventory():
    modules = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=str(path))
        definitions = []
        for item in tree.body:
            owners = [(item, "")]
            if isinstance(item, ast.ClassDef) and item.name.startswith("Test"):
                owners = [(method, item.name + ".") for method in item.body]
            for node, prefix in owners:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name.startswith("test_"):
                    definitions.append(
                        {
                            "name": prefix + node.name,
                            "line": node.lineno,
                            "docstring": ast.get_docstring(node),
                            "decorators": [ast.unparse(d) for d in node.decorator_list],
                        }
                    )
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
        "source_snapshot": "e27a7747c711c4714fd0616422b405a3132a34e6",
        "method": "stdlib AST only; no imports/collection/execution of test modules",
        "count_semantics": "Collected-looking top-level test functions and Test-class methods, not parametrized pytest cases or PASS counts",
        "modules": len(modules),
        "test_definitions": sum(m["definition_count"] for m in modules),
        "test_modules": modules,
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2, ensure_ascii=False))
