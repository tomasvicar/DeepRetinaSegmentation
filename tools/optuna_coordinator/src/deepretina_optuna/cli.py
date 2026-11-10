"""JSON-lines coordinator interface. Workers never import this package."""

import argparse
import json
import sys
from pathlib import Path

from .core import Coordinator, canonical_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--profiles", type=Path, required=True)
    p.add_argument("--baseline-csv", type=Path)
    p.add_argument("--baseline-qa", type=Path)
    a = p.parse_args()
    with Coordinator(
        a.db, json.loads(a.protocol.read_text()), json.loads(a.profiles.read_text())
    ) as c:
        if a.baseline_csv or a.baseline_qa:
            if c.finite:
                p.error(
                    "Finite v2 uses actual measured control0, not an imported baseline"
                )
            if not (a.baseline_csv and a.baseline_qa):
                p.error("Both baseline CSV andQA required")
            c.import_baseline(
                a.baseline_csv.read_text(),
                a.baseline_qa.read_bytes(),
                c.protocol["manual_baseline"]["qa_sha256"],
            )
        for line in sys.stdin:
            try:
                command = json.loads(line)
                op = command["op"]
                if op == "ask":
                    answer = c.ask(command["worker_id"], command.get("gpu_uuid"))
                elif op == "report":
                    answer = c.report(command["result"])
                elif op == "status":
                    answer = c.status()
                else:
                    raise ValueError("Unknown operation")
                sys.stdout.buffer.write(canonical_json({"ok": True, "answer": answer}))
                sys.stdout.buffer.flush()
            except (ValueError, KeyError, TypeError) as e:
                sys.stdout.buffer.write(
                    canonical_json(
                        {"ok": False, "error_type": type(e).__name__, "error": str(e)}
                    )
                )
                sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
