from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        from core import export
        print("ArgGYM")
        print()
        print("  python3 run.py gate <name>            run one gate")
        print("  python3 run.py gates                  list gates")
        print("  python3 run.py export <task> [path]   export one task as JSONL")
        print("  python3 run.py export-all             export every task")
        print("  python3 run.py inspect                start the inspector on :5000")
        print()
        print("tasks:", ", ".join(sorted(export._EXPORTABLE)))
        try:
            from validation import gates
            print("gates:", ", ".join(sorted(gates.ALL)))
        except Exception:
            print("gates: (validation/ not installed - optional)")
        return
    cmd = argv[0]
    if cmd == "gates":
        from validation import gates
        for n in sorted(gates.ALL):
            print(" ", n)
        return
    if cmd == "gate":
        from validation import gates
        gates.ALL[argv[1]]()
        return
    if cmd == "export":
        from core import export
        task = argv[1]
        path = argv[2] if len(argv) > 2 else f"{task}.jsonl"
        export.export_task(path, task)
        return
    if cmd == "export-all":
        from core import export
        for t in sorted(export._EXPORTABLE):
            export.export_task(f"{t}.jsonl", t)
        return
    if cmd == "inspect":
        import inspector
        inspector.app.run(debug=False, port=5000)
        return
    raise SystemExit(f"unknown command {cmd!r}; try --help")


if __name__ == "__main__":
    main()
