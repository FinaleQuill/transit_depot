"""2026-11-02 packaging entrypoint; NOT the historically executed script.

Historical execution used train_teacher.py, train_distill.py and study_student.py.
This wrapper dispatches to their unchanged copies only after an explicit CLI call.
See evidence/training for executed snapshots, arguments, checkpoints and logs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import runpy
import sys

TARGETS = {"teacher": "train_teacher.py", "distill": "train_distill.py", "study": "study_student.py"}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=sorted(TARGETS))
    options, remaining = parser.parse_known_args(argv)
    target = Path(__file__).resolve().parent / TARGETS[options.workflow]
    sys.argv = [str(target), *remaining]
    runpy.run_path(str(target), run_name="__main__")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
