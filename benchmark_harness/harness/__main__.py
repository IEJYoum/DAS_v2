"""python -m harness run|score   (run from DAS_v2/benchmark_harness)

  run   --label NAME --adapter ADAPTER --bundle BUNDLE [--config JSON]
  score --runs RUN_DIR [RUN_DIR ...] --out DIR
DAS (or anything else) calls this as a subprocess; nothing here is imported by DAS.
"""
import argparse
import json

from harness import runner, score


def main():
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--adapter", required=True)
    r.add_argument("--bundle", required=True)
    r.add_argument("--config", default="{}")
    s = sub.add_parser("score")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "run":
        runner.run(args.label, args.adapter, args.bundle, json.loads(args.config))
    else:
        score.score(args.runs, args.out)


if __name__ == "__main__":
    main()
