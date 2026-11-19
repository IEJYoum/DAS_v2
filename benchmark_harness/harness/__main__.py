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
    b = sub.add_parser("bars")
    b.add_argument("--results", required=True)
    b.add_argument("--paper", type=float)
    b.add_argument("--paper-label", default="paper")
    b.add_argument("--title")
    x = sub.add_parser("boxes")
    x.add_argument("--results", required=True)
    x.add_argument("--metric", default="f1")
    x.add_argument("--title")
    x.add_argument("--unit", default="scenes", help="what one dot is, e.g. cores, FoVs")
    v = sub.add_parser("render")
    v.add_argument("--run", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--results")
    v.add_argument("--per-group", type=int, default=2)
    v.add_argument("--markers-per-scene", type=int, default=0, help="0 = all; else the N with most positives")
    args = parser.parse_args()
    if args.command == "run":
        runner.run(args.label, args.adapter, args.bundle, json.loads(args.config))
    elif args.command == "score":
        score.score(args.runs, args.out)
    elif args.command == "bars":
        from harness import report
        report.bars(args.results, args.paper, args.paper_label, args.title)
    elif args.command == "boxes":
        from harness import report
        report.boxes(args.results, args.metric, args.title, args.unit)
    else:
        from harness import report
        report.render(args.run, args.out, args.results, args.per_group, args.markers_per_scene)


if __name__ == "__main__":
    main()
