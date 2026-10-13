"""Compatibility entry point for AH08. Edit settings only in run_project.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from CLASSIFIER.run_project import main as project_main


def main(prepare_only=False):
    return project_main(mode='prepare' if prepare_only else 'train_test', dataset='ah08')


if __name__ == '__main__':
    # Retain the two commonly used old commands; all other controls live centrally.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.prepare_only)
