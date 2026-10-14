"""Deprecated filename: forwards to the settings in run_project.py."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from CLASSIFIER.run_project import main as project_main


def main(prepare_only=False):
    print('run_bcc.py is deprecated; using run_project.py dataset/mode settings.', flush=True)
    return project_main(mode='prepare' if prepare_only else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args().prepare_only)
