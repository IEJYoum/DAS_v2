"""Press Run for a preview, or supply a target folder and --apply from a terminal."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from CLASSIFIER.cache_cleanup import main

TARGET_FOLDER = r'\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies\Isaac_Youm\CLASSIFIER_PROJECT\runs'

if __name__ == '__main__':
    raise SystemExit(main([TARGET_FOLDER] if len(sys.argv)==1 else None))
