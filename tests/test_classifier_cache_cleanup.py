"""Cleanup deletes only recognized disposable files inside the literal target."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from CLASSIFIER.cache_cleanup import plan_cleanup, apply_cleanup


class CacheCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'run'
        self.run.mkdir()
        (self.run/'config.json').write_text(json.dumps({'model':{},'class_names':[], 'target_um_per_px':.5}))
        (self.run/'result.json').write_text('{"status":"complete"}')
        self.cache = self.run/'normalized_cache'/'roi'
        self.cache.mkdir(parents=True)
        self.array = self.cache/'CD45.npy'
        self.array.write_bytes(b'disposable')
        self.staged = self.run/'dataset'/'roi'
        self.staged.mkdir(parents=True)
        self.tiff = self.staged/'CD45.tif'
        self.tiff.write_bytes(b'copy of image')
        (self.staged/'scene.json').write_text(json.dumps({'markers':{'CD45':'CD45.tif'},'mask':'mask.tif'}))
        (self.staged/'ingest_report.json').write_text(json.dumps({'source_markers':{'CD45':str(self.root/'original.tif')}}))
        self.keep = []
        for name in ['best.pt','predictions.csv','metrics.json','debug_summary.txt','errors0.png','original.tif']:
            p = self.run/name
            p.write_bytes(b'preserve')
            self.keep.append(p)

    def test_preview_keeps_every_file_and_selects_only_normalized_arrays(self):
        plan = plan_cleanup(self.run,min_bytes=0)
        self.assertEqual([Path(c['path']) for c in plan['candidates']],[self.array])
        self.assertTrue(self.array.exists())
        self.assertTrue(self.tiff.exists())
        receipt, record = apply_cleanup(plan)
        self.assertTrue(receipt.is_file())
        self.assertEqual(record['status'],'complete')
        self.assertFalse(self.array.exists())
        self.assertTrue(self.tiff.exists())
        self.assertTrue(all(p.exists() for p in self.keep))

    def test_staged_copies_require_explicit_option_and_keep_metadata(self):
        plan = plan_cleanup(self.run,min_bytes=0,include_staged=True)
        self.assertEqual({Path(c['path']) for c in plan['candidates']},{self.array,self.tiff})
        _, record = apply_cleanup(plan)
        self.assertEqual(record['status'],'complete')
        self.assertFalse(self.tiff.exists())
        self.assertTrue((self.staged/'scene.json').exists())
        self.assertTrue((self.staged/'ingest_report.json').exists())

    def test_active_run_and_unrecognized_sources_are_protected(self):
        (self.run/'result.json').write_text('{"status":"running"}')
        self.assertFalse(plan_cleanup(self.root,min_bytes=0,include_staged=True)['candidates'])
        (self.run/'result.json').write_text('{"status":"complete"}')
        (self.run/'config.json').unlink()
        self.assertFalse(plan_cleanup(self.root,min_bytes=0,include_staged=True)['candidates'])

    def test_modified_file_and_forged_outside_target_are_rejected(self):
        plan = plan_cleanup(self.run,min_bytes=0)
        self.array.write_bytes(b'new cache contents')
        _, record = apply_cleanup(plan)
        self.assertEqual(len(record['errors']),1)
        self.assertTrue(self.array.exists())
        external = self.root/'outside.npy'
        external.write_bytes(b'keep outside')
        forged = deepcopy(plan)
        forged['candidates'][0]['path'] = str(external)
        _, record = apply_cleanup(forged)
        self.assertTrue(record['errors'])
        self.assertTrue(external.exists())

    def test_symlink_is_not_followed(self):
        external = self.root/'external'
        external.mkdir()
        (external/'outside.npy').write_bytes(b'keep outside')
        try:
            (self.cache/'linked').symlink_to(external,target_is_directory=True)
        except OSError:
            self.skipTest('Creating Windows symlinks is unavailable')
        plan = plan_cleanup(self.run,min_bytes=0)
        self.assertNotIn(str(external/'outside.npy'),[c['path'] for c in plan['candidates']])


if __name__=='__main__':
    unittest.main()
