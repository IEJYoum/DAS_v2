"""Shared setup contracts, using tiny files and mocked model launches only."""
import json
import os
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

import controler
from support import io_adapter as io
from support.ingest_sources import expand_source_spec, prompt_source_specs
from support.path_compat import resolve_native_path, is_foreign_path_text
from support.project_config import load_project_settings, save_project_settings, select_project_path
from support.shared_utils import flexMenu
from support.shared_utils import append_artifact_manifest_row
from data_extraction import segmentation_bridge
from CLASSIFIER import data, run_project

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'misc mIHC utility'))
import seg_v0 as seg


class ProjectIOTests(unittest.TestCase):
    def test_overlay_inheritance_origins_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / 'child'
            child.mkdir()
            (root / 'project_config.txt').write_text('a=parent\nb=baseline\n', encoding='utf-8')
            (root / 'project_config_posix.txt').write_text('b=linux\n', encoding='utf-8')
            (child / 'project_config.txt').write_text('a=child\nc=keep\n', encoding='utf-8')
            values, origins = load_project_settings(child, os_name='posix')
            self.assertEqual(values, {'a': 'child', 'c': 'keep', 'b': 'linux'})
            self.assertEqual(origins['b'], str(root / 'project_config_posix.txt'))
            self.assertEqual(load_project_settings(child, os_name='nt')[0]['b'], 'baseline')
            callback = []
            save_project_settings(child, {'a': '', 'd': 'new'}, on_update=callback.append)
            current, _ = load_project_settings(child)
            self.assertEqual(current['a'], 'parent')  # clear is local, not a tombstone
            self.assertEqual(current['c'], 'keep')
            self.assertEqual(current['d'], 'new')
            self.assertEqual(callback, [child.resolve()])

    def test_native_anchor_glob_and_strict_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'A').mkdir()
            target = root / 'A' / 'nuc.tif'
            target.touch()
            self.assertEqual(resolve_native_path('A/nuc.tif', base_dir=root, must_exist=True, kind='file'), target)
            self.assertEqual(expand_source_spec('**/*.tif', 'files', base_dir=root), [target])
            with self.assertRaises(FileNotFoundError):
                select_project_path(root / 'missing', 'image', must_exist=True)
            with self.assertRaises(ValueError):
                select_project_path(root, 'image', kind='file')
        self.assertTrue(is_foreign_path_text('/tmp/data', os_name='nt'))
        self.assertTrue(is_foreign_path_text(r'Z:\data', os_name='posix'))
        self.assertFalse(is_foreign_path_text('/mnt/rdscoussens/data', os_name='nt'))

    def test_routed_prompt_retry_and_shared_list_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answers = iter(['use', str(root)])
            metadata = []
            def ask(text, **kwargs):
                metadata.append(kwargs)
                return next(answers)
            self.assertEqual(select_project_path(root / 'bad', 'image folder', must_exist=True,
                interactive=True, input_fn=ask, print_fn=lambda *a: None), root)
            self.assertEqual(metadata[0]['prompt_meta']['options'][0]['value'], 'use')
            answers = iter([str(root), ''])
            self.assertEqual(prompt_source_specs('', input_fn=ask), [str(root)])
        answers = iter([' first ', 'first', 'first', ''])
        self.assertEqual(flexMenu(input_fn=lambda *a: next(answers)), [' first ', 'first', 'first'])

    def test_io_alias_is_one_state_in_both_import_orders(self):
        root = Path(__file__).resolve().parents[1]
        for imports in ['import io_adapter as a; from support import io_adapter as b',
                        'from support import io_adapter as b; import io_adapter as a']:
            code = f"import sys; sys.path.insert(0, 'support'); {imports}; assert a is b; a._MODE='contract'; assert b._MODE=='contract'"
            result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_controller_cache_invalidation_and_classifier_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controler._PROJECT_CONFIG_CACHE[str(root)] = {'old': 'value'}
            save_project_settings(root, {'new': 'value'}, on_update=controler.invalidate_project_config)
            self.assertNotIn(str(root), controler._PROJECT_CONFIG_CACHE)
            state = controler.SessionState()
            state.data_folder = root
            output = root/'run'
            output.mkdir()
            (output/'result.json').write_text('{"status":"complete"}')
            with patch.object(run_project, 'main', return_value=output) as launch:
                controler.startup_cell_classifier(state, mode='predict', sources=[{'id':'fixture'}],
                    checkpoint='parent.pt', output_root=str(root), training={}, inference={})
            self.assertEqual(launch.call_args.kwargs['mode'], 'predict')
            self.assertIs(launch.call_args.kwargs['input_fn'], io.iget)
            self.assertIs(launch.call_args.kwargs['config_changed'], controler.invalidate_project_config)

    def test_raw_tiff_parity_and_project_segmentation_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'nuc.tif'
            raw = np.arange(80, dtype=np.uint16).reshape(8, 10)
            tifffile.imwrite(image, raw)
            np.testing.assert_array_equal(data.read_plane(image), raw)
            np.testing.assert_array_equal(seg.read_2d_tiff(image, 'mask'), raw.astype(np.float32))
            checkpoint = root / 'parent.pt'
            checkpoint.touch()  # no model is loaded in this setup test
            save_project_settings(root, {'seg_custom_image': str(image),
                'seg_custom_checkpoint': str(checkpoint), 'seg_custom_output_root': str(root/'saved')})
            attrs = ['DAPI_PATH_FORMAT','MASK_PATH_FORMAT','CHECKPOINT_PATH','OUTPUT_ROOT','CORES',
                     'RUN_MODE','RM_0','CORE_FOLDER_FORMAT','MASK_FOLDER_FORMAT','RUN_CONTEXT','CORE']
            with patch.multiple(seg, **{key: getattr(seg, key) for key in attrs}):
                seg.configure(project_root=root, output_root=root/'explicit', mode='test', cores=['ROI'], print_fn=lambda *a: None)
                seg.CORE = 'ROI'
                self.assertEqual(seg.dapi_path(), image)
                self.assertEqual(seg.latest_model_path(), checkpoint)
                self.assertEqual(seg.OUTPUT_ROOT, root/'explicit')
                self.assertFalse((root/'explicit').exists())

    def test_custom_bridge_uses_noninteractive_child_with_explicit_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, checkpoint = root/'nuc.tif', root/'parent.pt'
            image.touch(); checkpoint.touch()
            save_project_settings(root, {'seg_custom_image': str(image),
                'seg_custom_checkpoint': str(checkpoint), 'seg_custom_output_root': str(root/'out')})
            answers = iter(['use', 'use', 'use', 'use'])
            def fake_child(cmd, cwd):
                product = root/'out'/'001_test'/'ROI_labeled_cells.tif'
                product.parent.mkdir(parents=True)
                tifffile.imwrite(product, np.zeros((8,10), np.uint32))
                manifest = append_artifact_manifest_row(product.parent, {'kind': 'labeled_cells_tif',
                    'status': 'complete', 'path': str(product)})
                Path(cmd[cmd.index('--result-json')+1]).write_text(json.dumps({'artifact_manifests': [str(manifest)]}))
                return 0
            with patch.object(segmentation_bridge, '_stream_subprocess', side_effect=fake_child) as child:
                result = segmentation_bridge.run_custom_interactive(project_root=root,
                    input_fn=lambda *a, **kw: next(answers), print_fn=lambda *a: None)
            cmd = child.call_args.args[0]
            self.assertEqual(result, root/'out')
            self.assertEqual(cmd[cmd.index('--checkpoint')+1], str(checkpoint))
            self.assertNotIn('--interactive', cmd)
            self.assertIn('--no-stardist', cmd)

            answers = iter(['use', 'use', 'use', 'use'])
            with patch.object(segmentation_bridge, '_stream_subprocess', return_value=0):
                with self.assertRaisesRegex(RuntimeError, 'without an output receipt'):
                    segmentation_bridge.run_custom_interactive(project_root=root,
                        input_fn=lambda *a, **kw: next(answers), print_fn=lambda *a: None)

    def test_custom_cli_fails_when_input_loading_fails(self):
        with patch.object(seg, 'CORES', ['ROI']), patch.object(seg, 'run_one_core', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'ROI'):
                seg.run_all(strict=True)

    def test_custom_standalone_forward_pass_receipt_and_parent_preservation(self):
        import torch
        from prototype_seg_v0 import NucleusSegNet
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, checkpoint, receipt = root/'nuc.tif', root/'parent.pt', root/'receipt.json'
            tifffile.imwrite(image, np.arange(32*32, dtype=np.uint16).reshape(32,32))
            torch.save(NucleusSegNet().state_dict(), checkpoint)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = [sys.executable, str(Path(seg.__file__)), '--image', str(image),
                '--checkpoint', str(checkpoint), '--output-root', str(root/'out'),
                '--cores', 'ROI', '--mode', 'test', '--no-stardist', '--result-json', str(receipt)]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifests = json.loads(receipt.read_text())['artifact_manifests']
            rows = [json.loads(line) for line in Path(manifests[0]).read_text().splitlines()]
            labels = next(row for row in rows if row['kind'] == 'labeled_cells_tif')
            self.assertEqual(tifffile.imread(labels['path']).shape, (32,32))
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
            self.assertTrue((Path(labels['path']).parent/'launch_context.json').is_file())


if __name__ == '__main__':
    unittest.main()
