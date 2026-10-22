"""Lineage aliases, agreement, artifact precedence and auditable ingest."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from support.annotation_agreement import agree_annotations
from support.label_conventions import (CELLTYPE_ALIASES, canonical_celltype, label_key,
                                        normalize_celltypes, resolve_unet_labels)


class LabelConventionTests(unittest.TestCase):
    def test_aliases_are_explicit_case_and_space_normalized(self):
        names = pd.Series(['mac','Mac ',' macrophage ','Myeloid','M1','m2'])
        self.assertEqual(normalize_celltypes(names).tolist(),['2: immune']*6)
        self.assertEqual(normalize_celltypes(names,output='model').tolist(),['IMMUNE']*6)
        with self.assertRaises(KeyError):
            canonical_celltype('Macrfophage')
        self.assertEqual(canonical_celltype('new lab name',overrides={'new lab name':'IMMUNE'}),'IMMUNE')

    def test_registry_has_no_conflicting_aliases(self):
        owners = {}
        for primary, aliases in CELLTYPE_ALIASES.items():
            for value in (primary,*aliases):
                key = label_key(value)
                self.assertIn(owners.get(key,primary),[primary],value)
                owners[key] = primary

    def test_canonicalize_then_intersect_and_artifact_overrides_missing_parent(self):
        table = pd.DataFrame({'Primary Celltype: Matrix':['2: immune','3: epithelial','5: stromal','2: immune'],
                              'Celltype: Gating':['Myeloid','Other Cells',None,None],
                              'Subtype: Gating':['mac','other non-immune','Artifact','NL']})
        result = resolve_unet_labels(table)
        self.assertEqual(result.labels.fillna('').tolist(),['IMMUNE','','ARTIFACT',''])
        self.assertEqual(result.summary['policy'],'intersection')
        self.assertEqual(result.summary['disagreements'],1)
        self.assertEqual(result.summary['artifact_rows'],1)
        self.assertIn('label_raw: Subtype: Gating',result.audit)

    def test_preferred_labels_keep_blanks_but_honor_global_artifact(self):
        table = pd.DataFrame({'DAS_unet_labels':['EPI',None,'IMMUNE'],
                              'class: mac':['mac','mac','artifact'],
                              'old Subtype: Gating':['Artifact']*3})
        result = resolve_unet_labels(table)
        self.assertEqual(result.labels.fillna('').tolist(),['EPI','','ARTIFACT'])
        self.assertEqual(result.summary['selected_columns'],['DAS_unet_labels'])
        self.assertEqual(result.summary['changed_by_artifact_override'],1)
        # An explicit alternate column is still available for deliberate control.
        result = resolve_unet_labels(table,label_column='class: mac')
        self.assertEqual(result.labels.tolist(),['IMMUNE','IMMUNE','ARTIFACT'])

    def test_single_class_column_and_primary_labels(self):
        table = pd.DataFrame({'class':['mac','endothelial','fibroblast','Tumor','Other Cells','NL']})
        result = resolve_unet_labels(table)
        self.assertEqual(result.labels.fillna('').tolist(),['IMMUNE','UNCLASSIFIED','UNCLASSIFIED','EPI','UNCLASSIFIED',''])
        self.assertEqual(canonical_celltype('Endothelial',output='primary'),'1: endothelial')

    def test_unknown_values_fail_even_when_another_convention_is_known(self):
        table = pd.DataFrame({'class':['not registered'],'Subtype: Gating':['Mac']})
        with self.assertRaisesRegex(ValueError,'Unmapped cell labels'):
            resolve_unet_labels(table)
        with self.assertRaisesRegex(ValueError,'Unmapped cell labels'):
            resolve_unet_labels(pd.DataFrame({'class':[99]}))
        table['Subtype: Gating'] = 'Artifact'
        self.assertEqual(resolve_unet_labels(table).labels.tolist(),['ARTIFACT'])
        self.assertEqual(canonical_celltype('artifact',overrides={'artifact':None}),'ARTIFACT')

    def test_shared_literal_agreement_retains_threshold_and_union_semantics(self):
        table = pd.DataFrame({'one':['A','A','yes','nan'], 'two':['A','B','no','B']})
        self.assertEqual(agree_annotations(table,1).tolist(),['A_','','',''])
        self.assertEqual(agree_annotations(table,.5).tolist(),['A_','A_B_','','B_'])
        self.assertEqual(agree_annotations(table,0).tolist(),['A_A_','A_B_','one_yes_','B_'])

    def test_ifa_named_shortcut_uses_shared_resolver_without_importing_legacy_runtime(self):
        path = Path(__file__).resolve().parents[1]/'IFA.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='agreeThresh')
        namespace = {'cm':SimpleNamespace(multiObMenu=lambda *args:['boss','matrix']),
                     'logInput':lambda prompt:'DAS_unet_labels'}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        table = pd.DataFrame({'boss':[' Mac ','Artifact'],'matrix':['2: immune','EPI']})
        with patch('builtins.print'):
            _, result, _ = namespace['agreeThresh'](pd.DataFrame(),table,pd.DataFrame())
        self.assertEqual(result.DAS_unet_labels.tolist(),['IMMUNE','ARTIFACT'])

    def test_staged_labels_keep_raw_audit_and_log_selection(self):
        import numpy as np
        import tifffile
        from CLASSIFIER import adapters, data, summary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'roi'
            raw.mkdir()
            mask = np.zeros((16,16),np.uint16)
            mask[1:4,1:4], mask[6:9,6:9], mask[11:14,11:14] = 1,2,3
            tifffile.imwrite(raw/'mask.tif',mask)
            tifffile.imwrite(raw/'marker.tif',np.arange(256,dtype=np.uint16).reshape(16,16))
            pd.DataFrame({'ObjectNumber':[1,2,3], 'class':['2: immune','Tumor','Tumor'],
                          'Subtype: Gating':[' Mac ','Artifact','Other Cells']}).to_csv(raw/'labels.csv',index=False)
            source=dict(adapter='classified_csv',image_root=str(root),mask_root=str(root),csv_root=str(root),
                        marker_patterns={m:'{scene_id}/marker.tif' for m in ['DNA','CD45','CK']},
                        marker_groups={'NUC':['DNA'],'IMMUNE':['CD45'],'EPI':['CK']},
                        mask_pattern='{scene_id}/mask.tif',csv_pattern='{scene_id}/labels.csv',
                        source_um_per_px=.5,resolution_note='fixture',scenes=[{'scene_id':'roi','split':'test'}])
            messages=[]
            manifest=adapters.stage(source,root/'staged',.5,print_fn=messages.append)
            meta_path=manifest.parent/'roi/scene.json'
            _,_,labels,meta=data.read_scene(meta_path)
            self.assertEqual(labels['class'].tolist(),['IMMUNE','ARTIFACT'])
            self.assertEqual(meta['label_resolution']['disagreements'],1)
            audit=pd.read_csv(meta_path.parent/'label_audit.csv')
            self.assertEqual(audit.label_reason.tolist(),['selected','artifact_override','disagreement'])
            self.assertEqual(audit['label_raw: Subtype: Gating'].tolist(),[' Mac ','Artifact','Other Cells'])
            self.assertTrue(any('intersection' in line and 'columns=' in line for line in messages))
            (manifest.parent/'config.json').write_text(json.dumps({'manifest':str(manifest),'evaluation_scope':'source'}))
            text=summary.write_debug_summary(manifest.parent).read_text()
            self.assertIn('Labels: intersection',text)
            self.assertIn('disagreements=1',text)

    def test_gui_uses_recognized_label_shortcut_without_column_prompt(self):
        from CLASSIFIER.gui_sources import _label_choice
        with patch('CLASSIFIER.gui_sources._column',side_effect=AssertionError('unnecessary label prompt')):
            self.assertEqual(_label_choice(None,'labels',['cellid','das_unet_labels','class'],{}),'auto')


if __name__=='__main__':
    unittest.main()
