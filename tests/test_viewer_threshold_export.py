"""Exercise generated viewer/threshold JavaScript in a real browser, without TIFFs."""
import html
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visualization import visu_html_functions7 as vhf

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


@unittest.skipUnless(EDGE.is_file(), "Headless Edge is required for generated JavaScript tests")
class ThresholdExportTests(unittest.TestCase):
    def browser_probe(self, path, script):
        source = path.read_text(encoding="utf-8")
        source = source.replace("</body>", "<script>\n" + script + "\n</script></body>")
        path.write_text(source, encoding="utf-8")
        with tempfile.TemporaryDirectory() as profile:
            result = subprocess.run([
                str(EDGE), "--headless", "--disable-gpu", "--no-first-run",
                "--allow-file-access-from-files", "--user-data-dir=" + profile,
                "--dump-dom", "--virtual-time-budget=8000", path.as_uri(),
            ], capture_output=True, encoding="utf-8", timeout=45,
                creationflags=subprocess.CREATE_NO_WINDOW)
        match = re.search(r'<pre id="probe-result">(.*?)</pre>', result.stdout, re.S)
        self.assertIsNotNone(match, result.stdout[-1800:] + result.stderr[-1000:])
        report = json.loads(html.unescape(match.group(1)))
        self.assertNotIn("error", report, report)
        return report

    def test_saved_thresholds_export_selected_scenes_with_exact_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenes = ["SceneA", "SceneB", "SceneC", "SceneD", "SceneE"]
            cores, tiles = {}, {}
            for scene in scenes:
                rows = [{"row_index": scene + "_cell" + str(i), "x": i, "y": i,
                         "expr": {"Pin1": value}} for i, value in [(1, 5), (2, 15)]]
                rel = scene + ".js"
                (root / rel).write_text("window.__ROI_CORE_PAYLOAD__ = " + json.dumps({"rows": rows}) + ";", encoding="utf-8")
                cores[scene] = {"slide_scene": scene, "width": 8, "height": 8,
                                "payload_rel": rel, "row_count": 2}
                tiles[scene] = [{"tile_kind": "composite", "core": scene,
                                 "channels": [{"marker": "Pin1", "rel": "image.png"}]}]
            data = {
                "core_tiles": tiles, "core_meta": {}, "scene_sources": {},
                "roi_data": {"cores": cores, "marker_list": ["Pin1"], "has_expression_data": True},
                "threshold_store": {"mode": "study_thresholds", "roi_ids": scenes,
                                    "marker_list": ["Pin1"], "core_to_roi_id": dict(zip(scenes, scenes))},
                "groupings": {"all data": {"all data": scenes}},
                "view_sets": [{"id": "all", "group": "all data", "value": "all data",
                               "core_names": scenes, "layout": "compact"}],
                "default_view_id": "all", "roi_mailbox": {},
            }
            main_path = Path(vhf.write_catalog_viewer_html(root, data))
            report = self.browser_probe(main_path, """
try {
  const payload = currentThreshSnapshotForCore('SceneA');
  const box = document.createElement('pre'); box.id = 'probe-result';
  const body = document.getElementById('workspace').textContent;
  box.textContent = JSON.stringify({payload: payload, initFailed: body.includes('Viewer init failed:'), body: body});
  document.body.appendChild(box);
} catch (err) {
  const box = document.createElement('pre'); box.id = 'probe-result';
  box.textContent = JSON.stringify({error: String(err)}); document.body.appendChild(box);
}
""")
            self.assertFalse(report["initFailed"], report["body"])
            payload = report["payload"]
            self.assertEqual([p["roi_id"] for p in payload["threshold_scene_payloads"]], scenes)
            self.assertNotIn("rows", payload["threshold_scene_payloads"][0])
            payload["threshold_store"]["saved_thresholds"] = {}
            payload["threshold_store"]["study_thresholds_path"] = "test-only/studythresholds.csv"
            runtime = Path(vhf.write_thresh_runtime_html(root))
            source = runtime.read_text(encoding="utf-8")
            launch = "<script>localStorage.clear(); history.replaceState(null, '', '?key=probe'); localStorage.setItem('probe', " + json.dumps(json.dumps(payload)) + ");</script>"
            source = source.replace("<script>", launch + "\n<script>", 1)
            source = source.replace("await boot();", "await boot(); window.probeBooted = true;")
            runtime.write_text(source, encoding="utf-8")
            result = self.browser_probe(runtime, """
(async function() {
  const box = document.createElement('pre'); box.id = 'probe-result'; document.body.appendChild(box);
  const check = function(ok, message) { if (!ok) throw new Error(message); };
  try {
    for (let i = 0; i < 100 && !window.probeBooted; i++) await new Promise(r => setTimeout(r, 25));
    check(window.probeBooted, 'Threshold UI did not boot: ' + el('errorBox').textContent);
    const boxes = Array.from(document.querySelectorAll('#applyRoiList input[type="checkbox"]'));
    check(boxes.length === 5, 'Scene checklist did not render');
    check(el('savedThresholdTable').textContent.includes('SceneB'), 'Saved table did not render');
    el('applyAllToggle').click();
    state.xMarker = 'Pin1'; el('xMarker').value = 'Pin1'; el('thresholdInput').value = '10';
    el('saveMemoryBtn').click();
    check(Object.keys(DATA.threshold_store.saved_thresholds).length === 5, 'Apply all / Save to Memory did not save five scenes');
    boxes.forEach(b => b.checked = b.value === 'SceneB');
    el('thresholdInput').value = '3'; el('saveMemoryBtn').click();
    check(DATA.threshold_store.saved_thresholds.SceneA.Pin1 === 10, 'Changing SceneB overwrote SceneA');
    delete DATA.threshold_store.saved_thresholds.SceneC;
    DATA.threshold_store.saved_thresholds.SceneD.Pin1 = null;
    DATA.threshold_store.saved_thresholds.SceneE.Pin1 = '';
    boxes.forEach(b => b.checked = true);
    el('thresholdInput').value = '999';
    const expected = [{index:'SceneA_cell1',label:'-'},{index:'SceneA_cell2',label:'+'},
                      {index:'SceneB_cell1',label:'+'},{index:'SceneB_cell2',label:'+'}];
    const labels = await thresholdAssignments();
    check(JSON.stringify(labels) === JSON.stringify(expected), 'Per-scene saved values or unset-scene skip failed');
    check(_buildMailboxCsv('thresh_Pin1', labels).split('\\n').length === 6, 'Mailbox CSV format changed');
    boxes.forEach(b => b.checked = b.value === 'SceneB');
    check((await thresholdAssignments()).length === 2, 'Unchecked scene was exported');
    boxes.forEach(b => b.checked = b.value === 'SceneC');
    check((await thresholdAssignments()).length === 0, 'Unset scene was not skipped');
    boxes.forEach(b => b.checked = true);
    let writes = 0; const downloaded = [];
    DATA.writer_url = 'http://test.invalid/ifa_roi_patch';
    DATA.threshold_store.writer_url = 'http://test.invalid/study_threshold';
    window.fetch = async function(url, opts) {
      writes++; const body = JSON.parse(opts.body);
      if (url.endsWith('/ifa_roi_patch')) check(JSON.stringify(body.assignments) === JSON.stringify(expected), 'Wrong labels POSTed');
      return {ok:true, json:async function() {return {ok:true};}};
    };
    await saveThreshold(); check(writes === 2, 'Study table and mailbox not saved');
    await saveThreshold(); check(writes === 2, 'Duplicate save was not suppressed');
    _downloadCsvFallback = function(column, assignments) { downloaded.push({column:column, assignments:assignments}); };
    _downloadThresholdFallback = function() {};
    window.fetch = async function() { throw new Error('writer offline'); };
    state.lastSavedSignature = '';
    await saveThreshold();
    check(downloaded.length === 1 && downloaded[0].assignments.length === 4, 'Offline CSV fallback lost scenes');
    DATA.threshold_scene_payloads.find(s => s.core === 'SceneB').payload_rel = 'missing.js';
    let failed = '';
    try { await saveThreshold(); } catch (err) { failed = String(err); }
    check(failed.includes('SceneB'), 'Missing sidecar did not name the failing scene');
    check(downloaded.length === 1 && !state.saveInFlight && !el('saveBtn').disabled, 'Failed load wrote partial mail or left save disabled');
    DATA.threshold_scene_payloads.find(s => s.core === 'SceneB').payload_rel = 'SceneB.js';
    DATA.threshold_store.saved_thresholds.SceneB.Pin1 = 0;
    check((await thresholdAssignments()).filter(a => a.index.startsWith('SceneB')).every(a => a.label === '+'), 'Zero threshold was skipped');
    DATA.threshold_store = null;
    el('thresholdInput').value = '10';
    check((await thresholdAssignments()).length === 2, 'Standalone current-core export regressed');
    box.textContent = JSON.stringify({passed:true, labels:labels, sceneCheckboxes:boxes.length,
      buttons:['Save to Memory','Save to CSV'].every(text => document.body.textContent.includes(text))});
  } catch (err) { box.textContent = JSON.stringify({error:String(err), stack:err.stack}); }
})();
""")
            self.assertTrue(result["passed"])
            self.assertTrue(result["buttons"])


if __name__ == "__main__":
    unittest.main()
