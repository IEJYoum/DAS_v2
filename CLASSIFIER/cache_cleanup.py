"""Preview/remove known generated CNN caches while preserving scientific results.

Default: normalized NPY/NPZ arrays only. Staged TIFF copies require the explicit
--include-staged option, because prepared manifests depend on those images.
Never select files merely because they are large or have an image extension.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat

from support.filesystem_utils import atomic_write_with_retry
from support.path_compat import resolve_native_path


def _json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def _reparse(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info,'st_file_attributes',0) &
                                    getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400))


def _inside(path, root):
    return path.resolve().is_relative_to(root)


def _files(root):
    """Walk literal directories without following symlinks/junctions."""
    for folder, dirs, files in os.walk(root, followlinks=False):
        parent = Path(folder)
        dirs[:] = [name for name in dirs if not _reparse(parent/name)]
        for name in files:
            path = parent/name
            if not _reparse(path) and _inside(path,root):
                yield path


def _owner(path):
    for parent in path.parents:
        config = parent/'config.json'
        if config.is_file():
            value = _json(config)
            if isinstance(value,dict) and {'model','class_names','target_um_per_px'} <= set(value):
                return parent
    return None


def _running(owner):
    result = _json(owner/'result.json')
    if result.get('status') in {'running','starting'}:
        return True
    summary = owner/'debug_summary.txt'
    if summary.is_file():
        text = summary.read_text(encoding='utf-8',errors='replace')
        if any(('Status: '+s) in text for s in ('running','starting')):
            return True
        if any(('Status: '+s) in text for s in ('complete','failed','interrupted')):
            return False
    if result.get('status') in {'complete','failed','interrupted'}:
        return False
    return not any((owner/name).is_file() for name in
                   ('training_summary.json','test_metrics.json','verification.json'))


def plan_cleanup(target, *, min_bytes=1024**2, include_staged=False):
    root = resolve_native_path(target,must_exist=True,kind='directory')
    if root == Path(root.anchor) or _reparse(root):
        raise ValueError('Choose a literal run/output folder, not a drive root or link')
    candidates, skipped, seen, owners = [], [], set(), {}
    files = list(_files(root))

    def offer(path, kind):
        # Validate the resolved target before selecting any object for deletion.
        path = Path(path)
        if not path.is_file() or _reparse(path) or not _inside(path,root):
            return
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        info = path.stat()
        if info.st_size < min_bytes:
            return
        owner = _owner(path)
        if owner not in owners:
            owners[owner] = owner is None or _running(owner)
        if owners[owner]:
            skipped.append(dict(path=str(path),reason='No recognized completed/failed CNN run, or run is active'))
            return
        candidates.append(dict(path=str(path),kind=kind,bytes=info.st_size,
                               mtime_ns=info.st_mtime_ns,run_folder=str(owner)))

    for path in files:
        if (path.suffix.lower() in {'.npy','.npz'} and
                any(p.name=='normalized_cache' for p in path.parents)):
            offer(path,'normalized_array')
        if include_staged and path.name=='scene.json':
            meta, evidence = _json(path), _json(path.parent/'ingest_report.json')
            if not evidence or not isinstance(meta.get('markers'),dict):
                continue
            original = {str(resolve_native_path(p)) for p in
                        [evidence.get('source_mask'),*evidence.get('source_markers',{}).values()] if p}
            for name in [meta.get('mask'),*meta['markers'].values()]:
                if not isinstance(name,str):
                    continue
                candidate = resolve_native_path(name,base_dir=path.parent)
                if candidate.suffix.lower() in {'.tif','.tiff'} and str(candidate) not in original:
                    offer(candidate,'staged_tiff_copy')
    return dict(target=str(root),min_bytes=min_bytes,include_staged=include_staged,
                candidates=sorted(candidates,key=lambda r:r['bytes'],reverse=True),
                skipped=skipped,total_bytes=sum(r['bytes'] for r in candidates))


def _write(path, payload):
    text = json.dumps(payload,indent=2)
    atomic_write_with_retry(path,lambda temporary:Path(temporary).write_text(text,encoding='utf-8'))


def apply_cleanup(plan):
    """Remove only individually validated files; never recursively delete a tree."""
    if _reparse(Path(plan['target'])):
        raise ValueError('Selected target became a link/junction')
    root = resolve_native_path(plan['target'],must_exist=True,kind='directory')
    receipt = root/('cache_cleanup_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
    record = {**plan,'status':'running','deleted':[],'errors':[]}
    _write(receipt,record)  # Keep the selected inventory before removing anything.
    for candidate in plan['candidates']:
        path = Path(candidate['path'])
        try:
            if not _inside(path,root) or _reparse(path):
                raise ValueError('Resolved deletion target escaped the selected folder or became a link')
            # Also reject a newly substituted junction anywhere in the path.
            if any(_reparse(p) for p in path.parents if p.is_relative_to(root)):
                raise ValueError('A parent directory became a link/junction')
            info = path.stat()
            if info.st_size!=candidate['bytes'] or info.st_mtime_ns!=candidate['mtime_ns']:
                raise ValueError('File changed since preview')
            owner = _owner(path)
            if owner is None or _running(owner):
                raise ValueError('Run is active or no longer recognized')
            if candidate['kind']=='normalized_array':
                if path.suffix.lower() not in {'.npy','.npz'} or not any(p.name=='normalized_cache' for p in path.parents):
                    raise ValueError('Object is not a normalized array')
            elif candidate['kind']=='staged_tiff_copy':
                if not plan['include_staged'] or path.suffix.lower() not in {'.tif','.tiff'}:
                    raise ValueError('Staged deletion was not requested')
            else:
                raise ValueError('Unknown disposable object kind')
            path.unlink()
            record['deleted'].append(candidate)
        except (OSError,ValueError) as exc:
            record['errors'].append(dict(path=str(path),error=str(exc)))
    record.update(status='complete' if not record['errors'] else 'complete_with_errors',
                  freed_bytes=sum(r['bytes'] for r in record['deleted']))
    _write(receipt,record)
    return receipt, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',help='Literal run folder or CLASSIFIER_OUTPUT parent')
    parser.add_argument('--apply',action='store_true',help='Actually remove selected objects; default is preview only')
    parser.add_argument('--include-staged',action='store_true',help='Also remove generated TIFF copies; prepared manifests will require restaging')
    parser.add_argument('--min-mib',type=float,default=1,help='Minimum file size (default 1 MiB)')
    parser.add_argument('--report',help='Optional JSON preview inventory path')
    args = parser.parse_args(argv)
    if not args.min_mib>=0 or args.min_mib==float('inf'):
        parser.error('--min-mib must be finite and nonnegative')
    plan = plan_cleanup(args.target,min_bytes=int(args.min_mib*1024**2),include_staged=args.include_staged)
    print(f"{'APPLY' if args.apply else 'PREVIEW'}: {len(plan['candidates'])} objects; {plan['total_bytes']/1024**3:.2f} GiB")
    print('Target:',plan['target'])
    print('Preserved: checkpoints, CSV/JSON/TXT records, debug PNGs, reports, and staged inputs by default.')
    if args.include_staged:
        print('Staged TIFF copies are included: recipes referencing these prepared inputs will need restaging from their original files.')
    for item in plan['candidates'][:20]:
        print(f"  {item['bytes']/1024**2:8.1f} MiB  {item['path']}")
    if plan['skipped']:
        print('Skipped active/unrecognized objects:',len(plan['skipped']))
    if args.report:
        _write(resolve_native_path(args.report),plan)
    if args.apply:
        receipt, record = apply_cleanup(plan)
        print(f"Freed {record['freed_bytes']/1024**3:.2f} GiB; receipt: {receipt}")
        return 1 if record['errors'] else 0
    print('Preview only. Add --apply to remove the listed objects.')
    return 0
