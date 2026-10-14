"""Shared project policy over existing key/value I/O; no controller dependency.

Legacy relative values remain CWD-relative unless a caller explicitly supplies
base_dir. Origins remain available for a later project-relative migration.
"""
import os
from pathlib import Path

from .path_compat import resolve_native_path
from .shared_utils import load_project_config_values, save_project_config_updates, checkChange


def project_config_filename(os_name=None):
    return 'project_config.txt' if (os.name if os_name is None else os_name) == 'nt' else 'project_config_posix.txt'


def load_project_settings(folder, *, inherit=True, os_name=None):
    """Return effective strings plus per-key declaring file; nearest nonempty wins."""
    folder = resolve_native_path(folder, must_exist=True, kind='directory')
    filenames = ['project_config.txt']
    active = project_config_filename(os_name)
    if active not in filenames:
        filenames.append(active)
    values, origins = {}, {}
    for directory in [folder, *folder.parents] if inherit else [folder]:
        local, local_origins = {}, {}
        for filename in filenames:
            for key, value in load_project_config_values(directory, filename=filename).items():
                local[key] = value
                local_origins[key] = str(directory / filename)
        for key, value in local.items():
            if key not in values and str(value).strip():
                values[key], origins[key] = value, local_origins[key]
    return values, origins


def save_project_settings(folder, updates, *, on_update=None):
    folder = resolve_native_path(folder, must_exist=True, kind='directory')
    result = save_project_config_updates(folder, updates, filename=project_config_filename())
    if on_update is not None:
        on_update(folder)
    return result


def select_project_path(value, label, *, base_dir=None, must_exist=False, kind=None,
                        optional=False, interactive=False, input_fn=None, print_fn=None):
    """One path validation loop using existing use/change UI, or strict headless errors."""
    if interactive:
        from . import io_adapter as io
        input_fn, print_fn = input_fn or io.iget, print_fn or io.iprint
    while True:
        if interactive:
            value = checkChange(str(value or ''), label, input_fn=input_fn)
        try:
            if optional and not str(value or '').strip():
                return None
            return resolve_native_path(value, base_dir=base_dir, must_exist=must_exist, kind=kind)
        except (ValueError, OSError) as exc:
            if not interactive:
                raise
            if print_fn:
                print_fn(str(exc))
            value = input_fn(f'new {label}: ')
            # Validate replacement directly rather than asking to use it twice.
            try:
                if optional and not str(value or '').strip():
                    return None
                return resolve_native_path(value, base_dir=base_dir, must_exist=must_exist, kind=kind)
            except (ValueError, OSError) as replacement_error:
                if print_fn:
                    print_fn(str(replacement_error))
