from __future__ import annotations

import unittest

from support.path_compat import is_foreign_path_text, native_path_text


class PathCompatTests(unittest.TestCase):
    def test_known_windows_share_maps_to_linux_mount(self):
        self.assertEqual(
            native_path_text(
                r"\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies\Alex",
                os_name="posix",
            ),
            "/mnt/rdscoussens/Multiplex_IHC_studies/Alex",
        )

    def test_known_linux_mount_maps_to_windows_share(self):
        self.assertEqual(
            native_path_text("/mnt/rdscoussens/Multiplex_IHC_studies/Alex", os_name="nt"),
            r"\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies\Alex",
        )

    def test_unrelated_foreign_path_is_not_invented(self):
        path = r"\\other-server\research\dataset"
        self.assertEqual(native_path_text(path, os_name="posix"), path)
        self.assertTrue(is_foreign_path_text(path, os_name="posix"))
