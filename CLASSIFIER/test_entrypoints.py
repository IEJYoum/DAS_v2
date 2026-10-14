"""Deprecated entry points must never override the central experiment selection."""
import unittest
from unittest.mock import patch

from CLASSIFIER import run_real, run_bcc


class EntrypointTests(unittest.TestCase):
    def test_old_names_use_central_settings(self):
        for entrypoint in (run_real, run_bcc):
            with self.subTest(entrypoint=entrypoint.__name__):
                with patch.object(entrypoint, 'project_main', return_value='output') as central:
                    self.assertEqual(entrypoint.main(), 'output')
                    central.assert_called_once_with(mode=None)

    def test_prepare_override_keeps_central_dataset(self):
        for entrypoint in (run_real, run_bcc):
            with self.subTest(entrypoint=entrypoint.__name__):
                with patch.object(entrypoint, 'project_main') as central:
                    entrypoint.main(prepare_only=True)
                    central.assert_called_once_with(mode='prepare')


if __name__ == '__main__':
    unittest.main()
