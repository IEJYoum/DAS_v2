from __future__ import annotations

import unittest

import pandas as pd

import controler
import spine


class SpineTests(unittest.TestCase):
    def _state(self):
        state = controler.SessionState()
        state.df = pd.DataFrame({"marker": [1.0, 2.0]}, index=["a", "b"])
        state.obs = pd.DataFrame({"class": ["x", "y"]}, index=["a", "b"])
        state.dfxy = pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=["a", "b"])
        state.logdf = pd.DataFrame({"event": ["loaded"]})
        state.stem = "baseline"
        return state

    def test_restore_replaces_mutated_triplet_and_log(self):
        state = self._state()
        spine.capture_home_baseline(state)

        state.df["temporary"] = [9.0, 9.0]
        state.obs["temporary"] = "changed"
        state.dfxy.loc["a", "x"] = 99
        state.logdf = pd.DataFrame({"event": ["mutated"]})
        state.stem = "mutated"

        self.assertTrue(spine.restore_home_baseline(state))
        self.assertListEqual(list(state.df.columns), ["marker"])
        self.assertListEqual(list(state.obs.columns), ["class"])
        self.assertEqual(int(state.dfxy.loc["a", "x"]), 1)
        self.assertEqual(state.logdf.iloc[0, 0], "loaded")
        self.assertEqual(state.stem, "baseline")

    def test_keyboard_interrupt_restores_then_returns_to_home_loop(self):
        state = self._state()
        spine.capture_home_baseline(state)
        calls = {"main": 0}

        def startup_menu(_state):
            self.fail("Loaded data should route to main_menu.")

        def main_menu(active_state):
            calls["main"] += 1
            if calls["main"] == 1:
                active_state.df["temporary"] = [9.0, 9.0]
                active_state.stem = "mutated"
                raise KeyboardInterrupt
            return False

        spine.run_session(state, startup_menu=startup_menu, main_menu=main_menu)

        self.assertEqual(calls["main"], 2)
        self.assertListEqual(list(state.df.columns), ["marker"])
        self.assertEqual(state.stem, "baseline")


if __name__ == "__main__":
    unittest.main()
