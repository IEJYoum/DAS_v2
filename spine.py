"""Small, transport-independent DAS session loop."""

from __future__ import annotations

import io_adapter as io


def capture_home_baseline(state) -> None:
    """Keep a complete in-memory triplet baseline for interrupt recovery."""
    state.home_df = state.df.copy(deep=True)
    state.home_obs = state.obs.copy(deep=True)
    state.home_dfxy = state.dfxy.copy(deep=True)
    state.home_logdf = state.logdf.copy(deep=True)
    state.home_stem = str(state.stem)


def restore_home_baseline(state) -> bool:
    """Restore the last captured baseline without changing project context."""
    if any(
        value is None
        for value in (
            state.home_df,
            state.home_obs,
            state.home_dfxy,
            state.home_logdf,
            state.home_stem,
        )
    ):
        return False

    # Copy again so a later in-place legacy mutation cannot corrupt the snapshot.
    state.df = state.home_df.copy(deep=True)
    state.obs = state.home_obs.copy(deep=True)
    state.dfxy = state.home_dfxy.copy(deep=True)
    state.logdf = state.home_logdf.copy(deep=True)
    state.stem = str(state.home_stem)
    return True


def run_session(state, *, startup_menu, main_menu):
    """Run the two DAS root menus and handle propagated Ctrl-C once."""
    while True:
        try:
            keep_running = main_menu(state) if state.has_data() else startup_menu(state)
            if not keep_running:
                return state
        except KeyboardInterrupt:
            io.clear_progress()
            restored = restore_home_baseline(state)
            if restored:
                io.iprint("Interrupted; restored data baseline and returned to home.")
            else:
                io.iprint("Interrupted; no data baseline was captured. Returned to home.")
            io.flush_session_log()
