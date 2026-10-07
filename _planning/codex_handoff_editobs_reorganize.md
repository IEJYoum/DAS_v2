# Handoff: "reorganize all obs" in editObs silently skips the manual re-pick

## Symptom (reported by user)

From the "edit observations" menu action, the expected flow is:
1. Prompt: `re-organize all obs? (y) :`
2. On "y": merge the full (df, obs, dfxy) triplet back into one table and
   re-run the manual column-picking flow (coordinate columns, then
   observation columns) so the user can re-pick what goes where.

**What actually happens:** no re-pick prompt is shown. Execution goes
straight into a "split" step without asking the user to choose columns.
User's words: *"It doesn't prompt me to re-pick, it goes straight into
split. So the merge and split code isn't firing."*

This is **not** a stale-process/caching issue — user confirmed the
`editObs`/reorganize code itself is ~4 years old and untouched recently.
What *did* recently change is `data_extraction/tabular_ingest.py` (a
separate, newer ingest path) — but `editObs` does not call into that
module at all (verified: no `tabular_ingest`/`ingest_sources`/
`image_conventions` import anywhere in `IFA.py`). So if the two are
related, the connection isn't a direct import — see the lead below.

## Code path (entry point user is using)

`IFA.py`, reached via `controler.py: main_menu -> "data editing" ->
legacy_data_editing_menu -> _run_legacy_data_editing ->
module.loadingMenu(...)` (the legacy `IFA.py` module), option index 2
("edit observations").

```python
# IFA.py:681 loadingMenu — op[2]/fn[2] = "edit observations" / editObs
def loadingMenu(df=9,obs=9,dfxy=9):
    ...
    op = ["save","drop cells with less than N% of data",
          "edit observations","drop columns based on key string","import biomarkers","import observations",
          ...]
    fn = [save,dropCells,editObs,dropCols,importBiom, ...]
    df,obs,dfxy,*log = menu(op,fn,df,obs,dfxy)
```

```python
# IFA.py:2032
def editObs(df,obs,dfxy):
    """Entry point for observation restructuring (optional full re-split + unpackObs)."""
    ch = logInput("re-organize all obs? (y) :")
    if ch == "1" or ch == "y":
        df,obs,dfxy=makeObs(pd.concat([df,obs,dfxy],axis=1))
    obs=unpackObs(obs)
    return(df,obs,dfxy)
```

```python
# IFA.py:2077
def makeObs(df):
    """Split a merged table into measurement df, observation obs, and coordinate dfxy tables."""
    if _koei_runmode():
        auto = _koei_make_obs(df)
        if auto is not None:
            df,obs,dfxy = auto
            ...
            return(df,obs,dfxy)
    df,dfxy = splitDF(df,"X and Y coordinate columns")      # <- the manual "re-pick" the user expects
    if input('swap X and Y? (y)') == 'y':
        ...
    df,obs = splitDF(df,"Observation columns")              # <- and this one
    ...
    return(df,obs,dfxy)
```

`unpackObs` (IFA.py:1810) is the function whose *first* prompt is
literally `"split column number:"` — this is almost certainly the
"split" the user says it jumps straight into, i.e. **the merge+makeObs
branch in `editObs` appears not to run (or runs invisibly with no
prompts), and the flow lands directly in `unpackObs`'s own split-by-
delimiter prompt.**

## What's been ruled out

- `editObs` is defined exactly once in the repo (`grep -rn "^def editObs" DAS_v2` → one hit).
- `loadingMenu`'s `op`/`fn` lists line up correctly (index 2 is `editObs` in both).
- The generic dispatcher `menu()` (IFA.py:633) calls `functions[ch](df,obs,dfxy)` with no off-by-one.
- `load_legacy_ifa5()` in `controler.py` does cache the module per-process
  (`global _LEGACY_IFA5`), but the user has ruled out staleness as the cause.
- Commits that touched both `IFA.py` and `tabular_ingest.py`
  (`a6282bf` "index reconstruction patch...", `02390ac` "ingest modded a
  bit for prefix...") do **not** touch `editObs`/`makeObs`/`unpackObs`/
  `loadingMenu` — confirmed via `git show <hash> -- IFA.py`.

## Strongest lead found so far

Commit `b267414` ("import obs", `git show b267414 -- IFA.py`) changed
`_koei_obs_column` — the predicate `_koei_make_obs` uses to
auto-classify columns as observations when `makeObs` takes its
auto-detect branch:

```diff
 def _koei_obs_column(df, col):
     if col in KOEI_XY_COLUMNS: return False
     if col in KOEI_OBS_COLUMNS: return True
     ser = df.loc[:, col]
-    return (
+    if (
         pd.api.types.is_bool_dtype(ser)
         or pd.api.types.is_string_dtype(ser)
         or pd.api.types.is_object_dtype(ser)
-    )
+    ):
+        return True
+    low = str(col).strip().lower()
+    if low.endswith("_func") or low.endswith("_functional"):
+        return True
+    numeric = pd.to_numeric(ser, errors="coerce").dropna().unique()
+    return len(numeric) == 2 and set(numeric).issubset({0,1})
```

And critically, `RUNMODE` **defaults to `"Koei"` at module scope**:

```python
# IFA.py:158-160
RUNMODE = "Koei"
KOEI_XY_COLUMNS = ["DAPI_X", "DAPI_Y"]
KOEI_OBS_COLUMNS = ["cellid", "patient", "slide", "slide_scene", "scene"]
```

(overridden only at IFA.py:534, inside `main(..., runmode=None)` — which
is a *different* entry point than the `loadingMenu(df,obs,dfxy)` call
used by the legacy-editing menu path, so it's worth checking whether
anything sets `RUNMODE` away from its "Koei" default on the path the
user is actually exercising).

`_koei_make_obs` only returns `None` (falling through to the manual
`splitDF` prompts) when `DAPI_X`/`DAPI_Y` are missing from the merged
table. Since `dfxy` is always merged in by `editObs` via
`pd.concat([df,obs,dfxy],axis=1)`, those columns should normally be
present — meaning if `RUNMODE` is "Koei" (the default), `makeObs` likely
takes the **auto-detect branch every time** and never reaches
`splitDF` at all, silently picking obs columns via `_koei_obs_column`'s
(now broader) rule instead of asking the user. That would explain "no
re-pick prompt" from `editObs`'s perspective. It does *not* on its own
explain why the user still lands in a "split" prompt afterward — that
part still needs to be traced live (most likely `unpackObs`'s own
`"split column number:"` prompt immediately following the
now-silent/auto `makeObs` call — see `editObs` above, `unpackObs` always
runs unconditionally after the `if` block).

## Suggested next steps for Codex

1. Instrument (or step through) `editObs` -> `makeObs` on the user's
   actual data to confirm: does `_koei_runmode()` return `True` on the
   live path the user exercises (`loadingMenu` from
   `legacy_data_editing_menu`)? Print/log `RUNMODE` at entry to
   `makeObs`.
2. If `_koei_runmode()` is `True`: confirm `_koei_make_obs` is
   succeeding (not returning `None`) and silently consuming the
   "re-pick" step. Check whether the broadened `_koei_obs_column`
   predicate from `b267414` is now sweeping columns into `obs` that the
   user doesn't want there (the `_func`/`_functional` suffix rule and
   the binary-0/1-numeric rule are both new).
3. Decide the intended fix with the user: should `editObs`'s explicit
   "re-organize all obs?" confirmation bypass the Koei auto-detect
   entirely (since the user is explicitly asking to manually re-pick),
   e.g. by calling `splitDF` directly instead of going through
   `makeObs`'s `_koei_runmode()` gate? Or should `_koei_runmode()`
   simply not apply inside this manually-triggered reorganize flow?
4. Separately worth asking the user: confirm on which entry point /
   RUNMODE value they're actually running (is `RUNMODE` ever set to
   something other than "Koei" anywhere in their session?).

## Relevant files

- `DAS_v2/IFA.py` — `loadingMenu` (681), `editObs` (2032), `makeObs`
  (2077), `_koei_runmode`/`_koei_obs_column`/`_koei_make_obs`
  (2040-2075), `unpackObs` (1810), `splitDF` (2164), `RUNMODE`/
  `KOEI_XY_COLUMNS`/`KOEI_OBS_COLUMNS` (158-160).
- `DAS_v2/controler.py` — `_run_legacy_data_editing` (2654),
  `legacy_data_editing_menu` (1931), `load_legacy_ifa5` (2052, module
  caching).
- Commit `b267414` ("import obs") — changed `_koei_obs_column` and
  `loadingMenu`'s op/fn lists (index 5 only; index 2/`editObs` itself
  unchanged).
