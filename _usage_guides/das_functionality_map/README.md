# DAS Functionality Map

For PowerPoint, use the ready-made `das_workflow.png` (5480 x 1940) or `das_workflow.svg` (vector).

Open `das_functionality.html` in a browser. It is self-contained and works
offline; there are no downloads, installs, servers or DAS imports.

- **Data workflow:** a slide-friendly view of inputs, masks, triplets and hubs.
- **All menu hubs:** actual controller Startup/Home entries and their branches.
- **Focused views:** preparation, analysis, visualization or data editing.
- **Recipes / repeat / catlist:** a separate explanation of execution context.
- Click a node for source references; click branches to collapse/expand.
- Set depth and show/hide prompt options to control granularity.
- Export **SVG** for a sharp PowerPoint image, or a high-resolution **PNG**.
  Exports contain only the displayed diagram. PNGs are capped at 32 megapixels.

## Keep Truth And Appearance Separate

`functionality.json` is the generated, source-linked map. It distinguishes
navigation, data-flow, option and context relationships. Every registered
prompt remains in the JSON even when hidden in the visual.

`annotations.json` is the small editable supplement for reviewed data-flow and
context relationships plus presentation layouts. `viewer.html` controls style
and rendering. Changes to either do not modify DAS execution code.

Regenerate from this directory with an existing Python installation:

```powershell
python build_map.py
```

The script inspects Python syntax. It does not import DAS or run analyses.
It writes `functionality.json` and embeds a copy in `das_functionality.html`.
To view hand-edited truth JSON immediately, use **Load edited JSON** in the HTML.
Regeneration replaces the generated JSON, so put lasting semantic edits in
`annotations.json` or update extraction in `build_map.py`.

## Scope

This maps current controller menus, recipe registry actions/options, active
IFA data-editing/scaling submenus, and reviewed registration/data/context
connections. It is not a complete repository call graph or proof that each
feature is operational. Cellpose's stub and standalone training are labeled.

The script inventory lists Python files not directly linked to entry points.
Unlinked does **not** mean unused. Helpers, tests and standalone scripts are
not forced into the user menu. Add reviewed links as useful rather than
inventing navigation based on imports.

The recipe registry's `home` is an action catalogue, not the visible home menu;
the generator uses the controller's actual lists to determine Startup and Home
placement. Scientific data-flow arrows are separate reviewed relationships.
