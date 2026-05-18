# Developer Notes

## Repo structure

- `src/ewald/` is the active package.
- `src/ewald/ui/` contains Qt windows and widgets.
- `src/ewald/data/`, `src/ewald/io/`, `src/ewald/processing/` contain core state, I/O,
  and workflow logic.
- `.github/workflows/` contains CI and docs deployment.

## How to run tests

From an activated EWALD environment:

```bash
pytest
```

## How to build docs

```bash
python -m pip install -r docs/requirements.txt
mkdocs build
mkdocs serve
```

## How to add a new UI tool

1. Add the new tool module under `src/ewald/ui/`.
2. Wire launch/close behavior into `src/ewald/ui/main_window.py`.
3. Register signals so project state changes flow back into existing views.
4. Add docs coverage (overview + tutorial if user-facing).
5. Add regression test coverage where feasible.

## How to add a new documentation page

1. Follow the dedicated contributor guide:
   [`add-new-documentation-page`](add-new-documentation-page.md).
1. Add a `.md` file under `docs/`.
2. Add a short parent section in `mkdocs.yml`.
3. Include:
   - purpose
   - prerequisites
   - expected outcomes
   - known limitations
4. Include planned/experimental markers where needed.

## Documentation style rules

- Follow [style guide](../guides/style-guide.md).
- Keep section names and feature names consistent with active UI names.
