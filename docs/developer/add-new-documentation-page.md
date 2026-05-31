# Add a New Documentation Page

!!! warning "Documentation notice"
    This documentation was generated with help from a large language model and has not been fully vetted by the developer. Verify critical details against the source code and current application behavior.


Use this short process whenever adding user-facing documentation content.

## 1) Create the markdown file

1. Pick the section folder:
   - `docs/getting-started/`
   - `docs/guides/`
   - `docs/tutorials/`
   - `docs/developer/`
   - `docs/development/`
2. Use kebab-case filenames that match the URL intent:
   - `my-topic.md`
   - `advanced-workflow.md`
3. Start with a short description, then numbered sections.

## 2) Add to `mkdocs.yml`

1. Add a human-readable title and the file path under the correct nav section.
2. Preserve consistent phrasing with existing top-level groups.
3. Keep order from overview to advanced behavior when possible.

## 3) Add required metadata style

Every page should include:

- Concise summary at the top.
- What users need to have done before this page applies.
- Step-by-step procedure.
- Known limitations with explicit state labels:
  - **Implemented**
  - **Experimental**
  - **Planned**
- Screenshot or diagram references where UI visibility is discussed.

## 4) Keep terminology consistent

- Use canonical names:
  - `EWALD`
  - `Data Viewer`
  - `Peak Identification`
  - `Peak Fit`
  - `Structure Analysis`
  - `GIWAXS simulation`
  - `q<sub>xy</sub>` and `q<sub>z</sub>`
- Use units in titles and tables where relevant.

## 5) Cross-link from existing pages

1. Add forward links from overview and related pages.
2. If a tutorial depends on this page, add it to the Tutorials section.
3. Include at least one “next step” link.

## 6) Update and verify

1. Run docs build:
   ```bash
   mkdocs build
   ```
2. Rebuild and review the page with `mkdocs serve`.
3. Commit both the content file and any required image updates.
