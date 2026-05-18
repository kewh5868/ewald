# Tutorial: Using film optics inputs

This workflow controls low-angle optical corrections in EWALD.

## 1) Open film optics controls

1. Open **Apply Image Corrections**.
2. Find the film optics section in the panel.
3. Enter stoichiometry text and density value.

## 2) Estimate optical constants

- Use the estimate action with composition and density input.
- Review estimated critical angle and refractive constant values.
- Confirm values are applied to the active correction state before saving project state.

## 3) Use structure reference files

EWALD accepts:

- `*.cif`
- `*.mcif`
- `POSCAR*`
- `CONTCAR*`
- `*.vasp`

Load a valid structure reference to support composition- or phase-aware optics behavior.

## 4) Save and reuse values

- Save named film memory entries for repeated sessions.
- Clear or replace memory entries when the sample changes.

## 5) Confirm runtime behavior

- Verify correction outputs use the active film values.
- Save project and reopen later to confirm persistent memory/reload.

![Film optics placeholder](../assets/placeholders/film-optics.svg)

## Planned and current support

- Basic film optics workflow and memory items are implemented.
- Advanced multilayer stack modeling is **Planned**.
