# Structure Catalogs

Catalog files list the crystallographic structures that EWALD can simulate for
training. Each entry should point to a CIF, MCIF, POSCAR, or VASP-style file and
carry enough provenance to make future online datasets auditable.

Minimal fields:

- `structure_id`: stable id used in manifests.
- `name`: readable structure name.
- `path`: path to the structure file, relative to the catalog file.
- `family`: broad material family, such as hybrid perovskite or polymer.
- `phase_class`: finer class, such as 2D layered, 3D cubic, or PbI2 solvate.
- `source` and `license`: provenance for redistribution decisions.

Do not copy collaborator code or datasets into this directory unless the license
and redistribution permissions are clear. Use local paths for private structures
and let cluster staging copy only the files needed for a run.

