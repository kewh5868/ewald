# Peak Family Benchmark Simulations

These deterministic q-space TIFFs exercise Structure Analysis peak-family grouping
against small known-structure patterns. The adjacent `manifest.json` stores axis
ranges, peak coordinates, and expected family memberships.

Regenerate the TIFFs and manifest from the repository root with:

```bash
python scripts/generate_peak_family_benchmark_tiffs.py
```

The cases intentionally include missing-fundamental harmonic series so tests can
catch regressions where grouping only works when the first-order reflection is
visible.
