---
craft_id: sage_docking_v1
version: 0.2.0
project: sage
license: Apache-2.0
tags: ['status:trained', 'project:sage', 'provenance:engine_truth', 'release_metric:affinity_R2_vs_mean']
library_name: sage-models
---

# sage_docking_v1

**Project:** sage · **Version:** 0.2.0 · **License:** Apache-2.0
**Produced by:** [Intel — Universal ML Inference Engine](https://huntingtonai.com)

> Sage Drug-Protein Docking Predictor Craft (4.4)

> ✅ **Engine-truth data (`engine_truth`).** Targets are direct real simulation-engine output; the headline is the **`affinity_R2_vs_mean`** metric.
>
> _Eval note:_ Re-registered to current checkpoint (120-epoch retrain, 48 by-run split). binding_affinity + energy_decomposition = engine-truth sage ForceCalculator interaction energy; affinity R2 0.9984 vs mean, gate PASS. Pose head self-supervised.

## Model description

`sage_docking_v1` is a **surrogate model** ("craft") used by the AHL Intel inference
engine to replace expensive numerical simulation with fast neural inference.
Output kind: **fields** (0 outputs). Sage Drug-Protein Docking Predictor Craft (4.4)

- **Status:** trained
- **Type:** neural inference surrogate (physics-trained)
- **Parameters:** 2,012,738
- **Weights format:** `model.safetensors` (float32)
- **Device family:** `—`

## Intended use

Drop-in inference surrogate for sage. You provide a physical **device
specification** (geometry, materials, doping, operating point); the package
returns the physical output channels below. The device→model featurization and
de-normalization are handled inside the package. Not intended for regimes
outside its training domain (below).

## Inputs & outputs

**Input — physical device specification** (`input.kind: device_spec`):

Provide a physical device/scene specification; see the project's model docs.

The internal encoding (how the device becomes model inputs) and the normalization are handled **inside the package** — you never build feature vectors by hand.

**Output — physical fields:**

| | |
|---|---|
| **Output channels** | 0 |
| **Parameters** | 2,012,738 |

## Training domain / compatibility

| Property | Value |
|---|---|
| Solution type | `—` |
| Physics models | — |
| Temperature | — K |
| Bias range (Vds) | — |
| Tags | `status:trained`, `project:sage`, `provenance:engine_truth`, `release_metric:affinity_R2_vs_mean` |

## Training data

Trained on sage simulation output for the `—` solution regime (physics models: see config). Labels derive from the numerical sage engine / reference closed-forms; the data-generation pipeline is proprietary.

## Limitations & out-of-scope use

- **Do not** use outside the training domain above (solution type, temperature, bias range) — accuracy degrades with no guarantee.
- This is a **surrogate**, not the ground-truth solver: treat outputs as fast approximations; validate against the numerical engine where correctness is critical.
- Output kind is **fields** — consuming code must match the documented I/O contract exactly.


## Evaluation

| Metric | Value |
|---|---|
| **Held-out accuracy** | 0.9984 |
| Best validation loss | 0.0004420215954121611 |

_**Held-out accuracy** is an independent benchmark (`offline_eval` on a held-out set); the loss rows are the model's own recorded train/val losses. See `config.json → metrics` for the machine-readable copy._

## Caveats & recommendations

- An independent **held-out** benchmark is reported (Held-out accuracy, via `offline_eval`); the loss rows remain the model's own recorded train/val losses.
- Verify integrity before use: `shasum -a 256 -c SHA256SUMS` (or `sha256sum -c SHA256SUMS` on Linux). Weights are `safetensors` — safe to load, no pickle.
- This is a **surrogate**, not the ground-truth solver — validate against the numerical engine where correctness is critical.

## Usage

Install the inference package and run it on a physical device spec — it fetches the
weights, builds the model, and returns physical fields (no PyTorch required):

```bash
pip install sage-models
```

```python
from sage_models import load

model = load("sage_docking_v1")           # pulls + verifies weights, builds the compiled pipeline
fields = model.predict(device_spec)  # physical device in -> physical fields out
```

`device_spec` is the physical device specification documented under **Inputs & outputs**
above. Featurization and de-normalization happen inside the package.

## Provenance

- **Author:** sim-train release (sage docking, engine-truth)
- **Trained:** 2026-07-19T12:23:07.304357+00:00
- **Source checksum (runtime .pt):** `cf4ac86ba816cbb70adb8f01c27122209f3245c939b26721be71e3a074478e95`

## Citation

```bibtex
@software{sage_docking_v1_0_2_0,
  title  = {sage_docking_v1},
  author = {AHL / Intel},
  year   = {2026},
  note   = {Version 0.2.0},
  url    = {https://downloads.sagesimulation.com/models/sage_docking_v1/0.2.0/}
}
```

## License

Licensed under the **Apache-2.0** (Apache-2.0) — free to use for
inference; redistribution and reverse engineering of the compiled inference core
are not permitted. See `LICENSE`.
