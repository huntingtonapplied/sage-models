# sage-models

Runnable neural-surrogate inference for the **Sage** biological-simulation platform.
Torch-free (ONNX + `onnxruntime`, CPU).

```python
from sage_models import load

model = load("sage_docking_v1")
out = model.predict(inputs={
    "protein_atom_types": p_types,   # [Np] int atom types
    "protein_positions":  p_pos,     # [Np, 3]
    "protein_edge_index": p_edges,   # [2, Ep]
    "drug_atom_types":    d_types,   # [Nd] int
    "drug_positions":     d_pos,     # [Nd, 3]
    "drug_edge_index":    d_edges,   # [2, Ed]
})
# -> {binding_affinity, binding_pose[6], energy_decomposition[4]}
```

`model.input_names` / `model.output_names` and each craft's `MODEL_CARD.md` document I/O.

## Install
```
pip install sage-models      # numpy + onnxruntime only, no torch
```

## Scope
- **Preview build**: ships `sage_docking_v1`; more crafts follow.
- **Fast surrogates, not the ground-truth solver.**

## License
Closed-source, compiled distribution under the **AHL Model EULA** (see `LICENSE`).
© 2025–2026 Huntington Applied.
