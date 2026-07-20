"""
sage-models — runnable inference package (compiled distribution, not source).

Public surface, two lines:

    from sage_models import load
    model = load("<craft_id>")
    out = model.predict(inputs)      # physical inputs in -> physical outputs out

Two input styles, depending on the craft:
  * **Field crafts** (a dense physical scene): pass `field_grids=` — a mapping of physical
    field name -> dense `[Nx,Ny,Nz]` (scalar) or `[Nx,Ny,Nz,C]` (vector) array. The package
    encodes them into the model's graph via a compiled featurizer and reshapes outputs back
    to grids.
  * **Graph / tensor crafts**: pass `inputs=` — a mapping of the model's input names
    (see the model card) -> arrays. Returned as named output arrays.

Design (see MODEL_INFERENCE_PACKAGE_STRATEGY.md):
  * The model forward + any de-normalization are baked into an ONNX graph and run under
    `onnxruntime` — no torch, no PyG, CPU-only.
  * Where a physical->model featurization exists, it ships **compiled** as `_featurizer`
    (an opaque extension module), never as readable source.
  * These are fast surrogates, not the ground-truth solver — validate against the numerical
    engine where correctness is critical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover - dependency is declared in pyproject
    raise ImportError(
        "onnxruntime is required to run inference. Install with `pip install onnxruntime`."
    ) from e

# Optional compiled featurizer (present only for projects with a physical->graph encoder).
try:  # noqa: SIM105
    from . import _featurizer  # type: ignore
except Exception:  # pragma: no cover - absent for direct-input crafts
    _featurizer = None  # type: ignore

_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _ROOT / "models"


def list_models() -> List[str]:
    """Craft ids shipped in this package (those with published weights)."""
    if not _MODELS_DIR.exists():
        return []
    return sorted(d.name for d in _MODELS_DIR.iterdir() if (d / "model.onnx").exists())


_ONNX_DTYPE = {
    "tensor(float)": np.float32, "tensor(double)": np.float32,
    "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": np.bool_,
}


def _cast_to(onnx_type: str, arr) -> np.ndarray:
    """Cast an input to the dtype the ONNX graph declares (int embeddings vs floats)."""
    return np.asarray(arr).astype(_ONNX_DTYPE.get(onnx_type, np.float32))


class Model:
    """A single loaded craft: a physical -> physical surrogate."""

    def __init__(self, craft_id: str):
        cdir = _MODELS_DIR / craft_id
        onnx_path = cdir / "model.onnx"
        if not onnx_path.exists():
            avail = ", ".join(list_models()) or "(none published in this build)"
            raise FileNotFoundError(f"No model weights for '{craft_id}'. Available: {avail}.")
        self.craft_id = craft_id
        self.config: dict = json.loads((cdir / "public_config.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._sess.get_inputs()]
        self._input_types = {i.name: i.type for i in self._sess.get_inputs()}
        self._output_names = [o.name for o in self._sess.get_outputs()]
        self._contract = (
            _featurizer.get_contract(craft_id) if _featurizer is not None else None
        )

    @property
    def input_names(self) -> List[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> List[str]:
        return list(self._output_names)

    def predict(
        self,
        inputs: Optional[Mapping[str, np.ndarray]] = None,
        *,
        field_grids: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run the surrogate. Provide EITHER `field_grids=` (field crafts) OR `inputs=`
        (graph/tensor crafts). Returns a dict of output name -> physical array."""
        if field_grids is not None:
            return self._predict_field_grids(field_grids)
        if inputs is None:
            raise ValueError("pass inputs= (model input arrays) or field_grids= (a physical scene)")
        feeds = {k: _cast_to(self._input_types[k], v) for k, v in inputs.items() if k in self._input_names}
        missing = [n for n in self._input_names if n not in feeds]
        if missing:
            raise ValueError(f"missing model inputs {missing}; expected {self._input_names}")
        out = self._sess.run(None, feeds)
        return {name: out[i] for i, name in enumerate(self._output_names)}

    def _predict_field_grids(self, field_grids: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if _featurizer is None or self._contract is None:
            raise RuntimeError(
                f"'{self.craft_id}' takes model inputs (inputs=), not field_grids — this build "
                "ships no featurizer."
            )
        grids = {k: np.asarray(v, dtype=np.float32) for k, v in field_grids.items()}
        first = next(iter(grids.values()))
        if first.ndim < 3:
            raise ValueError(f"each field grid must be at least 3-D [Nx,Ny,Nz(,C)]; got {first.shape}")
        shape = tuple(first.shape[:3])
        for k, v in grids.items():
            if tuple(v.shape[:3]) != shape:
                raise ValueError(f"field grid '{k}' spatial shape {v.shape[:3]} != {shape}")
        graph = _featurizer.grid_to_graph(grids, self._contract)
        feeds = {
            "node_features": np.asarray(graph["node_features"], dtype=np.float32),
            "edge_index": np.asarray(graph["edge_index"], dtype=np.int64),
            "edge_features": np.asarray(graph["edge_features"], dtype=np.float32),
        }
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        out = self._sess.run(None, feeds)[0]
        n = int(np.prod(shape))
        if out.shape[0] != n:
            raise RuntimeError(f"model returned {out.shape[0]} nodes but grid has {n}")
        channels = (self.config.get("output") or {}).get("channels") or [
            f"c{i}" for i in range(out.shape[1])
        ]
        return {ch: out[:, i].reshape(shape) for i, ch in enumerate(channels) if i < out.shape[1]}


def load(craft_id: str) -> Model:
    """Load a craft by id (e.g. ``load("landslide_terrain_deformation_v1")``)."""
    return Model(craft_id)


__all__ = ["load", "list_models", "Model"]
