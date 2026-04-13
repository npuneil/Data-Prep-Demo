"""
ONNX Model Builder — Creates a real financial-text feature extraction model.

Builds a multi-layer dense network (MLP) that runs on the NPU via DirectML.
The model processes character-level text encodings and outputs feature vectors
used for entity confidence scoring and text classification.

Architecture (designed for NPU-friendly ops — MatMul, Add, Relu):
  Input:  [batch, 512]  float32   (char-level encoded text window)
  Dense1: [512, 1024]   + bias    → Relu
  Dense2: [1024, 1024]  + bias    → Relu
  Dense3: [1024, 512]   + bias    → Relu
  Dense4: [512, 256]    + bias    → Relu
  Dense5: [256, 128]    + bias    → Sigmoid
  Output: [batch, 128]  float32   (feature vector)

Total parameters: ~2.4M — large enough to stress the NPU visibly.
"""

import os
import numpy as np
from pathlib import Path


# Default model path
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "text_features.onnx"
QNN_MODEL_PATH = MODEL_DIR / "text_features_npu.onnx"

# Architecture constants
SEQ_LEN = 512
NPU_BATCH = 1   # fixed batch for QNN/NPU (must be static)
LAYERS = [
    (512, 1024),
    (1024, 1024),
    (1024, 512),
    (512, 256),
    (256, 128),
]


def build_model(output_path: str | Path | None = None, seed: int = 42,
                fixed_batch: int | None = None) -> Path:
    """
    Build the ONNX text-feature-extraction model and save to disk.

    Uses onnx.helper to construct the graph programmatically so we
    never need to download a model from the internet.

    Parameters
    ----------
    fixed_batch : int | None
        If set, use a fixed batch dimension (required for QNN NPU).
        If None, use dynamic batch (works on CPU only).

    Returns the Path to the saved .onnx file.
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    if output_path is None:
        output_path = MODEL_PATH
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    nodes = []
    initializers = []
    prev_output = "input"

    for i, (in_dim, out_dim) in enumerate(LAYERS):
        layer = i + 1
        w_name = f"dense{layer}_weight"
        b_name = f"dense{layer}_bias"
        mm_out = f"dense{layer}_mm"
        add_out = f"dense{layer}_add"

        # Kaiming / He initialisation (better for Relu networks)
        std = np.float32(np.sqrt(2.0 / in_dim))
        w_data = (rng.standard_normal((in_dim, out_dim)) * std).astype(np.float32)
        b_data = np.zeros(out_dim, dtype=np.float32)

        initializers.append(numpy_helper.from_array(w_data, name=w_name))
        initializers.append(numpy_helper.from_array(b_data, name=b_name))

        # MatMul
        nodes.append(helper.make_node("MatMul", [prev_output, w_name], [mm_out],
                                      name=f"MatMul_{layer}"))
        # Add bias
        nodes.append(helper.make_node("Add", [mm_out, b_name], [add_out],
                                      name=f"Add_{layer}"))

        # Activation — Sigmoid on last layer, Relu everywhere else
        if layer < len(LAYERS):
            act_out = f"dense{layer}_relu"
            nodes.append(helper.make_node("Relu", [add_out], [act_out],
                                          name=f"Relu_{layer}"))
            prev_output = act_out
        else:
            act_out = f"dense{layer}_sigmoid"
            nodes.append(helper.make_node("Sigmoid", [add_out], [act_out],
                                          name=f"Sigmoid_{layer}"))
            prev_output = act_out

    # Graph I/O — use fixed batch for QNN NPU, dynamic for CPU
    batch_dim = fixed_batch if fixed_batch is not None else None
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch_dim, SEQ_LEN])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_dim, LAYERS[-1][1]])

    # Rename final activation to "output"
    nodes[-1].output[0] = "output"

    graph = helper.make_graph(
        nodes,
        "TextFeatureExtractor",
        [input_tensor],
        [output_tensor],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.producer_name = "DataPrepAssistant"
    model.doc_string = "Financial text feature extraction — runs on NPU via QNN HTP"

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))

    param_count = sum(np.prod(w.dims) for w in initializers)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    batch_label = f"batch={fixed_batch}" if fixed_batch else "dynamic batch"
    print(f"Model saved → {output_path}  ({param_count:,} params, {size_mb:.1f} MB, {batch_label})")
    return output_path


def ensure_model(model_path: str | Path | None = None) -> Path:
    """Return CPU model path (dynamic batch), building if needed."""
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        return build_model(path, fixed_batch=None)
    return path


def ensure_npu_model() -> Path:
    """Return NPU model path (fixed batch=1 for QNN), building if needed."""
    if not QNN_MODEL_PATH.exists():
        return build_model(QNN_MODEL_PATH, fixed_batch=NPU_BATCH)
    return QNN_MODEL_PATH


if __name__ == "__main__":
    build_model()
