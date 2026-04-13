"""NPU vs CPU benchmark — QNN Hexagon NPU on Snapdragon X Elite."""
import onnxruntime as ort
import numpy as np
import time

model_path = "models/text_features.onnx"

# QNN session options for NPU
qnn_opts = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "burst",
    "enable_htp_fp16_precision": "1",
}

print("Creating QNN HTP (NPU) session...")
sess_npu = ort.InferenceSession(
    model_path, providers=[("QNNExecutionProvider", qnn_opts)]
)
print(f"  Active providers: {sess_npu.get_providers()}")

print("Creating CPU session...")
sess_cpu = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
print(f"  Active providers: {sess_cpu.get_providers()}")

# Warm-up
dummy = np.random.randn(1, 512).astype(np.float32)
for _ in range(10):
    sess_npu.run(None, {"input": dummy})
    sess_cpu.run(None, {"input": dummy})
print("Warm-up done.\n")

# Benchmark at different batch sizes
for batch_size in [1, 8, 32, 64, 128]:
    batch = np.random.randn(batch_size, 512).astype(np.float32)
    iters = 200

    # CPU
    start = time.perf_counter()
    for _ in range(iters):
        sess_cpu.run(None, {"input": batch})
    cpu_ms = (time.perf_counter() - start) * 1000

    # NPU
    start = time.perf_counter()
    for _ in range(iters):
        sess_npu.run(None, {"input": batch})
    npu_ms = (time.perf_counter() - start) * 1000

    speedup = cpu_ms / npu_ms if npu_ms > 0 else 0
    print(f"batch={batch_size:>3d}  CPU: {cpu_ms:>8.1f}ms  NPU: {npu_ms:>8.1f}ms  Speedup: {speedup:.2f}x")
