"""Quick smoke test for the rewritten NPU engine."""
from src.npu_engine import NPUEngine
import time

engine = NPUEngine()
hw = engine.get_status_display()
print(f"NPU available: {hw['npu_available']}")
print(f"NPU name:      {hw['npu_name']}")
print(f"Provider:       {hw['preferred_provider']}")
print(f"Accelerator:    {hw['accelerator']}")
print()

# Test real inference
text = "Vanguard 500 Index Fund VFIAX NAV 502.87 expense ratio 0.04%"
features = engine.run_inference(text, batch_size=64)
print(f"Inference output shape: {features.shape}")
print(f"Mean feature value:     {features.mean():.4f}")
print()

# Quick benchmark
start = time.perf_counter()
results = engine.benchmark_text_processing(text, iterations=50, batch_size=64)
elapsed = time.perf_counter() - start
print(f"Benchmark done in {elapsed:.1f}s")
perf = engine.get_performance_summary()
for k, v in perf.items():
    print(f"  {k}: {v}")
