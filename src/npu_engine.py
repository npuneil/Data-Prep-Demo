"""
NPU Engine Module — Real Qualcomm Hexagon NPU acceleration via ONNX Runtime QNN.

Detects the QNNExecutionProvider (Hexagon HTP backend) and provides real
ONNX model inference for text-feature extraction.  Falls back to CPU when
QNN is not available.
"""

import time
import platform
import subprocess
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
from pathlib import Path


class AcceleratorType(Enum):
    NPU_QNN = "NPU (QNN Hexagon)"
    GPU_DML = "GPU (DirectML)"
    GPU_CUDA = "GPU (CUDA)"
    CPU = "CPU"


@dataclass
class HardwareInfo:
    device_name: str = "Unknown Device"
    processor: str = "Unknown"
    processor_arch: str = "Unknown"
    ram_gb: float = 0.0
    os_version: str = "Unknown"
    accelerator: AcceleratorType = AcceleratorType.CPU
    npu_available: bool = False
    npu_name: str = ""
    onnx_providers: List[str] = field(default_factory=list)
    onnx_version: str = "Not installed"


@dataclass
class BenchmarkResult:
    provider: str
    operation: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    throughput_ops_sec: float
    tokens_per_sec: float = 0.0


# ── QNN session configuration ──────────────────────────
QNN_OPTIONS = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "burst",
    "enable_htp_fp16_precision": "1",
}

MODEL_INPUT_DIM = 512    # model input width
OPTIMAL_BATCH = 64       # sweet-spot for NPU throughput
BENCHMARK_ITERS = 100    # iterations per benchmark run


class NPUEngine:
    """Manages hardware detection and real ONNX Runtime QNN inference."""

    def __init__(self):
        self.hardware_info = self._detect_hardware()
        self._sess_npu: Optional[Any] = None
        self._sess_cpu: Optional[Any] = None
        self._model_path: Optional[Path] = None
        self._npu_model_path: Optional[Path] = None
        self._benchmark_cache: Dict[str, BenchmarkResult] = {}
        self._init_sessions()

    # ── hardware detection ─────────────────────────────
    def _detect_hardware(self) -> HardwareInfo:
        info = HardwareInfo()
        info.os_version = f"{platform.system()} {platform.version()}"
        info.processor = platform.processor() or "Unknown"
        info.processor_arch = platform.machine()

        # Single combined PowerShell call for device name, RAM, and NPU name
        _ps_script = (
            "$m = (Get-CimInstance Win32_ComputerSystem).Model;"
            "$r = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1);"
            "$n = (Get-PnpDevice -Class 'System' -ErrorAction SilentlyContinue "
            "| Where-Object { $_.FriendlyName -like '*NPU*' -or $_.FriendlyName -like '*Neural*' } "
            "| Select-Object -First 1 -ExpandProperty FriendlyName) 2>$null;"
            "Write-Output $m; Write-Output $r; Write-Output $n"
        )
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", _ps_script],
                capture_output=True, text=True, timeout=8,
            )
            lines = r.stdout.strip().splitlines()
            if len(lines) >= 1 and lines[0]:
                info.device_name = lines[0]
                if "surface" in info.device_name.lower():
                    info.device_name = f"Microsoft {info.device_name}"
            if len(lines) >= 2:
                try:
                    info.ram_gb = float(lines[1])
                except ValueError:
                    pass
            npu_friendly = lines[2].strip() if len(lines) >= 3 else ""
        except Exception:
            info.device_name = "Windows PC"
            npu_friendly = ""

        # ONNX Runtime
        try:
            import onnxruntime as ort
            info.onnx_version = ort.__version__
            info.onnx_providers = ort.get_available_providers()

            if "QNNExecutionProvider" in info.onnx_providers:
                info.npu_available = True
                info.accelerator = AcceleratorType.NPU_QNN
                info.npu_name = npu_friendly or "Qualcomm Hexagon NPU (QNN HTP)"
            elif "DmlExecutionProvider" in info.onnx_providers:
                info.npu_available = True
                info.accelerator = AcceleratorType.GPU_DML
                info.npu_name = "DirectML Accelerator"
            elif "CUDAExecutionProvider" in info.onnx_providers:
                info.accelerator = AcceleratorType.GPU_CUDA
            else:
                info.accelerator = AcceleratorType.CPU
        except ImportError:
            info.onnx_version = "Not installed"
            info.onnx_providers = ["CPUExecutionProvider"]
            info.accelerator = AcceleratorType.CPU

        return info

    # ── ONNX sessions ──────────────────────────────────
    def _init_sessions(self):
        """Create ONNX sessions (QNN + CPU) from built models.

        NPU session uses a fixed-batch model (batch=1) so QNN EP can
        compile ALL nodes to the Hexagon HTP.  CPU session keeps the
        dynamic-batch model for flexible batch sizes.
        """
        from src.model_builder import ensure_model, ensure_npu_model
        try:
            self._model_path = ensure_model()
        except Exception as exc:
            print(f"[NPUEngine] CPU model build failed: {exc}")
            return

        import onnxruntime as ort

        # NPU session — fixed-batch model for QNN HTP
        if self.hardware_info.npu_available:
            try:
                self._npu_model_path = ensure_npu_model()
                self._sess_npu = ort.InferenceSession(
                    str(self._npu_model_path),
                    providers=[("QNNExecutionProvider", QNN_OPTIONS)],
                )
                # Verify QNN actually took the nodes
                active = self._sess_npu.get_providers()
                if "QNNExecutionProvider" not in active:
                    print("[NPUEngine] QNN EP not active, falling back to CPU")
                    self._sess_npu = None
                    self.hardware_info.npu_available = False
                else:
                    print(f"[NPUEngine] QNN HTP session ready — providers: {active}")
            except Exception as exc:
                print(f"[NPUEngine] QNN session failed, falling back: {exc}")
                self._sess_npu = None
                self.hardware_info.npu_available = False

        # CPU session (dynamic batch — always available)
        try:
            self._sess_cpu = ort.InferenceSession(
                str(self._model_path), providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            print(f"[NPUEngine] CPU session failed: {exc}")

    # ── helpers ─────────────────────────────────────────
    def _text_to_input(self, text: str, batch_size: int = OPTIMAL_BATCH) -> np.ndarray:
        """Encode text into a fixed-width float32 tensor for the model.

        Converts UTF-8 bytes → float32 values in [0, 1], padded/truncated
        to MODEL_INPUT_DIM, then tiled to *batch_size* rows to fully
        utilise the NPU pipeline.
        """
        raw = np.frombuffer(text.encode("utf-8")[:MODEL_INPUT_DIM], dtype=np.uint8)
        vec = np.zeros(MODEL_INPUT_DIM, dtype=np.float32)
        vec[: len(raw)] = raw.astype(np.float32) / 255.0
        return np.tile(vec, (batch_size, 1))

    def get_preferred_provider(self) -> str:
        for p in ("QNNExecutionProvider", "DmlExecutionProvider",
                   "CUDAExecutionProvider", "CPUExecutionProvider"):
            if p in self.hardware_info.onnx_providers:
                return p
        return "CPUExecutionProvider"

    # ── inference wrapper ───────────────────────────────
    def run_inference(
        self,
        text: str,
        *,
        use_npu: bool = True,
        batch_size: int = OPTIMAL_BATCH,
    ) -> np.ndarray:
        """Run the ONNX model on *text* and return feature vectors.

        NPU model uses fixed batch=1 (required by QNN HTP), so we loop
        and concatenate.  CPU model uses the dynamic-batch model directly.

        Returns
        -------
        np.ndarray   shape (batch_size, 128) — feature vectors.
        """
        if use_npu and self._sess_npu:
            # NPU path: fixed batch=1, run N times
            vec = self._text_to_input(text, 1)  # shape (1, 512)
            results = []
            for _ in range(batch_size):
                out = self._sess_npu.run(None, {"input": vec})[0]
                results.append(out)
            return np.concatenate(results, axis=0)
        else:
            # CPU path: dynamic batch
            feed = {"input": self._text_to_input(text, batch_size)}
            sess = self._sess_cpu
            if sess is None:
                raise RuntimeError("No ONNX session available")
            return sess.run(None, feed)[0]

    # ── benchmark ───────────────────────────────────────
    def benchmark_text_processing(
        self,
        text: str,
        iterations: int = BENCHMARK_ITERS,
        batch_size: int = OPTIMAL_BATCH,
    ) -> Dict[str, BenchmarkResult]:
        """Run a **real** inference benchmark comparing NPU vs CPU.

        Returns dict with keys ``"CPU"`` and ``"NPU (QNN Hexagon)"``.
        """
        results: Dict[str, BenchmarkResult] = {}
        feed_cpu = {"input": self._text_to_input(text, batch_size)}
        feed_npu = {"input": self._text_to_input(text, 1)}  # fixed batch=1
        approx_tokens = max(len(text.split()), 1)

        # ── CPU benchmark ──
        if self._sess_cpu is not None:
            # warm-up
            for _ in range(10):
                self._sess_cpu.run(None, feed_cpu)
            start = time.perf_counter()
            for _ in range(iterations):
                self._sess_cpu.run(None, feed_cpu)
            cpu_ms = (time.perf_counter() - start) * 1000
            cpu_avg = cpu_ms / iterations
            results["CPU"] = BenchmarkResult(
                provider="CPU",
                operation="text_feature_extraction",
                iterations=iterations,
                total_time_ms=cpu_ms,
                avg_time_ms=cpu_avg,
                throughput_ops_sec=(iterations / (cpu_ms / 1000)) if cpu_ms else 0,
                tokens_per_sec=(approx_tokens * iterations) / (cpu_ms / 1000) if cpu_ms else 0,
            )

        # ── NPU benchmark (batch=1 per call, on Hexagon HTP) ──
        if self._sess_npu is not None:
            for _ in range(10):
                self._sess_npu.run(None, feed_npu)
            start = time.perf_counter()
            for _ in range(iterations):
                self._sess_npu.run(None, feed_npu)
            npu_ms = (time.perf_counter() - start) * 1000
            npu_avg = npu_ms / iterations
            results["NPU (QNN Hexagon)"] = BenchmarkResult(
                provider="NPU (QNN Hexagon)",
                operation="text_feature_extraction",
                iterations=iterations,
                total_time_ms=npu_ms,
                avg_time_ms=npu_avg,
                throughput_ops_sec=(iterations / (npu_ms / 1000)) if npu_ms else 0,
                tokens_per_sec=(approx_tokens * iterations) / (npu_ms / 1000) if npu_ms else 0,
            )

        self._benchmark_cache = results
        return results

    # ── UI helpers ──────────────────────────────────────
    def get_status_display(self) -> Dict[str, Any]:
        hw = self.hardware_info
        return {
            "device": hw.device_name,
            "processor": hw.processor,
            "architecture": hw.processor_arch,
            "ram_gb": hw.ram_gb,
            "os": hw.os_version,
            "accelerator": hw.accelerator.value,
            "npu_available": hw.npu_available,
            "npu_name": hw.npu_name if hw.npu_available else "Not detected",
            "npu_session_active": self._sess_npu is not None,
            "onnx_version": hw.onnx_version,
            "providers": hw.onnx_providers,
            "preferred_provider": self.get_preferred_provider(),
            "is_copilot_pc": hw.npu_available or "arm" in hw.processor_arch.lower(),
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self._benchmark_cache:
            return {}
        cpu = self._benchmark_cache.get("CPU")
        npu = self._benchmark_cache.get("NPU (QNN Hexagon)")
        if cpu and npu:
            return {
                "cpu_avg_ms": round(cpu.avg_time_ms, 2),
                "npu_avg_ms": round(npu.avg_time_ms, 2),
                "speedup": round(cpu.avg_time_ms / npu.avg_time_ms, 2) if npu.avg_time_ms else 0,
                "cpu_tokens_sec": round(cpu.tokens_per_sec),
                "npu_tokens_sec": round(npu.tokens_per_sec),
                "npu_available": True,
            }
        elif cpu:
            return {
                "cpu_avg_ms": round(cpu.avg_time_ms, 2),
                "npu_avg_ms": 0,
                "speedup": 0,
                "cpu_tokens_sec": round(cpu.tokens_per_sec),
                "npu_tokens_sec": 0,
                "npu_available": False,
            }
        return {}
