import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_tf32(enabled: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled


def benchmark_once(B: int, K: int, N: int, use_tf32: bool, iters: int = 100) -> float:
    if device != "cuda":
        raise RuntimeError("This benchmark must run on a CUDA GPU.")

    set_tf32(use_tf32)

    x = torch.randn(B, K, device=device, dtype=torch.float32)
    w = torch.randn(K, N, device=device, dtype=torch.float32)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = x @ w
    end.record()

    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / iters
    return avg_ms


def main() -> None:
    B, K, N = 128, 512, 512

    baseline_ms = benchmark_once(B, K, N, use_tf32=False)
    tf32_ms = benchmark_once(B, K, N, use_tf32=True)

    print(f"Baseline avg latency (ms): {baseline_ms:.4f}")
    print(f"TF32 avg latency (ms): {tf32_ms:.4f}")


if __name__ == "__main__":
    main()