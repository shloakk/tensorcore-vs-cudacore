import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_tf32(enabled: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled


def run_once(B: int, K: int, N: int, use_tf32: bool) -> torch.Tensor:
    set_tf32(use_tf32)
    x = torch.randn(B, K, device=device, dtype=torch.float32)
    w = torch.randn(K, N, device=device, dtype=torch.float32)
    y = x @ w
    return y


def main() -> None:
    B, K, N = 128, 512, 512

    baseline = run_once(B, K, N, use_tf32=False)
    tensor_core = run_once(B, K, N, use_tf32=True)

    print("Ran both benchmark modes:")
    print(f"Baseline shape: {tuple(baseline.shape)}")
    print(f"TF32 shape: {tuple(tensor_core.shape)}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()