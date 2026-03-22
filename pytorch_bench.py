import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_once(B: int, K: int, N: int) -> torch.Tensor:
    x = torch.randn(B, K, device=device, dtype=torch.float32)
    w = torch.randn(K, N, device=device, dtype=torch.float32)
    y = x @ w
    return y


def main() -> None:
    B, K, N = 128, 512, 512
    y = run_once(B, K, N)
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()