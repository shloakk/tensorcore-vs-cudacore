import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def main() -> None:
    pytorch_csv = Path("results_pytorch.csv")
    cublas_csv = Path("results_cublas.csv")

    if not pytorch_csv.exists() and not cublas_csv.exists():
        raise FileNotFoundError(
            "No input CSV found. Run pytorch_bench.py and/or cuda_bench first."
        )

    frames = {}
    if pytorch_csv.exists():
        frames["PyTorch"] = pd.read_csv(pytorch_csv)
    if cublas_csv.exists():
        frames["cuBLAS"] = pd.read_csv(cublas_csv)

    first_df = next(iter(frames.values()))
    x = list(range(len(first_df)))
    labels = [
        f"{b}x{k}x{n}"
        for b, k, n in zip(first_df["B"], first_df["K"], first_df["N"])
    ]

    plt.figure(figsize=(11, 6))
    for name, df in frames.items():
        plt.plot(x, df["baseline_ms"], marker="o", label=f"{name} baseline FP32")
        plt.plot(x, df["tf32_ms"], marker="s", label=f"{name} TF32 path")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Latency (ms)")
    plt.title("Forward Pass Latency: Baseline vs TF32")
    plt.legend()
    plt.tight_layout()
    plt.savefig("latency_comparison.png")

    plt.figure(figsize=(11, 6))
    for name, df in frames.items():
        plt.plot(
            x,
            df["baseline_tflops"],
            marker="o",
            label=f"{name} baseline FP32",
        )
        plt.plot(x, df["tf32_tflops"], marker="s", label=f"{name} TF32 path")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Throughput (TFLOPS)")
    plt.title("Forward Pass Throughput: Baseline vs TF32")
    plt.legend()
    plt.tight_layout()
    plt.savefig("throughput_comparison.png")

    plt.figure(figsize=(11, 6))
    for name, df in frames.items():
        plt.plot(x, df["speedup"], marker="o", label=f"{name} speedup")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Speedup (baseline / TF32)")
    plt.title("Tensor Core Speedup by Backend")
    plt.legend()
    plt.tight_layout()
    plt.savefig("speedup_comparison.png")

    print(
        "Saved latency_comparison.png, throughput_comparison.png, and speedup_comparison.png"
    )


if __name__ == "__main__":
    main()