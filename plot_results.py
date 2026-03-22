import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    df = pd.read_csv("results_pytorch.csv")

    x = list(range(len(df)))
    labels = [f"{b}x{k}x{n}" for b, k, n in zip(df["B"], df["K"], df["N"])]

    plt.figure(figsize=(10, 6))
    plt.plot(x, df["baseline_ms"], marker="o", label="FP32, TF32 disabled")
    plt.plot(x, df["tf32_ms"], marker="o", label="FP32, TF32 enabled")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Latency (ms)")
    plt.title("Forward Pass Latency Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("latency.png")

    plt.figure(figsize=(10, 6))
    plt.plot(x, df["baseline_tflops"], marker="o", label="FP32, TF32 disabled")
    plt.plot(x, df["tf32_tflops"], marker="o", label="FP32, TF32 enabled")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Throughput (TFLOPS)")
    plt.title("Forward Pass Throughput Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("throughput.png")

    plt.figure(figsize=(10, 6))
    plt.plot(x, df["speedup"], marker="o")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Speedup (baseline / TF32)")
    plt.title("Tensor Core Speedup")
    plt.tight_layout()
    plt.savefig("speedup.png")

    print("Saved latency.png, throughput.png, and speedup.png")


if __name__ == "__main__":
    main()