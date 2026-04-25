# plot_results.py
import numpy as np
import matplotlib.pyplot as plt

# ── Gait spectrum plot ────────────────────────────────────────────────────────
data = np.fromfile("gait_data.bin", dtype=np.float32).reshape(12, 4096)
Fs = 1000.0
joint_names = [f"{leg}-{j}" for leg in ["FL","FR","BL","BR"]
                             for j in ["hip","thigh","knee"]]

fig, axes = plt.subplots(3, 4, figsize=(16, 8))
fig.suptitle("Quadruped Gait — FFT Frequency Spectrum (12 Joints)", fontsize=14)
for i, ax in enumerate(axes.flat):
    freqs = np.fft.rfftfreq(4096, d=1/Fs)
    mag   = np.abs(np.fft.rfft(data[i]))
    ax.plot(freqs[:100], mag[:100], color='steelblue', linewidth=1)
    ax.axvline(2.0, color='red', linestyle='--', linewidth=1, label='2 Hz gait')
    ax.axvline(4.0, color='orange', linestyle='--', linewidth=0.8, label='harmonic')
    ax.set_title(joint_names[i], fontsize=9)
    ax.set_xlabel("Freq (Hz)", fontsize=8)
    ax.set_ylabel("Magnitude", fontsize=8)
    ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig("gait_spectrum.png", dpi=150)
print("Saved gait_spectrum.png")

# ── CPU vs GPU speedup bar chart ──────────────────────────────────────────────
import csv
with open("results.txt") as f:
    reader = csv.DictReader(f)
    row = next(reader)
    cpu_ms = float(row["cpu_ms"])
    gpu_ms = float(row["gpu_ms"])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("CPU vs GPU — Gait FFT Performance", fontsize=13)

axes[0].bar(["CPU\n(Sequential)", "GPU\n(cuFFT Batch)"],
            [cpu_ms, gpu_ms],
            color=["#e74c3c", "#2ecc71"], width=0.5)
axes[0].set_ylabel("Time (ms)")
axes[0].set_title("Execution Time (12 joints, 4096 samples)")
for i, v in enumerate([cpu_ms, gpu_ms]):
    axes[0].text(i, v + 0.5, f"{v:.2f} ms", ha='center', fontsize=10)

speedup = cpu_ms / gpu_ms
axes[1].bar(["Speedup"], [speedup], color="#3498db", width=0.3)
axes[1].set_ylabel("Speedup (×)")
axes[1].set_title(f"GPU Speedup: {speedup:.2f}×")
axes[1].text(0, speedup + 0.1, f"{speedup:.2f}×", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig("speedup.png", dpi=150)
print("Saved speedup.png")