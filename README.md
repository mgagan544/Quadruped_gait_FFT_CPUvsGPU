# Quadruped FFT: CPU vs GPU

This project demonstrates **data parallelism** using FFT-based gait frequency analysis for a quadruped robot. It compares a **sequential CPU implementation** against a **GPU implementation using NVIDIA cuFFT**, then reports the execution speedup for the same workload.[1][2]

The main idea is simple: each joint signal is independent, so the CPU processes them one by one, while the GPU processes them together as a batch using `cufftPlanMany`.[1][3]

## Repository structure

The repository is organized as follows:

```text
QUADRUPED_FFT_CPUVGPU/
├── results/
│   ├── gait_data.bin
│   ├── gait_fft.exe
│   ├── gait_spectrum.png
│   ├── results.txt
│   └── speedup.png
├── scripts/
│   ├── gait_signal_gen.py
│   └── plot_results.py
└── src/
    └── gait_fft.cu
```

## What each file does

| File | Purpose |
|---|---|
| `src/gait_fft.cu` | Main CUDA/C++ source file that runs CPU FFT, GPU cuFFT, timing, and speedup calculation |
| `scripts/gait_signal_gen.py` | Generates synthetic gait joint-angle data and saves it to `gait_data.bin` |
| `scripts/plot_results.py` | Reads generated binary/timing files and produces output plots |
| `results/gait_data.bin` | Binary file containing the simulated gait signal data |
| `results/gait_fft.exe` | Compiled executable of the CUDA project |
| `results/results.txt` | Timing output used for plotting CPU and GPU performance |
| `results/gait_spectrum.png` | Frequency-domain visualization of all joint signals |
| `results/speedup.png` | Plot comparing CPU execution time, GPU execution time, and speedup |

## Project objective

The objective is to parallelize a meaningful computation and measure the speed improvement of GPU execution over CPU execution. Instead of using a generic benchmark, this project applies FFT to quadruped gait signals so that the result has both computational and robotics relevance.[4][5]

## Problem being solved

A quadruped robot has 12 joints: hip, thigh, and knee for each of the four legs. During walking, each joint produces a periodic angle trajectory over time. By applying FFT to these signals, the dominant motion frequency can be extracted, which corresponds to the gait or stride frequency.[4][5]

This project simulates those joint signals and computes the dominant frequency for each one. Then it compares how long the same job takes on CPU versus GPU.

## Why FFT is used

FFT converts a signal from the **time domain** into the **frequency domain**, making it easier to see which periodic components are present in the signal.[5][6] In this project, the FFT reveals the dominant gait frequency, which is usually around 2 Hz for the simulated trot example.[4]

For example, if a joint oscillates with a stride frequency of 2 Hz, the frequency spectrum shows a strong peak near 2 Hz. Harmonics may also appear at higher frequencies because the signal includes additional periodic components and noise.[5]

## Parallelism in this project

This project demonstrates **data parallelism** because the FFT of one joint does not depend on the FFT of another joint. Each signal can therefore be processed independently.[3]

- **CPU version:** computes FFT for each joint sequentially in a loop.
- **GPU version:** computes FFT for all joints in one batched call with cuFFT.[1]

That is why the GPU can achieve speedup: the workload is naturally parallel and maps well to batched processing.[2]

## Methodology

### 1. Signal generation

The Python script `gait_signal_gen.py` generates synthetic joint-angle signals for 12 joints. Each signal is modeled as a sine wave with:

- a main gait frequency,
- a harmonic component,
- and random noise.
- 

Phase offsets are assigned so that diagonal legs move together, which matches a trot gait pattern.[4]

### 2. CPU FFT

The CPU implementation in `gait_fft.cu` uses a recursive Cooley-Tukey FFT. It processes one joint signal at a time, computes the spectrum, and finds the dominant frequency peak. This serves as the sequential baseline.

### 3. GPU FFT

The GPU implementation uses CUDA memory allocation and transfer functions such as `cudaMalloc` and `cudaMemcpy`, then uses `cufftPlanMany` and `cufftExecR2C` to perform batched FFTs on all joints simultaneously.[1][2]

### 4. Timing and speedup

The CPU section is timed using host timing, while the GPU section is timed using CUDA events. Speedup is computed as:

$$
\text{Speedup} = \frac{\text{CPU time}}{\text{GPU time}}
$$

## Workflow

Run the project in the following order:

### Step 1: Generate gait data

```bash
python scripts/gait_signal_gen.py
```

This creates `results/gait_data.bin`.

### Step 2: Compile the CUDA code

```bash
nvcc src/gait_fft.cu -o results/gait_fft.exe -lcufft
```

### Step 3: Run the executable

```bash
results/gait_fft.exe
```

This produces timing output and writes the results used for plotting.

### Step 4: Plot the output

```bash
python scripts/plot_results.py
```

This generates:

- `results/gait_spectrum.png`
  <img width="1365" height="669" alt="image" src="https://github.com/user-attachments/assets/0884f992-612f-4b94-b86a-164d5f47b3a1" />

- `results/speedup.png`
  <img width="1369" height="668" alt="image" src="https://github.com/user-attachments/assets/fc5379d4-a410-4ac9-8914-c14dd4de578e" />


## Expected output

The program should print the dominant frequency for each joint and show both CPU and GPU timing results. The dominant frequency should match the simulated gait frequency closely, validating that the FFT is correct.[5][4]

The plots should shows a speedup of 400x times and:

- a strong spectral peak near the gait frequency,
- a visual comparison between CPU and GPU execution time,
- and a speedup greater than 1 for sufficiently large workloads.[2][3]

## Key learning outcomes

This project demonstrates:

- practical use of CUDA for parallel computing,[2]
- the use of batched FFT with cuFFT,[1]
- how signal processing applies to robotics gait analysis,[4]
- and how to measure performance gains between CPU and GPU implementations.[2]

## Why this project is stronger than a simple demo

This is more meaningful than a generic rendering example because the computation is explicit and measurable. The same mathematical task is implemented on both CPU and GPU, and the speedup is reported using a real signal-processing application rather than a graphics pipeline abstraction.[2][7]

## Future improvements

Possible extensions include:

- adding an OpenMP CPU version,
- benchmarking multiple FFT sizes and batch counts,
- using real IMU or encoder data from a physical quadruped,
- and comparing cuFFT against a custom CUDA FFT implementation.[3][7]

## Notes for Windows users

On Windows, `nvcc` requires the Microsoft Visual C++ toolchain as the host compiler, so the Visual Studio x64 build environment may need to be loaded before compilation.[8] If `cl.exe` is not found, the Visual Studio Build Tools setup must be configured correctly before running `nvcc`.[8]
