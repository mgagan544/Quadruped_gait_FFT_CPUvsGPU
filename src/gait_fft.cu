// gait_fft.cu
// Compile: nvcc gait_fft.cu -o gait_fft.exe -lcufft
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define NUM_JOINTS  12
#define N_SAMPLES   4096
#define SAMPLE_RATE 1000.0f
#define GAIT_FREQ   2.0f

// ── Signal generation ────────────────────────────────────────────────
void generate_signal(float* out, int joint, int N, float fs, float gait_freq) {
    float phases[4] = {0.0f, 3.14159f, 3.14159f, 0.0f};
    float amps[3]   = {0.3f, 0.5f, 0.4f};
    int leg   = joint / 3;
    float ph  = phases[leg % 4];
    float amp = amps[joint % 3];
    for (int i = 0; i < N; i++) {
        float t = (float)i / fs;
        out[i] = amp * sinf(2 * 3.14159f * gait_freq * t + ph)
               + 0.3f * amp * sinf(2 * 3.14159f * 2 * gait_freq * t + ph)
               + ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
    }
}

// ── CPU Cooley-Tukey FFT ─────────────────────────────────────────────
// Replace the typedef and 3 functions with this:
struct Cplx {
    float r, i;
    Cplx(float r=0, float i=0): r(r), i(i){}
};
Cplx c_add(Cplx a, Cplx b){ return Cplx(a.r+b.r, a.i+b.i); }
Cplx c_sub(Cplx a, Cplx b){ return Cplx(a.r-b.r, a.i-b.i); }
Cplx c_mul(Cplx a, Cplx b){ return Cplx(a.r*b.r-a.i*b.i, a.r*b.i+a.i*b.r); }
void fft_cpu(Cplx* x, int n) {
    if (n <= 1) return;
    Cplx *even = (Cplx*)malloc(n/2 * sizeof(Cplx));
    Cplx *odd  = (Cplx*)malloc(n/2 * sizeof(Cplx));
    for (int i = 0; i < n/2; i++) { even[i] = x[2*i]; odd[i] = x[2*i+1]; }
    fft_cpu(even, n/2);
    fft_cpu(odd,  n/2);
    for (int k = 0; k < n/2; k++) {
        float ang = -2.0f * 3.14159f * k / n;
        Cplx t = c_mul(Cplx(cosf(ang), sinf(ang)), odd[k]);
        x[k]       = c_add(even[k], t);
        x[k + n/2] = c_sub(even[k], t);
    }
    free(even); free(odd);
}

float dominant_freq(float* sig, int N, float fs) {
    Cplx* x = (Cplx*)malloc(N * sizeof(Cplx));
    for (int i = 0; i < N; i++) { x[i].r = sig[i]; x[i].i = 0; }
    fft_cpu(x, N);
    float max_mag = 0; int max_idx = 0;
    for (int i = 1; i < N/2; i++) {
        float mag = sqrtf(x[i].r*x[i].r + x[i].i*x[i].i);
        if (mag > max_mag) { max_mag = mag; max_idx = i; }
    }
    free(x);
    return (float)max_idx * fs / N;
}

// ── Main ─────────────────────────────────────────────────────────────
int main() {
    printf("=== Quadruped Gait FFT: CPU vs GPU ===\n");
    printf("Joints: %d | Samples: %d | Fs: %.0f Hz | Gait: %.1f Hz\n\n",
           NUM_JOINTS, N_SAMPLES, SAMPLE_RATE, GAIT_FREQ);

    // Generate all joint signals
    float* h_data = (float*)malloc(NUM_JOINTS * N_SAMPLES * sizeof(float));
    srand(42);
    for (int j = 0; j < NUM_JOINTS; j++)
        generate_signal(h_data + j * N_SAMPLES, j, N_SAMPLES, SAMPLE_RATE, GAIT_FREQ);

    // ── CPU Sequential ──────────────────────────────────────────────
    clock_t cpu_start = clock();
    float cpu_freqs[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; j++)
        cpu_freqs[j] = dominant_freq(h_data + j * N_SAMPLES, N_SAMPLES, SAMPLE_RATE);
    clock_t cpu_end = clock();
    double cpu_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    printf("[CPU Sequential]\n");
    const char* legs[]   = {"FL","FR","BL","BR"};
    const char* joints[] = {"Hip","Thigh","Knee"};
    for (int j = 0; j < NUM_JOINTS; j++)
        printf("  %s-%s: %.2f Hz\n", legs[j/3], joints[j%3], cpu_freqs[j]);
    printf("  Time: %.3f ms\n\n", cpu_ms);

    // ── GPU Batch cuFFT ─────────────────────────────────────────────
    cufftReal*    d_in;
    cufftComplex* d_out;
    int out_size = N_SAMPLES/2 + 1;

    cudaMalloc(&d_in,  NUM_JOINTS * N_SAMPLES * sizeof(cufftReal));
    cudaMalloc(&d_out, NUM_JOINTS * out_size  * sizeof(cufftComplex));

    cudaMemcpy(d_in, h_data, NUM_JOINTS * N_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);

    // Batch plan: all 12 joints in ONE kernel call
    cufftHandle plan;
    int n[]      = { N_SAMPLES };
    int idist    = N_SAMPLES;
    int odist    = out_size;
    cufftPlanMany(&plan, 1, n,
                  NULL, 1, idist,
                  NULL, 1, odist,
                  CUFFT_R2C, NUM_JOINTS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cufftExecR2C(plan, d_in, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Read back & find dominant freq per joint
    cufftComplex* h_out = (cufftComplex*)malloc(NUM_JOINTS * out_size * sizeof(cufftComplex));
    cudaMemcpy(h_out, d_out, NUM_JOINTS * out_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    printf("[GPU Batch cuFFT — all %d joints in parallel]\n", NUM_JOINTS);
    for (int j = 0; j < NUM_JOINTS; j++) {
        float max_mag = 0; int max_idx = 0;
        for (int k = 1; k < N_SAMPLES/2; k++) {
            float re = h_out[j * out_size + k].x;
            float im = h_out[j * out_size + k].y;
            float mag = sqrtf(re*re + im*im);
            if (mag > max_mag) { max_mag = mag; max_idx = k; }
        }
        printf("  %s-%s: %.2f Hz\n", legs[j/3], joints[j%3],
               (float)max_idx * SAMPLE_RATE / N_SAMPLES);
    }
    printf("  Time: %.3f ms\n\n", gpu_ms);

    // ── Results ─────────────────────────────────────────────────────
    printf("╔══════════════════════════════════╗\n");
    printf("║  CPU time : %8.3f ms          ║\n", cpu_ms);
    printf("║  GPU time : %8.3f ms          ║\n", gpu_ms);
    printf("║  Speedup  : %8.2fx             ║\n", cpu_ms / gpu_ms);
    printf("╚══════════════════════════════════╝\n");

    // Save signals for Python plotting
FILE* f = fopen("gait_data.bin", "wb");
fwrite(h_data, sizeof(float), NUM_JOINTS * N_SAMPLES, f);
fclose(f);

// Save timing results
FILE* r = fopen("results.txt", "w");
fprintf(r, "cpu_ms,gpu_ms\n%.4f,%.4f\n", cpu_ms, gpu_ms);
fclose(r);
printf("Saved gait_data.bin and results.txt\n");

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_data); free(h_out);
    return 0;
}