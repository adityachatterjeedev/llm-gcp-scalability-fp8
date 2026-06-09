# Scalability Analysis and FP8 Optimization of Large Language Models on GCP

This repository contains the benchmarking suite, containerization architecture, and infrastructure configurations used to investigate the performance envelopes and hardware-level bottlenecks associated with deploying Large Language Models (LLMs) in serverless, GPU-accelerated environments. 

Using the Gemma-7B model deployed on NVIDIA L4 hardware within Google Cloud Platform (GCP), this study systematically analyzes the transition from compute-bound to memory-bound execution states under varying concurrency loads. It evaluates the efficacy of FP8 quantization as an architectural intervention to mitigate VRAM saturation and defer horizontal or vertical hardware scaling.

---

## Project Overview

The engineering focus of generative AI deployments has shifted from model training to cost-efficient, low-latency production inference. In cloud-native environments, the primary obstacle to scaling LLMs is the high memory footprint of the Key-Value (KV) cache during concurrent request processing.

This project implements a four-phase investigative protocol to map the relationship between request concurrency, memory precision, and system throughput:
* **Phase I (Discovery):** Establishing an infrastructure baseline to isolate cloud-orchestration bottlenecks from native hardware constraints.
* **Phase II (Stress Testing):** Expanding internal concurrency parameters up to a baseline ceiling (C=80) to unmask the hardware's native scaling phase under standard precision.
* **Phase III (Limit Testing):** Executing extreme high-concurrency burst patterns (C=160) under standard FP16 conditions to map the physical boundary of on-device VRAM and force system swapping.
* **Phase IV (Optimization):** Implementing FP8 weight and KV-cache quantization to evaluate baseline execution relief and hardware capacity expansion under identical maximum stress environments.

---

## System Architecture

The deployment infrastructure separates the execution environment from cloud-native orchestration overhead to isolate hardware-specific performance profiles.

```text
├── cloudbuild.yaml          # CI/CD automated build configuration with BuildKit caching
├── Dockerfile               # Production OCI image configuration (vLLM, CUDA 12.4)
├── src/
│   └── benchmark_suite.py   # Multi-threaded synchronous burst load testing framework
├── data/
│   ├── baseline_fp16.csv    # Telemetry data from unoptimized baseline execution
│   └── optimized_fp8.csv    # Telemetry data from FP8 optimized execution
└── docs/
    └── Cloud_ML_Term_Project.pdf # Academic Final Report
```

## Key Findings and Telemetry Analysis

### 1. The Infrastructure "Artificial Knee"
Initial testing under standard FP16 conditions with a conservative cloud-concurrency cap ($C=20$) revealed a sharp degradation in Time to First Token (TTFT), jumping from 0.5s to 2.1s. Forensic logging confirmed the bottleneck was not hardware saturation, but rather an infrastructure throttle forcing client requests to queue at the load balancer level before reaching the inference engine.

### 2. Hardware Saturation and CPU Swapping
Bypassing the load balancer limits ($C=160$) unmasked the true physical boundary of the NVIDIA L4. Between 100 and 160 concurrent users, the system hit a secondary performance knee as VRAM was fully exhausted. The vLLM engine was forced into a memory-swapping state, allocating 2,048 CPU blocks to prevent system failure, which introduced a 108% increase in system latency.

### 3. FP8 Architectural Relief
Transitioning the vLLM engine execution entrypoint to 8-bit precision (`fp8_e5m2`) for both weights and KV-cache blocks yielded a drastic optimization in hardware capacity:

| Metric | FP16 Stress Test | FP8 Stress Test | Delta |
| :--- | :--- | :--- | :--- |
| **Available GPU Blocks** | 588 | 1341 | +128% |
| **CPU Swap Blocks** | 2048 | 2048 | 0% |
| **TTFT at Max Concurrency** | 3.52s | 3.39s | Statistical Parity |

The implementation of FP8 deferred the performance knee entirely, expanding high-performance user capacity by 128% on the same mid-tier 24GB accelerator while maintaining statistical parity in response metrics.

---

## Implementation Guide

### Prerequisites
* Google Cloud SDK (`gcloud`) authenticated to a project with appropriate GPU quotas.
* Docker or an OCI-compliant runtime engine.
* A Hugging Face Read Token authorized to access the Gemma-7B model repository.

### Building the Container Environment
The deployment decouples the model configurations from the base image structure. Build the optimized OCI image using Google Cloud Build, which incorporates layer caching optimizations to reduce heavy image assembly overhead:

```bash
gcloud builds submit --config=cloudbuild.yaml --substitutions=_HF_TOKEN="your_hf_token_here"
```

### Deploying to Google Cloud Run

Deploy two discrete revisions to observe the performance delta between precision boundaries.

#### Deploying the FP16 Baseline Configuration:

```bash
gcloud run deploy ml-inference-baseline \
    --image=us-central1-docker.pkg.dev/$PROJECT_ID/ml-repo/vllm-inference:latest \
    --cpu=8 \
    --memory=32Gi \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --max-instances=1 \
    --no-allow-unauthenticated \
    --set-env-vars="VLLM_EXTRA_ARGS="
```

#### Deploying the FP8 Optimized Configuration:
```bash
gcloud run deploy ml-inference-optimized \
    --image=us-central1-docker.pkg.dev/$PROJECT_ID/ml-repo/vllm-inference:latest \
    --cpu=8 \
    --memory=32Gi \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --max-instances=1 \
    --no-allow-unauthenticated \
    --set-env-vars="VLLM_EXTRA_ARGS=--kv-cache-dtype fp8_e5m2 --quantization fp8"
```

#### Executing the Benchmarking Suite
The included load-testing utility uses persistent TCP connection pooling and a multi-threaded execution loop to simulate instantaneous, synchronous client bursts. This ensures telemetry measurements represent pure inference response curves isolated from handshake overhead.

Execute the performance sweep against a targeted deployment URL:
```bash
python3 src/benchmark_suite.py \
    --url "https://your-cloud-run-endpoint-url/v1/chat/completions" \
    --model "google/gemma-2-7b-it" \
    --output "data/optimized_fp8.csv"
```

## Production and Economic Implications

The operational transition from unoptimized FP16 to FP8 precision outlines clear architectural tradeoffs for corporate AI infrastructure:
* **Cost Efficiency:** By doubling the available KV-cache block allocation within the limits of 24GB hardware, FP8 acts as a financial force multiplier, effectively doubling the active user ceiling per dollar spent without provisioning higher-tier infrastructure.
* **SLA Preservation:** CPU swapping creates severe systemic latency jitter. True production service-level agreements (SLAs) must map high-performance capacity to the swapping threshold rather than instance crashes. FP8 extends this high-performance ceiling.
* **Serverless Cold Starts:** Telemetry tracked an initial execution cold start peak of 103.74s due to pulling and initializing the >15GB container space across the network. Consequently, a strict scale-to-zero operational loop is highly discouraged for latency-critical customer faces without setting a minimum warm instance baseline (`--min-instances`).

---

## References

* Google DeepMind. Gemma: Open Models Based on Gemini Research and Technology. (2024).
* Google Cloud. Cloud Run documentation: GPU acceleration in Cloud Run. (2024).
* Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention". Proceedings of the 29th Symposium on Operating Systems Principles (SOSP). (2023).
* Micikevicius, P., et al. "FP8 Formats for Deep Learning". arXiv preprint arXiv:2209.05430. (2022).

---

## Operational Note on Reproducibility

The telemetry data and benchmark results documented in this repository were originally generated using institutional compute clusters and temporary Google Cloud Platform (GCP) allocations granted during my graduate tenure. 

Because active GPU access has concluded post-graduation, live deployment endpoints are currently offline to manage cloud infrastructure costs. However, the exact OCI configurations (`Dockerfile`), CI/CD automation (`cloudbuild.yaml`), and multi-threaded stress-testing architecture (`benchmark_suite.py`) are fully preserved here to ensure the entire pipeline remains completely reproducible for anyone deploying this stack with an active NVIDIA L4 environment.