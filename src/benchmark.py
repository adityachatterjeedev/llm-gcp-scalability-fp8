#!/usr/bin/env python3
"""
LLM Benchmarking Suite
Designed to analyze concurrency scaling, TTFT, and TPS metrics under load.
References: Cloud_ML_Term_Project.pdf
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from statistics import mean
import requests

# --- DEFAULT CONFIGURATION ---
DEFAULT_URL = "https://ml-inference-service-1007258452604.us-central1.run.app/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-2-7b-it"
DEFAULT_PROMPT = "Explain the importance of GPU memory bandwidth in deep learning in 100 words."
CONCURRENCY_LEVELS = [1, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160]

# Initialize a persistent session to eliminate TCP handshake overhead from metrics
session = requests.Session()


def measure_request(request_id: int, url: str, model: str, prompt: str) -> dict:
    """
    Simulates a single client thread streaming tokens from the vLLM server.
    Measures Time to First Token (TTFT), Tokens Per Second (TPS), and Latency.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "stream": True  # Crucial for measuring TTFT
    }

    start_time = time.time()
    ttft = None
    total_tokens = 0

    try:
        # Utilizing the persistent session for performance isolation
        response = session.post(url, json=payload, headers=headers, stream=True, timeout=120)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode('utf-8').strip()
            if line_text.startswith("data: "):
                data_payload = line_text[6:]  # Strip 'data: ' prefix
                
                if data_payload == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_payload)
                    
                    # Capture TTFT upon receipt of the first content chunk
                    if ttft is None and data['choices'][0]['delta'].get('content'):
                        ttft = time.time() - start_time
                    
                    # Count tokens via internal usage metadata or fallback estimation
                    if 'usage' in data and data['usage'] is not None:
                        total_tokens = data['usage'].get('completion_tokens', total_tokens)
                    elif data['choices'][0]['delta'].get('content'):
                        total_tokens += 1
                        
                except json.JSONDecodeError:
                    continue

        end_time = time.time()
        total_latency = end_time - start_time
        
        # Ensure fallback protection if engine never yields an explicit chunk
        if ttft is None:
            ttft = total_latency

        generation_time = total_latency - ttft
        tps = total_tokens / generation_time if generation_time > 0 else 0

        return {
            "id": request_id,
            "ttft": ttft,
            "tps": tps,
            "total_latency": total_latency,
            "tokens_generated": total_tokens,
            "success": True
        }
        
    except Exception as e:
        print(f"  [ERROR] Worker {request_id} encountered an exception: {e}", file=sys.stderr)
        return {"success": False, "id": request_id}


def run_burst(concurrency: int, url: str, model: str, prompt: str) -> list:
    """
    Orchestrates synchronous burst patterns using a thread pool executor.
    Forces concurrent client traffic to target the endpoint at the same millisecond.
    """
    print(f"\n Initiating Synchronous Burst | Target Concurrency: {concurrency}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Map worker indices into concurrent payload functions
        futures = [
            executor.submit(measure_request, i, url, model, prompt) 
            for i in range(concurrency)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    successes = [r for r in results if r["success"]]
    failures = len(results) - len(successes)
    
    if successes:
        avg_ttft = mean([r["ttft"] for r in successes])
        avg_tps = mean([r["tps"] for r in successes])
        avg_lat = mean([r["total_latency"] for r in successes])
        aggregate_throughput = sum([r["tps"] for r in successes])
        
        print(f"Summary (C={concurrency}):")
        print(f"   Avg TTFT:    {avg_ttft:.3f}s")
        print(f"   Avg TPS:     {avg_tps:.2f} tokens/sec")
        print(f"   Avg Latency: {avg_lat:.3f}s")
        print(f"   Aggregate:   {aggregate_throughput:.2f} tokens/sec (Total System Efficiency)")
        if failures > 0:
            print(f"   Dropouts:  {failures} requests failed.")
            
        return {
            "concurrency": concurrency,
            "avg_ttft": avg_ttft,
            "avg_tps": avg_tps,
            "avg_latency": avg_lat,
            "aggregate_tps": aggregate_throughput,
            "failures": failures
        }
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded Concurrency & Quantization Stress Tester")
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint target URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Target model identifier tag")
    parser.add_argument("--output", default="benchmark_results.csv", help="Destination CSV track file")
    args = parser.parse_args()

    print("======================================================")
    print("Starting Cloud AI Load Testing Performance Sweep")
    print(f"Target Cluster: {args.url}")
    print(f"Active Model:   {args.model}")
    print("======================================================")

    summary_metrics = []

    for level in CONCURRENCY_LEVELS:
        burst_data = run_burst(level, args.url, args.model, DEFAULT_PROMPT)
        if burst_data:
            summary_metrics.append(burst_data)

    # Save summary data seamlessly to a CSV for automated visualization plotting
    if summary_metrics:
        keys = summary_metrics[0].keys()
        with open(args.output, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary_metrics)
        print(f"\n Data collection complete. Metrics exported clean to '{args.output}'.")
