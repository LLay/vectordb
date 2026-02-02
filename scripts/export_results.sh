#!/bin/bash
# Export SIFT benchmark results in VectorDBBench format
#
# Usage:
#   ./scripts/export_results.sh [output_file]
#
# This creates a JSON file compatible with VectorDBBench submissions

set -e

OUTPUT_FILE="${1:-vectordbbench_results.json}"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         VectorDBBench Results Export                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if benchmark has been run
if [ ! -d "target/criterion/sift_1m_balanced" ] && [ ! -d "target/criterion/sift_10k_balanced" ]; then
    echo "Error: No SIFT benchmark results found!"
    echo ""
    echo "Please run the benchmark first:"
    echo "  cargo bench --bench sift_benchmark"
    echo ""
    exit 1
fi

# Detect dataset size
DATASET_SIZE="unknown"
if [ -d "target/criterion/sift_10k_balanced" ]; then
    DATASET_SIZE="10k"
    DATASET_NAME="SIFT-10K"
elif [ -d "target/criterion/sift_1m_balanced" ]; then
    DATASET_SIZE="1m"
    DATASET_NAME="SIFT-1M"
fi

echo "Detected dataset: $DATASET_NAME"
echo ""

# Get system info
OS=$(uname -s)
ARCH=$(uname -m)
CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | grep "Model name" | cut -d: -f2 | xargs || echo "Unknown")
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc || echo "Unknown")
RAM=$(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 " GB"}' || free -g | awk '/^Mem:/{print $2 " GB"}' || echo "Unknown")

echo "System Information:"
echo "  OS:     $OS"
echo "  Arch:   $ARCH"
echo "  CPU:    $CPU"
echo "  Cores:  $CORES"
echo "  RAM:    $RAM"
echo ""

# Prompt for benchmark results
# These should come from the last benchmark run
echo "Enter benchmark results from your last run:"
echo "(You can find these in the console output after running the benchmark)"
echo ""

read -p "Low Latency - Recall@10 (%): " ll_r10
read -p "Low Latency - Recall@100 (%): " ll_r100
read -p "Low Latency - QPS: " ll_qps
read -p "Low Latency - p50 (ms): " ll_p50
read -p "Low Latency - p99 (ms): " ll_p99

echo ""
read -p "Balanced - Recall@10 (%): " bal_r10
read -p "Balanced - Recall@100 (%): " bal_r100
read -p "Balanced - QPS: " bal_qps
read -p "Balanced - p50 (ms): " bal_p50
read -p "Balanced - p99 (ms): " bal_p99

echo ""
read -p "High Recall - Recall@10 (%): " hr_r10
read -p "High Recall - Recall@100 (%): " hr_r100
read -p "High Recall - QPS: " hr_qps
read -p "High Recall - p50 (ms): " hr_p50
read -p "High Recall - p99 (ms): " hr_p99

echo ""
read -p "Index build time (seconds): " build_time
read -p "Index size (MB): " index_size

# Create JSON file
cat > "$OUTPUT_FILE" << EOF
{
  "system": "CuddleDB",
  "version": "0.1.0",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "dataset": {
    "name": "$DATASET_NAME",
    "type": "sift-128-euclidean",
    "vectors": "${DATASET_SIZE}",
    "dimensions": 128,
    "metric": "L2"
  },
  "hardware": {
    "os": "$OS",
    "arch": "$ARCH",
    "cpu": "$CPU",
    "cores": $CORES,
    "ram": "$RAM"
  },
  "index": {
    "type": "hierarchical_kmeans",
    "build_time_seconds": $build_time,
    "index_size_mb": $index_size,
    "parameters": {
      "branching_factor": 100,
      "target_leaf_size": 100,
      "max_iterations": 20
    }
  },
  "results": {
    "low_latency": {
      "configuration": {
        "probes": 2,
        "rerank_factor": 2
      },
      "metrics": {
        "qps": $ll_qps,
        "latency_p50_ms": $ll_p50,
        "latency_p99_ms": $ll_p99,
        "recall_at_10": $(echo "scale=4; $ll_r10 / 100" | bc),
        "recall_at_100": $(echo "scale=4; $ll_r100 / 100" | bc)
      }
    },
    "balanced": {
      "configuration": {
        "probes": 5,
        "rerank_factor": 3
      },
      "metrics": {
        "qps": $bal_qps,
        "latency_p50_ms": $bal_p50,
        "latency_p99_ms": $bal_p99,
        "recall_at_10": $(echo "scale=4; $bal_r10 / 100" | bc),
        "recall_at_100": $(echo "scale=4; $bal_r100 / 100" | bc)
      }
    },
    "high_recall": {
      "configuration": {
        "probes": 10,
        "rerank_factor": 5
      },
      "metrics": {
        "qps": $hr_qps,
        "latency_p50_ms": $hr_p50,
        "latency_p99_ms": $hr_p99,
        "recall_at_10": $(echo "scale=4; $hr_r10 / 100" | bc),
        "recall_at_100": $(echo "scale=4; $hr_r100 / 100" | bc)
      }
    }
  }
}
EOF

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Results Exported Successfully!                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "  1. Review the results: cat $OUTPUT_FILE | jq ."
echo "  2. Compare to leaderboard: https://zilliz.com/vdbbench-leaderboard"
echo "  3. Submit to VectorDBBench (optional):"
echo "     - Fork: https://github.com/zilliztech/VectorDBBench"
echo "     - Add your results to results/ directory"
echo "     - Submit PR with this file"
echo ""
