#!/bin/bash
# Quick benchmarking helper script

set -e

BENCH_DIR="target/criterion"
BASELINE_NAME=${1:-""}

echo "╔══════════════════════════════════════════╗"
echo "║  VectorDB Benchmark Runner               ║"
echo "╚══════════════════════════════════════════╝"
echo ""

if [ -z "$BASELINE_NAME" ]; then
    echo "Usage: ./scripts/benchmark.sh <baseline_name>"
    echo ""
    echo "Examples:"
    echo "  ./scripts/benchmark.sh before_opt    # Save baseline"
    echo "  ./scripts/benchmark.sh after_opt     # Compare to baseline"
    echo ""
    echo "Running without baseline (just measure current)..."
    echo ""
    
    # Quick profile first
    echo "=== Quick Profile ==="
    cargo run --release --example profile_query
    echo ""
    
    read -p "Run full benchmarks? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    
    # Full benchmark
    echo "=== Full Benchmark Suite ==="
    cargo bench --bench profile_bench
    
else
    # Check if baseline exists
    if [ -d "$BENCH_DIR/$BASELINE_NAME" ]; then
        echo "Comparing against baseline: $BASELINE_NAME"
        echo ""
        cargo bench --bench profile_bench -- --baseline $BASELINE_NAME
    else
        echo "Creating new baseline: $BASELINE_NAME"
        echo ""
        cargo bench --bench profile_bench -- --save-baseline $BASELINE_NAME
        echo ""
        echo "✅ Baseline saved! Next time run with a different name to compare."
    fi
fi

echo ""
echo "=== View Results ==="
echo "HTML Report: open target/criterion/report/index.html"
echo "Or run: open target/criterion/report/index.html"
