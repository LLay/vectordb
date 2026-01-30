# VectorDB Benchmarking Guide

## Quick Start

Run all benchmarks:
```bash
cargo bench
```

Run a specific benchmark suite:
```bash
cargo bench --bench distance_bench
cargo bench --bench quantization_bench
cargo bench --bench clustering_bench
cargo bench --bench index_bench
cargo bench --bench e2e_bench
```

View results:
```bash
open target/criterion/report/index.html
```

## Benchmark Suites Overview

### 1. **distance_bench** - SIMD Operations (Fast, ~30s)
Core NEON-optimized distance functions:
- Dot product (128-2048 dimensions)
- L2 squared distance
- Batch operations (sequential vs parallel)

**Key Insight:** Measures raw SIMD performance, memory bandwidth utilization

### 2. **quantization_bench** - Compression (Medium, ~5-10min)
Binary quantization (f32 → 1-bit) pipeline:
- Single vector quantization
- Batch quantization (sequential vs parallel)
- Hamming distance computation
- Threshold learning from data

**Key Insight:** Validates 32x compression, measures Hamming distance speedup

### 3. **clustering_bench** - K-Means (Slow, ~10-15min)
Clustering for index construction:
- k-means++ initialization
- Full k-means convergence
- Vector assignment
- Centroid search

**Key Insight:** Optimizes index build time, tunes cluster parameters

### 4. **index_bench** - Hierarchical Index (Very Slow, ~15-20min)
End-to-end index operations:
- Index build time (1K-50K vectors)
- Search with varying k, probes, rerank factor
- Multi-dimensional scaling

**Key Insight:** Tune production parameters for recall/latency trade-offs

### 5. **e2e_bench** - Real-World Workflows (Slow, ~10-15min)
Complete pipelines:
- Build + search workflow
- Quantized linear scan
- Two-phase search (binary → full precision)
- Batch queries
- Memory efficiency at scale

**Key Insight:** Validates production performance, measures end-to-end latency

---

## Sample Results

From the quantization benchmark on Apple Silicon:

```
quantize_single/1024    time:   [1.10 µs 1.13 µs 1.17 µs]
                        thrpt:  [3.27 GiB/s 3.39 GiB/s 3.48 GiB/s]

quantize_batch_sequential/10000
                        time:   [44.75 ms 44.83 ms 44.95 ms]
                        thrpt:  [222.49 Kelem/s 223.05 Kelem/s 223.46 Kelem/s]
```

**Analysis:**
- Single 1024-dim vector: ~1.1μs to quantize (32x compression)
- Batch of 10K vectors: ~45ms (222K vectors/sec throughput)
- Throughput: ~3.3 GiB/s input processing rate

---

## Understanding Your Results

### What to Look For

**1. SIMD Efficiency (distance_bench)**
- Dot product 1024-dim should be < 20ns
- ~4x speedup vs scalar (previously removed)
- Throughput near 100-120 GiB/s = good memory utilization

**2. Quantization Speed (quantization_bench)**
- Single vector < 2μs for 1024-dim
- Hamming distance < 10ns for 1024-bit
- Parallel should be 4-8x faster than sequential (CPU dependent)

**3. Index Performance (index_bench)**
- Build 10K vectors in < 2 seconds
- Search latency < 100μs for k=10
- Probes=2 should be ~2x slower than probes=1

**4. Regression Detection**
Look for:
```
change: time: [+5.77%] (p = 0.04 < 0.05)
Performance has regressed.
```
This means code got 5.77% slower with statistical confidence.

---

## Optimization Workflow

### 1. Baseline Measurement
```bash
# Save current performance
cargo bench -- --save-baseline main
```

### 2. Make Changes
Edit code, optimize algorithms, etc.

### 3. Compare
```bash
# Compare against baseline
cargo bench -- --baseline main
```

### 4. Iterate
Look for regressions, optimize hot paths.

---

## Performance Tuning Parameters

### Index Parameters (index_bench)
- **branching_factor**: More = faster build, slower search
  - Recommended: 10-20
- **max_leaf_size**: Larger = fewer levels, more leaf scanning
  - Recommended: 100-200
- **probes_per_level**: More = better recall, slower
  - Recommended: 2-5
- **rerank_factor**: Multiplier for precision phase
  - Recommended: 3-5

### K-Means Parameters (clustering_bench)
- **k**: Number of clusters per level
  - Same as branching_factor typically
- **max_iterations**: Convergence limit
  - Recommended: 10-20

### Quantization Strategy (e2e_bench)
- **Mixed precision**: Binary scan + full precision rerank
  - Best balance of speed and accuracy
- **Pure binary**: Fastest, lower recall
- **No quantization**: Highest recall, slowest

---

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run benchmarks
  run: cargo bench --bench distance_bench -- --quick

- name: Archive results
  uses: actions/upload-artifact@v3
  with:
    name: criterion-results
    path: target/criterion/
```

### Performance Regression Checks
```bash
# In CI: fail if performance drops > 10%
cargo bench -- --save-baseline ci-baseline
cargo bench -- --baseline ci-baseline --message-format=json | jq '.reason == "benchmark-complete" and .change.mean.point_estimate > 0.10'
```

---

## Troubleshooting

### Benchmarks Take Too Long
```bash
# Use --quick for faster (less accurate) results
cargo bench --bench distance_bench -- --quick
```

### High Variance in Results
- Close other applications
- Disable CPU frequency scaling
- Run on battery power (for laptops - consistent power)
- Increase sample size (edit benchmark code)

### Memory Issues
For large benchmarks (100K+ vectors):
```rust
group.sample_size(10); // Reduce samples
```

---

## Advanced: Custom Metrics

### Add Custom Measurements
```rust
c.bench_function("custom", |b| {
    b.iter_custom(|iters| {
        let start = Instant::now();
        for _ in 0..iters {
            // your code
        }
        start.elapsed()
    })
});
```

### Memory Profiling
```bash
# Requires cargo-instrument
cargo install cargo-instruments
cargo instruments --bench index_bench --template Allocations
```

### Flamegraphs
```bash
cargo install flamegraph
cargo flamegraph --bench distance_bench -- --bench
```

---

## Next Steps

1. **Establish Baselines**: Run full benchmark suite once, save results
2. **Monitor Over Time**: Criterion automatically tracks performance
3. **Optimize Hot Paths**: Use flamegraphs to find bottlenecks
4. **Tune Parameters**: Use index_bench to find optimal settings
5. **Validate Changes**: Always benchmark before/after optimizations

For detailed per-benchmark documentation, see `benches/README.md`.
