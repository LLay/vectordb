# VectorDB Benchmarks

Comprehensive benchmark suite for measuring VectorDB performance using [Criterion.rs](https://github.com/bheisler/criterion.rs).

## Running Benchmarks

### Run all benchmarks
```bash
cargo bench
```

### Run specific benchmark suite
```bash
cargo bench --bench distance_bench
cargo bench --bench quantization_bench
cargo bench --bench clustering_bench
cargo bench --bench index_bench
cargo bench --bench e2e_bench
```

### Quick benchmarks (less accurate, faster)
```bash
cargo bench -- --quick
```

### Run specific benchmark within a suite
```bash
cargo bench --bench distance_bench -- dot_product
cargo bench --bench index_bench -- search_varying_k
```

## Benchmark Suites

### 1. `distance_bench.rs` - Low-Level SIMD Operations
**What it measures:** Core distance computation performance

- **`dot_product`**: NEON-optimized dot product across various dimensions (128-2048)
- **`l2_squared`**: NEON-optimized L2 squared distance computation
- **`batch_distances`**: Sequential vs parallel batch distance computation (100-10K vectors)

**Key metrics:** Throughput (GiB/s), latency per operation

**Use case:** Validate SIMD optimizations, ensure memory bandwidth utilization

---

### 2. `quantization_bench.rs` - Compression Performance
**What it measures:** Binary quantization (f32 → 1-bit) performance

- **`quantize_single`**: Single vector quantization across dimensions
- **`quantize_batch_sequential`**: Sequential batch quantization
- **`quantize_batch_parallel`**: Parallel batch quantization (rayon)
- **`hamming_distance`**: Fast Hamming distance with NEON popcount
- **`hamming_batch`**: Batch Hamming distance computation
- **`quantizer_from_vectors`**: Threshold computation from training data

**Key metrics:** Compression throughput, distance computation speedup vs f32

**Use case:** Measure 32x compression impact, validate memory savings

---

### 3. `clustering_bench.rs` - K-Means Performance
**What it measures:** Clustering algorithm performance for index building

- **`kmeans_init_plusplus`**: k-means++ initialization for various k values
- **`kmeans_fit`**: Full k-means convergence time
- **`kmeans_assign`**: Vector-to-cluster assignment performance
- **`kmeans_nearest_centroid`**: Single query centroid search
- **`kmeans_varying_dimensions`**: Impact of vector dimensionality
- **`kmeans_get_clusters`**: Cluster grouping operations

**Key metrics:** Convergence time, assignment throughput

**Use case:** Optimize index build time, tune k values

---

### 4. `index_bench.rs` - Index Operations
**What it measures:** Hierarchical clustered index performance

- **`index_build`**: Index construction time (1K-50K vectors)
- **`index_search_varying_k`**: Query latency for k=1,10,50,100 neighbors
- **`index_search_varying_probes`**: Trade-off between accuracy (probes) and speed
- **`index_search_varying_rerank`**: Impact of rerank factor on precision/recall
- **`index_search_dimensions`**: Query performance across dimensions (128-1536)

**Key metrics:** Build time, query latency (ms), queries per second (QPS)

**Use case:** Tune index parameters (branching, max_leaf, probes, rerank)

---

### 5. `e2e_bench.rs` - End-to-End Workflows
**What it measures:** Real-world usage patterns

- **`e2e_full_pipeline`**: Build index + search (full workflow)
- **`e2e_quantized_search`**: Linear scan with binary quantization
- **`e2e_mixed_precision_search`**: Two-phase search:
  1. Fast binary scan (all vectors)
  2. Full precision rerank (top candidates)
- **`e2e_batch_queries`**: Multiple queries per batch (10-1000)
- **`e2e_memory_efficiency`**: Large-scale quantization (10K-100K vectors)

**Key metrics:** End-to-end latency, throughput, memory usage

**Use case:** Validate production performance, measure recall@k vs speed trade-offs

---

## Understanding Output

### Time Measurements
```
time:   [8.7537 ns 8.8109 ns 9.0399 ns]
        ^        ^         ^
        lower    estimate  upper (95% confidence interval)
```

### Throughput
```
thrpt:  [105.50 GiB/s 108.24 GiB/s 108.95 GiB/s]
```
Higher is better. Shows data processing rate.

### Performance Change
```
change:
    time:   [+2.16% +3.97% +5.77%] (p = 0.04 < 0.05)
    thrpt:  [-5.46% -3.82% -2.12%]
    Performance has regressed.
```
- **p < 0.05**: Statistically significant change
- **p > 0.05**: Within noise, no real change

---

## Benchmark Reports

After running benchmarks, view detailed reports:

```bash
open target/criterion/report/index.html
```

Reports include:
- Performance over time (regression detection)
- Violin plots (distribution visualization)
- Statistical analysis
- Comparison with previous runs

---

## Tips

### Fast Iteration
```bash
# Quick check (less accurate)
cargo bench --bench distance_bench -- --quick

# Single benchmark
cargo bench --bench index_bench -- search_varying_k/10
```

### CI/CD Integration
```bash
# Save baseline
cargo bench -- --save-baseline main

# Compare against baseline
cargo bench -- --baseline main
```

### Profile Hot Paths
```bash
# Generate flamegraphs (requires cargo-flamegraph)
cargo flamegraph --bench distance_bench -- --bench
```

---

## Key Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Dot product (1024-dim) | < 20ns | ~4x SIMD speedup |
| L2 distance (1024-dim) | < 30ns | Memory bandwidth bound |
| Hamming distance (1024-bit) | < 10ns | NEON popcount |
| Index search (10K vectors) | < 100μs | k=10, probes=2 |
| Build index (10K vectors) | < 2s | 1024-dim |
| Quantize batch (10K vectors) | < 50ms | Parallel |

---

## Adding New Benchmarks

1. Create new file in `benches/` (e.g., `my_bench.rs`)
2. Use Criterion template:
```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_my_feature(c: &mut Criterion) {
    c.bench_function("my_feature", |b| {
        b.iter(|| {
            // code to benchmark
        })
    });
}

criterion_group!(benches, bench_my_feature);
criterion_main!(benches);
```
3. Run: `cargo bench --bench my_bench`

---

## Interpreting Results for Optimization

### Memory Bandwidth Bound
If throughput plateaus ~100-120 GiB/s → hitting DRAM bandwidth limit (M1/M2)

### CPU Bound
If throughput scales with vector dimension → CPU computation dominant

### Cache Effects
Test with increasing data sizes to find L1/L2/L3 cache limits

### SIMD Efficiency
Compare scalar (removed) vs NEON → should see 4x speedup for f32 operations
