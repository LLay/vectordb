# VectorDB Benchmarks

Comprehensive benchmark suite for measuring VectorDB performance using [Criterion.rs](https://github.com/bheisler/criterion.rs).

---

## Quick Reference

### For Daily Development (< 10 seconds)
```bash
cargo run --release --example quick_recall_check
```

### For Fast Iteration (< 2 minutes)
```bash
cargo bench --bench speed_fast    # ~70s - latency only
cargo bench --bench recall_fast   # ~30s - recall + latency
```

### For Production Validation (2-10 minutes)
```bash
cargo bench --bench recall_proper      # ~3m - comprehensive recall
cargo bench --bench tune_tree_params   # ~5m - find optimal params
cargo bench --bench profile_bench      # ~10m - full profiling with 1M vectors
```

---

## Running Benchmarks

### Run all benchmarks
```bash
cargo bench
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

### View results
```bash
open target/criterion/report/index.html
```

---

## Available Benchmarks

| Name | Type | Time | Dataset | Purpose |
|------|------|------|---------|---------|
| `quick_recall_check` | Example | 8s | 1K×128d | Quick sanity check |
| `speed_fast` | Bench | 70s | 5K×128d | Fast latency measurement |
| `recall_fast` | Bench | 30s | 2K×128d | Fast recall testing |
| `recall_proper` | Bench | 3m | 10K×256d | Comprehensive recall |
| `tune_tree_params` | Bench | 5m | 10K×256d | Parameter optimization |
| `profile_bench` | Bench | 10m | 1M×1024d | Production profiling |
| `recall_bench` | Bench | 5m | 50K×512d | Original recall benchmark |

---

## Benchmark Suites

### 1. `distance_bench.rs` - Low-Level SIMD Operations (Fast, ~30s)
**What it measures:** Core distance computation performance

- **`dot_product`**: NEON-optimized dot product across various dimensions (128-2048)
- **`l2_squared`**: NEON-optimized L2 squared distance computation
- **`batch_distances`**: Sequential vs parallel batch distance computation (100-10K vectors)

**Key metrics:** Throughput (GiB/s), latency per operation

**Use case:** Validate SIMD optimizations, ensure memory bandwidth utilization

**Performance targets:**
- Dot product 1024-dim should be < 20ns (~4x SIMD speedup)
- Throughput near 100-120 GiB/s = good memory utilization

---

### 2. `quantization_bench.rs` - Compression Performance (Medium, ~5-10min)
**What it measures:** Binary quantization (f32 → 1-bit) performance

- **`quantize_single`**: Single vector quantization across dimensions
- **`quantize_batch_sequential`**: Sequential batch quantization
- **`quantize_batch_parallel`**: Parallel batch quantization (rayon)
- **`hamming_distance`**: Fast Hamming distance with NEON popcount
- **`hamming_batch`**: Batch Hamming distance computation
- **`quantizer_from_vectors`**: Threshold computation from training data

**Key metrics:** Compression throughput, distance computation speedup vs f32

**Use case:** Measure 32x compression impact, validate memory savings

**Performance targets:**
- Single vector < 2μs for 1024-dim
- Hamming distance < 10ns for 1024-bit
- Parallel should be 4-8x faster than sequential

**Sample Results:**
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

### 3. `clustering_bench.rs` - K-Means Performance (Slow, ~10-15min)
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

### 4. `index_bench.rs` - Hierarchical Index (Very Slow, ~15-20min)
**What it measures:** Hierarchical clustered index performance

- **`index_build`**: Index construction time (1K-50K vectors)
- **`index_search_varying_k`**: Query latency for k=1,10,50,100 neighbors
- **`index_search_varying_probes`**: Trade-off between accuracy (probes) and speed
- **`index_search_varying_rerank`**: Impact of rerank factor on precision/recall
- **`index_search_dimensions`**: Query performance across dimensions (128-1536)

**Key metrics:** Build time, query latency (ms), queries per second (QPS)

**Use case:** Tune index parameters (branching, max_leaf, probes, rerank)

**Performance targets:**
- Build 10K vectors in < 2 seconds
- Search latency < 100μs for k=10
- Probes=2 should be ~2x slower than probes=1

---

### 5. `e2e_bench.rs` - End-to-End Workflows (Slow, ~10-15min)
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

### Example Outputs

#### quick_recall_check (8 seconds)
```
Config          Recall@10    Latency(μs)  Build(ms)
-------------------------------------------------------
default         15.0%        45.2         5
tuned           23.0%        108.7        7
high_recall     25.0%        133.4        9
```

#### speed_fast (70 seconds)
```
speed_fast/low_latency     time: [21.7 μs 21.7 μs 21.8 μs]
speed_fast/balanced        time: [90.4 μs 91.6 μs 92.9 μs]
speed_fast/high_recall     time: [104.1 μs 104.6 μs 105.1 μs]
speed_fast/thorough        time: [108.8 μs 110.2 μs 112.3 μs]
```

#### recall_proper (3 minutes)
```
--- Scenario: In-Dataset ---
p1_r2          11.5%        27.2μs
p2_r3          13.0%        108.1μs
p3_r5          13.6%        119.3μs

--- Scenario: Random ---
p1_r2          1.0%         27.2μs
p5_r5          5.1%         160.7μs
```

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

cargo bench --bench index_bench -- search_varying_k/10

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



## Interpreting Results for Optimization

### Memory Bandwidth Bound
If throughput plateaus ~100-120 GiB/s → hitting DRAM bandwidth limit (M1/M2)

### CPU Bound
If throughput scales with vector dimension → CPU computation dominant

### Cache Effects
Test with increasing data sizes to find L1/L2/L3 cache limits

### SIMD Efficiency
Compare scalar vs NEON → should see 4x speedup for f32 operations

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

## Advanced: Custom Metrics

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

## Current Performance (as of latest run)

### Speed (5K vectors, 128d)
- Low latency: **21.7μs**
- Balanced: **91.6μs**
- High recall: **104.6μs**

### Recall (1K vectors, in-dataset)
- Default (p=2,r=3): **15%**
- Tuned (p=3,r=3): **23%**
- High recall (p=5,r=3): **25%**

**Note:** Low recall is due to default parameters. See recall documentation for tuning guidance.
