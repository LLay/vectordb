# VectorDB Benchmarking Guide

Complete guide for benchmarking CuddleDB against industry standards using VectorDBBench.

## Quick Start

```bash
# 1. Download SIFT dataset
cd datasets/sift && ./download.sh && cd ../..

# 2. Create 10K subset for fast iteration
cargo run --release --bin create_sift_subset -- 10000

# 3. Run benchmark
SIFT_SIZE=10000 cargo bench --bench sift_benchmark

# 4. View Criterion HTML reports
open target/criterion/report/index.html

# 5. Export results
./scripts/export_results.sh my_results.json
```

## Understanding VectorDBBench

[VectorDBBench](https://github.com/zilliztech/VectorDBBench) is the industry standard for comparing vector database performance. It provides:

- **Standard datasets**: SIFT, Cohere, OpenAI embeddings
- **Standard metrics**: QPS, latency (p50/p99), recall@K
- **Public leaderboard**: Compare with Milvus, Qdrant, Weaviate, etc.
- **Reproducible tests**: Same data, same metrics, fair comparison

### Why Use VectorDBBench?

1. **Credibility**: Industry-recognized benchmark
2. **Comparability**: Apples-to-apples comparison with other systems
3. **Visibility**: Results published on public leaderboard
4. **Standard datasets**: No bias from custom data

## Implementation Plan

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the complete 4-week roadmap.

### Phase 1: SIFT Baseline (Week 1) ✓
- [x] Download SIFT-1M dataset
- [x] Create subset support (10K, 100K)
- [x] Implement VectorDBBench-compatible benchmark
- [x] Measure all required metrics

### Phase 2: Additional Datasets (Week 2)
- [ ] Download Cohere (768d) dataset
- [ ] Download OpenAI (1536d) dataset
- [ ] Test performance across dimensions
- [ ] Identify scaling patterns

### Phase 3: Advanced Analysis (Week 3)
- [ ] Recall vs latency trade-off analysis
- [ ] Batch query optimization
- [ ] Memory profiling
- [ ] Parameter tuning

### Phase 4: Comparison & Submission (Week 4)
- [ ] Compare to leaderboard systems
- [ ] Generate comparison tables
- [ ] Submit results to VectorDBBench
- [ ] Document optimizations

## Running Benchmarks

### Development Workflow (10K subset)

Fast iteration for development:
```bash
# Setup (once)
cd datasets/sift && ./download.sh && cd ../..
cargo run --release --bin create_sift_subset -- 10000

# Iterate quickly (~10 seconds per run)
SIFT_SIZE=10000 cargo bench --bench sift_benchmark
```

### Production Validation (1M full)

Final validation on full dataset:
```bash
# This takes ~10 minutes
cargo bench --bench sift_benchmark

# Results saved to target/criterion/
```

### Other Subsets

For 100K vectors:
```bash
# Create subset (once)
cargo run --release --bin create_sift_subset -- 100000

# Run benchmark (~1 minute)
SIFT_SIZE=100000 cargo bench --bench sift_benchmark
```

## Required Metrics

VectorDBBench requires these metrics for submission:

| Metric | Source | Example |
|--------|--------|---------|
| **QPS** | Manual calculation or Criterion | 6,168 queries/sec |
| **p50 latency** | Manual or Criterion | 0.15 ms |
| **p99 latency** | Manual calculation | 0.34 ms |
| **Recall@10** | Manual (ground truth comparison) | 91.1% |
| **Recall@100** | Manual (ground truth comparison) | 74.5% |
| **Build time** | Manual measurement | 0.48 s |
| **Index size** | File system | 4.88 MB |

Our benchmark reports all of these automatically!

## Understanding the Output

### Console Output

```
╔═══════════════════════════════════════════════════════════════╗
║ balanced configuration (probes=5, rerank=3)                    
╚═══════════════════════════════════════════════════════════════╝

Criterion Performance Metrics:
  (Criterion measures and reports timing here)

Accuracy Metrics (manual calculation):
  Recall@10:  91.1%
  Recall@100: 74.5%
  QPS:        6,168 queries/sec
  p50:        0.15ms
  p95:        0.25ms
  p99:        0.34ms
```

### HTML Reports

Criterion generates beautiful HTML reports:
```bash
open target/criterion/report/index.html
```

Shows:
- Latency distribution (violin plots)
- Throughput over time
- Performance regression detection
- Statistical confidence intervals

## Comparing to VectorDBBench Leaderboard

### Step 1: Run Full Benchmark

```bash
# Run on full 1M dataset
cargo bench --bench sift_benchmark
```

### Step 2: Record Results

Note these metrics from the console output:
- Recall@10 and Recall@100 for each configuration
- QPS, p50, p95, p99 for each configuration
- Build time and index size

### Step 3: Compare to Leaderboard

Visit https://zilliz.com/vdbbench-leaderboard

**Typical Results (SIFT-1M, Recall@10 > 90%):**

| System | QPS | p99 (ms) | Recall@10 | Implementation |
|--------|-----|----------|-----------|----------------|
| Milvus | 1000+ | <10 | 95% | GPU-accelerated HNSW |
| Qdrant | 500+ | <20 | 92% | Rust, HNSW |
| Weaviate | 300+ | <30 | 91% | Go, HNSW |
| **CuddleDB (10K)** | **6,168** | **0.34** | **91.1%** | Hierarchical k-means |
| **CuddleDB (1M est)** | **~100** | **~15** | **~88%** | Hierarchical k-means |

### Step 4: Analyze Trade-offs

**Your advantages:**
- Simpler implementation
- Lower memory footprint
- Better disk-based scaling
- No GPU required

**HNSW advantages:**
- Higher QPS on 1M+ datasets
- More mature optimization
- Better recall at high scale

## Exporting Results

### Automated Export

Use the export script to create VectorDBBench-compatible JSON:

```bash
./scripts/export_results.sh my_results.json
```

This will:
1. Detect your dataset size (10K or 1M)
2. Collect system information
3. Prompt for benchmark metrics
4. Generate JSON in VectorDBBench format

### Manual Export

Create a JSON file with this structure:

```json
{
  "system": "CuddleDB",
  "version": "0.1.0",
  "dataset": "sift-128-euclidean",
  "hardware": {
    "cpu": "Apple M1 Max",
    "cores": 10,
    "ram": "32 GB"
  },
  "results": {
    "balanced": {
      "qps": 6168,
      "latency_p99_ms": 0.34,
      "recall_at_10": 0.911
    }
  }
}
```

## Submitting to VectorDBBench

### Option 1: Official Submission (Public Leaderboard)

1. **Fork the repository**
   ```bash
   # Go to https://github.com/zilliztech/VectorDBBench
   # Click "Fork"
   ```

2. **Add your results**
   ```bash
   git clone https://github.com/YOUR_USERNAME/VectorDBBench
   cd VectorDBBench
   
   # Add your results file
   cp ~/vectordb/my_results.json results/cuddledb_sift_1m.json
   
   # Add system details
   mkdir -p systems/cuddledb
   # Create README.md with implementation details
   ```

3. **Submit Pull Request**
   ```bash
   git add results/cuddledb_sift_1m.json systems/cuddledb/
   git commit -m "Add CuddleDB results for SIFT-1M"
   git push origin main
   
   # Create PR on GitHub
   ```

4. **PR Requirements**
   - Detailed system specs
   - Build instructions
   - Configuration parameters
   - Reproducibility instructions

### Option 2: Independent Comparison (README/Blog)

For internal use or documentation:

1. **Create comparison table**
   ```markdown
   ## Performance Comparison (SIFT-1M)
   
   | System | QPS | p99 | R@10 |
   |--------|-----|-----|------|
   | Qdrant | 500 | 20ms | 92% |
   | CuddleDB | 100 | 15ms | 88% |
   ```

2. **Document methodology**
   - Same dataset (SIFT-1M)
   - Same metrics (QPS, p99, recall@10)
   - Same hardware (specs)
   - Reproducible (instructions)

3. **Add to your README**
   ```bash
   # Update vectordb/README.md with results
   ```

## Tips for Good Results

### 1. Run on Representative Hardware

- Use production-like hardware
- Disable background processes
- Ensure stable CPU frequency
- Monitor temperature throttling

### 2. Warm Up Properly

```bash
# Run a few times to warm up caches
SIFT_SIZE=10000 cargo bench --bench sift_benchmark  # warm up
SIFT_SIZE=10000 cargo bench --bench sift_benchmark  # actual run
```

### 3. Multiple Runs

```bash
# Run 3 times and report median
for i in 1 2 3; do
  cargo bench --bench sift_benchmark >> results_$i.txt
done
```

### 4. Document Configuration

Always document:
- Hardware specs (CPU, RAM, disk)
- OS and version
- Rust version
- Index parameters (branching_factor, target_leaf_size)
- Search parameters (probes, rerank_factor)

## Interpreting Results

### Good Recall (>90%)
If your recall is good, focus on:
- Improving QPS (lower probes, optimize SIMD)
- Reducing latency (cache optimization)
- Scaling to larger datasets

### Low Recall (<90%)
If recall is low, try:
- Increasing `probes` (e.g., 10 → 15)
- Increasing `rerank_factor` (e.g., 3 → 5)
- Adjusting `branching_factor` (e.g., 100 → 150)
- Improving cluster quality

### High Latency
If latency is too high:
- Decrease `probes` (trade recall for speed)
- Optimize SIMD distance calculations
- Profile with `cargo flamegraph`
- Check for memory allocations in hot path

## Additional Resources

- [QUICK_START.md](QUICK_START.md) - Step-by-step getting started
- [CRITERION_GUIDE.md](CRITERION_GUIDE.md) - Understanding Criterion output
- [CRITERION_VECTORDBBENCH_FINAL.md](CRITERION_VECTORDBBENCH_FINAL.md) - Complete explanation
- [VECTORDBBENCH_COMPARISON.md](VECTORDBBENCH_COMPARISON.md) - Detailed comparison guide
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - 4-week roadmap

## External Links

- **VectorDBBench GitHub**: https://github.com/zilliztech/VectorDBBench
- **Leaderboard**: https://zilliz.com/vdbbench-leaderboard
- **SIFT Dataset**: http://corpus-texmex.irisa.fr/
- **Criterion.rs**: https://github.com/bheisler/criterion.rs

## Getting Help

If you have questions:
1. Check the guides above
2. Look at existing VectorDBBench submissions
3. Review Criterion documentation
4. Open an issue in VectorDBBench repo

## Summary

**You have everything you need!**

✓ Standard dataset (SIFT)
✓ All required metrics (QPS, latency, recall)
✓ Criterion integration (timing + HTML reports)
✓ Export script (JSON generation)
✓ Documentation (comparison guide)

Next steps:
1. Run full 1M benchmark
2. Export results
3. Compare to leaderboard
4. Optimize if needed
5. Submit results (optional)

Good luck!
