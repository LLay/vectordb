# SIFT-1M Benchmark

Industry-standard benchmark for vector search systems. Used by VectorDBBench leaderboard.

## Quick Start

```bash
# 1. Download dataset (256 MB)
cd datasets/sift && ./download.sh && cd ../..

# 2a. Run full benchmark (1M vectors, ~5-10 minutes)
cargo bench --bench sift_benchmark

# 2b. Create subset for fast iteration (10K vectors, ~10 seconds)
cargo run --release --bin create_sift_subset -- 10000
SIFT_SIZE=10000 cargo bench --bench sift_benchmark

# 3. View results
open target/criterion/report/index.html
```

## Dataset

- **1M vectors** (128 dimensions) - full dataset
- **Subsets available**: 10K, 100K vectors for fast iteration
- **10K queries** with ground truth (100 nearest neighbors each)
- Standard `.fvecs`/`.ivecs` format (loaded directly, no conversion needed)

### Creating Subsets for Fast Development

```bash
# Create 10K subset (~10 second benchmarks)
cargo run --release --bin create_sift_subset -- 10000

# Create 100K subset (~1 minute benchmarks)
cargo run --release --bin create_sift_subset -- 100000

# Run benchmark on subset
SIFT_SIZE=10000 cargo bench --bench sift_benchmark
```

Subsets use brute-force ground truth computation (fast enough for <100K vectors).

## Metrics

- **QPS**: Queries per second
- **Latency**: p50, p95, p99 in milliseconds
- **Recall@10**: Accuracy for top-10 results
- **Recall@100**: Accuracy for top-100 results

## Test Configurations

1. **low_latency**: probes=2, rerank=2 (fastest)
2. **balanced**: probes=5, rerank=3 (recommended)
3. **high_recall**: probes=10, rerank=5 (most accurate)

## Files

```
datasets/sift/
├── download.sh              # Download SIFT dataset
├── create_subset.rs         # Create subset datasets
├── loader.rs                # Load .fvecs/.ivecs directly
├── mod.rs                   # Module declaration
├── README.md               # This file
└── sift/                   # Downloaded dataset (gitignored)
    ├── sift_base.fvecs     # 1M vectors (full)
    ├── sift_base_10000.fvecs   # 10K subset (optional)
    ├── sift_base_100000.fvecs  # 100K subset (optional)
    ├── sift_query.fvecs    # 10K queries
    └── sift_groundtruth.ivecs  # Ground truth (for full dataset)
```

## Expected Results

| Config | Latency | QPS | Recall@10 | Recall@100 |
|--------|---------|-----|-----------|------------|
| Low Latency | ~5ms | ~200 | 60-70% | 50-60% |
| Balanced | ~10ms | ~100 | 80-90% | 70-80% |
| High Recall | ~20ms | ~50 | 95%+ | 85-95% |

## Comparing to Other Systems

See [VectorDBBench Leaderboard](https://zilliz.com/vdbbench-leaderboard?dataset=vectorSearch) for comparisons with:
- Milvus
- Qdrant
- Weaviate
- Pinecone
- And more...

## Format Details

### .fvecs (vectors)
```
[dim: i32][vector: f32*dim] repeated
```

### .ivecs (indices)
```
[k: i32][neighbors: i32*k] repeated
```

Both use little-endian byte order.

## Resources

- **Dataset Source**: http://corpus-texmex.irisa.fr/
- **VectorDBBench**: https://github.com/zilliztech/VectorDBBench
- **Leaderboard**: https://zilliz.com/vdbbench-leaderboard
