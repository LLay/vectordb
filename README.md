# CuddleDB - High-Performance Vector Database

A side project to see if I achieve 100ms p99 query latency over 100 billion vectors. 

Inspired by turbopuffer's [approximate nearest neighbor](https://turbopuffer.com/blog/ann-v3) architecture.

## Current Status

540 μs p99 query latency over 1 million vectors with 100% recall on my M1 macbook pro. Full details in [benches/saved/jan_29_1M_vectors/](benches/saved/jan_29_1M_vectors/).

## Next steps:
1. Run on 10 million vectors locally. Reduce latency as much as possible.
2. Run on 1B vectors on an r7iz.8xlarge instance. Requires updating architecture/instruction specific optimizations (e.g. NEON for Apple Silicon, AVX512 for x86_64, etc.
3. Run on 100B vectors on AWS u-18tb1.112xlarge instance. I'm probably can't afford to actually do this.

See [SCALING_GOALS.md](docs/goals/SCALING_GOALS.md) for more details.

## Quick Start

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Run the demo
cargo run --release --example hierarchical_demo

# Benchmark the index
cargo run --release -- bench --num 10000 --branching 10
```

## Current Status

### Done
- **Scalar distance functions**: L2, dot product, cosine similarity
- **Parallel batch distance computation** using Rayon
- **Benchmarking infrastructure** with Criterion
- **CLI tool** for testing and experimentation
- **Comprehensive test suite**
- **SIMD Optimizations** Using NEON for Apple Silicon (M1/M2/M3)
- **K-means clustered index**
- **Binary quantization** (with automatic thresholding) to reduce vector size
- **Two-phase search** Fast filtering with compressed binary vectors, precise ranking with full precision vectors
- **Hierarchical clustering** with adaptive splitting based on cluster density 

### TODO
- **Learned clustering** dynamically choosing cluster count, k-means iterations, and quantization thresholds
- **Learned rerank_factor** dynamically choosing rerank_factor based on query and cluster statistics
- **Tree sizing optimization** choose leaf size and branching factor. This should be done automatically based on the data distribution.
- **RaBitQ** implementation for binary quantization
- **Dimensionality reduction** before quantization
- **Quantization error bounds** for reranking
- **Product quantization**
- **Memory-mapped storage**
- **Support vector insertion and deletion** Index updates. Implement [SPFresh](https://dl.acm.org/doi/10.1145/3600006.3613166) style index updates.
- **Document arithmetic intensity** https://modal.com/gpu-glossary/perf/arithmetic-intensity
- **Early termination during leaf search** Instead of searching all vectors in selected leaves, track best-k distance threshold and skip vectors if binary distance > threshold.

## Project Structure

```
vectordb/
├── src/
│   ├── main.rs                   # CLI entry point
│   ├── distance/                 # Distance metrics
│   ├── index/                    # Index implementation    
│   ├── clustering/               # Clustering algorithms
│   ├── quantization/             # Quantization algorithms
│   ├── storage/                  # Memory-mapped vector storage
│   └── visualization.rs          # Search visualization tools
├── benches/                      # Benchmark suite
├── examples/
│   ├── hierarchical_demo.rs      # Basic index demo
│   ├── observability_demo.rs     # Full observability suite
│   ├── visualize_search.rs       # Search visualization
├── datasets/                     # Pre-generated test datasets
└── Cargo.toml                    # Dependencies & build config
```

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run quick benchmarks (less accurate, faster)
cargo bench -- --quick

# Run specific benchmark
cargo bench -- dot_product

# Generate comparison report
cargo bench -- --save-baseline main
```

Benchmark reports are generated in `target/criterion/` with HTML visualizations.

See more details in [benches/README.md](benches/README.md)

## Development

### Building for Release

```bash
# Full optimizations
cargo build --release

# Check build without optimizations
cargo check
```

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_l2_squared

# With output
cargo test -- --nocapture
```

## Resources

- [Project Idea](docs/Turbopuffer%20project%20idea.md)
- [Turbopuffer Architecture](https://turbopuffer.com/blog/ann-v3) (inspiration)
- [ARM NEON Intrinsics Guide](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

