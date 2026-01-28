# CuddleDB - High-Performance Local Vector Database

A side project to see if I achieve 100ms p99 query latency over 100 billion vectors. 

Inspired by turbopuffer's [approximate nearest neighbor](https://turbopuffer.com/blog/ann-v3) architecture.

## 🚀 Quick Start

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Run the demo
cargo run --release --example hierarchical_demo

# Benchmark the index
cargo run --release -- bench --num 10000 --branching 10

# Run SIMD benchmarks
cargo bench --bench distance_bench
```

## 📊 Current Status

### ✅ Implemented
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

### 🚧 To Be Implemented
- **Learned clustering** dynamically choosing cluster count, k-means iterations, and quantization thresholds
- **RaBitQ** implementation for binary quantization
- **Dimensionality reduction** before quantization
- **Quantization error bounds** for reranking
- **Learned rerank_factor** dynamically choosing rerank_factor based on query and cluster statistics
- **Product quantization**
- **Memory-mapped storage**
- **Support vector insertion and deletion** Index updates. Implement [SPFresh](https://dl.acm.org/doi/10.1145/3600006.3613166) style index updates.
- **Document arithmetic intensity** https://modal.com/gpu-glossary/perf/arithmetic-intensity

## 🏗️ Project Structure

```
vectordb/
├── src/
│   ├── lib.rs              # Library root
│   ├── main.rs             # CLI entry point
│   ├── distance/           # Distance calculation implementations
│   │   ├── mod.rs
│   │   └── scalar.rs       # Baseline scalar implementation
│   ├── index/              # Index structures (TODO)
│   ├── clustering/         # Clustering algorithms (TODO)
│   └── storage/            # Storage backends (TODO)
├── benches/
│   └── distance_bench.rs   # Performance benchmarks
├── tests/
│   └── integration_tests.rs
├── Cargo.toml
└── .cargo/
    └── config.toml         # Build configuration for native CPU optimizations
```

## 💡 Usage Examples

### Basic Distance Computation

```rust
use vectordb::{distance, DistanceMetric};

let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];

// Compute L2 distance
let l2_dist = distance(&a, &b, DistanceMetric::L2);

// Compute cosine distance
let cos_dist = distance(&a, &b, DistanceMetric::Cosine);

// Compute dot product
let dot_dist = distance(&a, &b, DistanceMetric::DotProduct);
```

### Batch Distance Computation

```rust
use vectordb::{batch_distances_parallel, DistanceMetric};

let query = vec![1.0, 0.0, 0.0];
let vectors = vec![
    vec![1.0, 0.0, 0.0],
    vec![0.0, 1.0, 0.0],
    vec![0.0, 0.0, 1.0],
];

// Compute distances in parallel
let results = batch_distances_parallel(&query, &vectors, DistanceMetric::L2);

// results: Vec<(usize, f32)> - (index, distance) pairs
```

## 🔬 Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- dot_product

# Run L2 distance benchmarks
cargo bench -- l2_squared

# Run batch processing benchmarks
cargo bench -- batch_distances

# Generate comparison report
cargo bench -- --save-baseline main
# ... make changes ...
cargo bench -- --baseline main
```

Benchmark reports are generated in `target/criterion/` with HTML visualizations.

## 🛠️ Development

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

### Adding SIMD Support

The project is configured to use native CPU features. For Apple Silicon (M1), this enables NEON SIMD instructions automatically when you implement them.

To add NEON optimizations:

1. Create `src/distance/simd_neon.rs`
2. Use `#[cfg(target_arch = "aarch64")]` and ARM NEON intrinsics
3. Add runtime detection with `std::arch::is_aarch64_feature_detected!()`
4. Update `src/distance/mod.rs` to dispatch to NEON implementations

## 📚 Learning Resources

- [Getting Started Guide](../../thinking/vectordb-getting-started.md)
- [Project Ideas](../../thinking/Turbopuffer%20project%20idea.md)
- [Turbopuffer Architecture](https://turbopuffer.com/) (inspiration)
- [ARM NEON Intrinsics Guide](https://developer.arm.com/architectures/instruction-sets/intrinsics/)


## 📄 License

MIT
