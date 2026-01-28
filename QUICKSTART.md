# Quick Start Cheatsheet

## 🚀 Essential Commands

```bash
# Development
cargo check              # Fast compile check
cargo build             # Debug build
cargo build --release   # Optimized build
cargo test              # Run all tests
cargo test -- --nocapture  # Run tests with output

# Benchmarking
cargo bench                                     # Run all benchmarks
cargo bench -- dot                              # Run dot product benchmarks only
cargo bench --bench distance_bench -- --quick   # Quick benchmark run

# Running
cargo run -- test --help          # Show CLI help
cargo run --release -- test       # Run with default settings
cargo run --release -- test --dim 2048 --num 50000  # Custom test

# Hierarchical Index
cargo run --example hierarchical_demo                              # Run demo
cargo run --release -- bench                                       # Bench with defaults
cargo run --release -- bench --dim 1024 --num 10000 --branching 10 --max-leaf 150

# Profiling (macOS)
cargo build --release
xcrun xctrace record --template "Time Profiler" --launch ./target/release/vectordb
```

## 📂 Key Files to Know

### Core Implementation
- `src/distance/scalar.rs` - Baseline distance functions (start here!)
- `src/distance/mod.rs` - Distance API and dispatcher
- `src/lib.rs` - Public API exports
- `src/main.rs` - CLI tool

### Testing & Benchmarking
- `tests/integration_tests.rs` - Integration tests
- `benches/distance_bench.rs` - Performance benchmarks

### Configuration
- `Cargo.toml` - Dependencies and build profiles
- `.cargo/config.toml` - CPU-specific optimizations

## 🎯 Development Workflow

### 1. Make Changes
```bash
# Edit files in src/
vim src/distance/scalar.rs
```

### 2. Check Correctness
```bash
cargo test                    # Run all tests
cargo test test_l2_squared    # Run specific test
```

### 3. Measure Performance
```bash
cargo bench -- l2_squared     # Benchmark your changes
```

### 4. Profile (if needed)
```bash
cargo build --release
# Use Instruments on macOS
```

## 🌳 Hierarchical Index

### Key Parameters

- **`branching_factor`** (typical: 10-20): Clusters per level
  - Higher = fewer levels, but more comparisons per level
  - Lower = more levels, but fewer comparisons per level
  
- **`max_leaf_size`** (typical: 100-200): Max vectors per leaf node
  - Smaller = deeper tree, more precise search
  - Larger = shallower tree, faster build, larger leaves
  - **Adaptive**: Dense clusters split deeper automatically
  
- **`probes_per_level`** (typical: 2-5): Clusters to explore at each level
  - Higher = better recall, slower search
  - Lower = faster search, may miss results
  
- **`rerank_factor`** (typical: 3-5): Multiplier for reranking candidates
  - Returns k results, but reranks k×rerank candidates
  - Higher = better accuracy, slightly slower

### Example: Building an Index

```rust
use vectordb::{ClusteredIndex, DistanceMetric};

let index = ClusteredIndex::build(
    vectors,           // Vec<Vec<f32>>
    10,                // branching_factor
    150,               // max_leaf_size (adaptive splitting)
    DistanceMetric::L2,
    20,                // k-means iterations
);

// Search
let results = index.search(
    &query,            // &[f32]
    10,                // k (return top 10)
    2,                 // probes_per_level
    3,                 // rerank_factor (rerank top 30, return 10)
);
```

## 🔧 Common Tasks

### Adding a New Distance Function

1. Add function to `src/distance/scalar.rs`:
```rust
pub fn new_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    // Your implementation
}
```

2. Add test:
```rust
#[test]
fn test_new_distance() {
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    let result = new_distance_scalar(&a, &b);
    assert!(result > 0.0);
}
```

3. Run tests:
```bash
cargo test test_new_distance
```

### Adding a Benchmark

Edit `benches/distance_bench.rs`:
```rust
fn bench_new_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_distance");
    // ... benchmark code ...
    group.finish();
}

criterion_group!(benches, /* existing */, bench_new_distance);
```

## 📊 Performance Targets

### Current (Scalar Only)
- Dot product (1024-dim): ~850ns, ~9 GB/s
- Batch (10k vectors): ~5M distances/sec

### Target with NEON
- Dot product: ~200ns, ~40 GB/s (4-5x speedup)
- Batch: ~20M distances/sec

### Target with Full Optimization
- Advanced loop unrolling
- Prefetching
- Cache optimization
- Target: 8-10x over scalar baseline

## 🐛 Debugging Tips

### Tests Failing?
```bash
cargo test -- --nocapture          # See debug output
RUST_BACKTRACE=1 cargo test        # Get stack traces
```

### Performance Regression?
```bash
# Save baseline
cargo bench -- --save-baseline before

# Make changes, then compare
cargo bench -- --baseline before
```

### Verify Correctness
```bash
# Run tests in release mode (catches optimization bugs)
cargo test --release
```

## 📚 M1-Specific Notes

### CPU Features
```bash
# Check CPU info
sysctl hw.optional
sysctl hw.optional.arm.FEAT_*
sysctl -a | grep machdep.cpu
```

### NEON vs AVX2
- NEON: 128-bit registers (4x f32)
- AVX2: 256-bit registers (8x f32)
- M1 has excellent memory bandwidth to compensate
- Focus on memory access patterns!

### Profiling Tools
- **Instruments**: Best for CPU profiling on macOS
- **Activity Monitor**: Quick CPU/memory check
- **sample**: Command-line profiling: `sample ./target/release/vectordb 10`

### Build Tips
```bash
# Check what optimizations are enabled
cargo rustc --release -- --print cfg | grep -i neon

# Verbose build to see flags
cargo build --release --verbose
```

## 🔗 Helpful Links

- [Rust std::arch docs](https://doc.rust-lang.org/core/arch/index.html)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
