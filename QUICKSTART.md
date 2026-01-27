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

## 🎓 Learning Path

1. **Week 1**: Understand scalar implementations
   - Read `src/distance/scalar.rs`
   - Run benchmarks, understand performance
   - Try modifying algorithms

2. **Week 2**: Add NEON SIMD
   - Study ARM NEON intrinsics
   - Implement dot product with NEON
   - Measure speedup

3. **Week 3**: Build index structures
   - Implement flat index
   - Add k-means clustering
   - Compare performance

4. **Week 4+**: Advanced topics
   - Binary quantization
   - Memory-mapped storage
   - Multi-level caching

## 🔗 Helpful Links

- [Rust std::arch docs](https://doc.rust-lang.org/core/arch/index.html)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
