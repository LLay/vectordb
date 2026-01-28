# Building a Vector Database from Scratch: Getting Started Guide

## Overview

This guide will help you build a simplified version of turbopuffer's architecture. By the end of Week 1-2, you'll have:
- A working development environment
- Basic k-means clustering implementation
- Brute-force search within clusters
- Benchmarking infrastructure

---

## Part 1: Choosing Your Language

### Recommended: Rust

Rust is ideal for this project because:
- Zero-cost abstractions for SIMD intrinsics
- No garbage collector (predictable latency)
- Excellent tooling for profiling and benchmarking
- Great ecosystem (`ndarray`, `rayon`, `memmap2`)

---

## Part 2: Hosting Options

### Option A: Local Development (Recommended to Start)

**Minimum specs:**
- 16GB RAM
- 4+ cores
- Any modern x86-64 CPU with AVX2 (most CPUs from 2013+)

**Check your CPU features:**
```bash
# Linux
cat /proc/cpuinfo | grep -E "avx|sse|popcnt" | head -1

# macOS (Apple Silicon won't have AVX, but has NEON)
sysctl -a | grep cpu.features
```

Start local to iterate quickly. You don't need cloud resources until you're testing at scale.

### Option B: Cloud VMs for Serious Testing

When you need more power or specific hardware:

| Provider | Instance Type | vCPUs | RAM | NVMe | Cost/hr | Best For |
|----------|--------------|-------|-----|------|---------|----------|
| **Hetzner** | AX102 | 32 | 128GB | 2x1.9TB | ~$0.15 | Best value, EU |
| **AWS** | i4i.2xlarge | 8 | 64GB | 1x1.8TB | ~$0.62 | NVMe testing |
| **AWS** | c7i.4xlarge | 16 | 32GB | - | ~$0.71 | AVX-512 (Sapphire Rapids) |
| **GCP** | n2d-standard-16 | 16 | 64GB | - | ~$0.67 | AMD EPYC |
| **Vultr** | Bare Metal | 24 | 256GB | 2x960GB | ~$0.35 | Bare metal access |

**Recommendation:** Start with Hetzner or Vultr for cost efficiency. Use AWS/GCP spot instances for burst testing.

### Option C: Bare Metal (Advanced)

For serious performance work, consider:
- **Hetzner Auction** - Cheap dedicated servers (~€30-50/month)
- **OVH Bare Metal** - Good EU options
- Your own hardware if available

Bare metal gives you:
- No hypervisor overhead
- Full NUMA control
- Direct NVMe access without virtualization

---

## Part 3: Project Setup

### Step 1: Create the Project

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Create project
cargo new vectordb --name vectordb
cd vectordb

# Add nightly toolchain for SIMD intrinsics
rustup override set nightly
```

### Step 2: Configure Cargo.toml

```toml
[package]
name = "vectordb"
version = "0.1.0"
edition = "2021"

[dependencies]
# Linear algebra
ndarray = { version = "0.16", features = ["rayon"] }
ndarray-rand = "0.15"

# Parallelism
rayon = "1.10"

# Random number generation
rand = "0.8"
rand_distr = "0.4"

# Memory mapping for disk-backed vectors
memmap2 = "0.9"

# Serialization
bincode = "1.3"
serde = { version = "1.0", features = ["derive"] }

# Benchmarking and profiling
criterion = { version = "0.5", features = ["html_reports"] }

# CLI for testing
clap = { version = "4.4", features = ["derive"] }

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
proptest = "1.4"  # Property-based testing

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"

[profile.bench]
inherits = "release"
debug = true  # For profiling

[[bench]]
name = "distance_bench"
harness = false
```

### Step 3: Configure for SIMD

Create `.cargo/config.toml`:

```toml
[build]
# Enable CPU-specific optimizations
rustflags = ["-C", "target-cpu=native"]

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native", "-C", "target-feature=+avx2,+fma,+popcnt"]
```

### Step 4: Project Structure

```
vectordb/
├── Cargo.toml
├── .cargo/
│   └── config.toml
├── src/
│   ├── main.rs           # CLI entry point
│   ├── lib.rs            # Library root
│   ├── vector.rs         # Vector types and operations
│   ├── distance/
│   │   ├── mod.rs
│   │   ├── scalar.rs     # Baseline scalar implementation
│   │   ├── simd_avx2.rs  # AVX2 optimized
│   │   └── simd_avx512.rs # AVX-512 optimized
│   ├── index/
│   │   ├── mod.rs
│   │   ├── flat.rs       # Brute force (baseline)
│   │   └── clustered.rs  # K-means clustered index
│   ├── clustering/
│   │   ├── mod.rs
│   │   └── kmeans.rs
│   └── storage/
│       ├── mod.rs
│       ├── memory.rs     # In-memory storage
│       └── mmap.rs       # Memory-mapped storage
├── benches/
│   └── distance_bench.rs
└── tests/
    └── integration_tests.rs
```

---

## Part 4: Writing Your First Distance Kernel

### Step 1: Scalar Baseline (src/distance/scalar.rs)

Always start with a correct, simple implementation:

```rust
//! Scalar (non-SIMD) distance functions - baseline for correctness testing

/// Euclidean (L2) distance squared between two vectors
/// 
/// We return squared distance to avoid the sqrt, which is expensive
/// and unnecessary for comparisons (sqrt is monotonic).
#[inline]
pub fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Dot product of two vectors
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Cosine similarity between two vectors
/// Returns value in [-1, 1] where 1 = identical direction
#[inline]
pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_scalar(a, b);
    let norm_a = dot_product_scalar(a, a).sqrt();
    let norm_b = dot_product_scalar(b, b).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

/// Cosine distance = 1 - cosine_similarity
/// Returns value in [0, 2] where 0 = identical direction
#[inline]
pub fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_squared_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(l2_squared_scalar(&v, &v), 0.0);
    }
    
    #[test]
    fn test_l2_squared_known() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 2.0];
        assert_eq!(l2_squared_scalar(&a, &b), 9.0); // 1 + 4 + 4
    }
    
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product_scalar(&a, &b), 32.0); // 4 + 10 + 18
    }
    
    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity_scalar(&v, &v) - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity_scalar(&a, &b).abs() < 1e-6);
    }
}
```

### Step 2: AVX2 SIMD Implementation (src/distance/simd_avx2.rs)

Now let's write an AVX2-optimized version:

```rust
//! AVX2 SIMD-optimized distance functions
//! 
//! AVX2 provides 256-bit registers, allowing us to process 8 f32s at once.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// L2 squared distance using AVX2
/// 
/// Processes 8 floats per iteration using 256-bit registers.
/// Falls back to scalar for the remainder.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let n = a.len();
    let mut i = 0;
    
    // Accumulator for 8 partial sums
    let mut sum = _mm256_setzero_ps();
    
    // Process 8 elements at a time
    while i + 8 <= n {
        // Load 8 floats from each vector
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        
        // Compute difference
        let diff = _mm256_sub_ps(va, vb);
        
        // Square and accumulate using FMA: sum = sum + diff * diff
        sum = _mm256_fmadd_ps(diff, diff, sum);
        
        i += 8;
    }
    
    // Horizontal sum of the 8 partial sums
    // sum = [s0, s1, s2, s3, s4, s5, s6, s7]
    
    // Add high 128 bits to low 128 bits
    let high = _mm256_extractf128_ps(sum, 1);
    let low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(high, low);
    // sum128 = [s0+s4, s1+s5, s2+s6, s3+s7]
    
    // Horizontal add within 128-bit register
    let sum64 = _mm_hadd_ps(sum128, sum128);
    // sum64 = [s0+s4+s1+s5, s2+s6+s3+s7, ...]
    
    let sum32 = _mm_hadd_ps(sum64, sum64);
    // sum32 = [total, total, ...]
    
    let mut result = _mm_cvtss_f32(sum32);
    
    // Handle remainder with scalar code
    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }
    
    result
}

/// Dot product using AVX2 with FMA
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let n = a.len();
    let mut i = 0;
    
    let mut sum = _mm256_setzero_ps();
    
    // Process 8 elements at a time
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        
        // FMA: sum = sum + va * vb
        sum = _mm256_fmadd_ps(va, vb, sum);
        
        i += 8;
    }
    
    // Horizontal sum
    let high = _mm256_extractf128_ps(sum, 1);
    let low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(high, low);
    let sum64 = _mm_hadd_ps(sum128, sum128);
    let sum32 = _mm_hadd_ps(sum64, sum64);
    
    let mut result = _mm_cvtss_f32(sum32);
    
    // Remainder
    while i < n {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

/// Optimized version that processes 32 elements per iteration
/// Uses 4 accumulators to hide FMA latency (4 cycles on modern CPUs)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_avx2_unrolled(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let n = a.len();
    let mut i = 0;
    
    // 4 independent accumulators to utilize instruction-level parallelism
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    
    // Process 32 elements at a time (4 x 8)
    while i + 32 <= n {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
        
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
        
        let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);
        
        let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);
        
        i += 32;
    }
    
    // Combine accumulators
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    
    // Horizontal sum
    let high = _mm256_extractf128_ps(sum0, 1);
    let low = _mm256_castps256_ps128(sum0);
    let sum128 = _mm_add_ps(high, low);
    let sum64 = _mm_hadd_ps(sum128, sum128);
    let sum32 = _mm_hadd_ps(sum64, sum64);
    
    let mut result = _mm_cvtss_f32(sum32);
    
    // Handle 8-element chunks
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(va, vb);
        
        let high = _mm256_extractf128_ps(prod, 1);
        let low = _mm256_castps256_ps128(prod);
        let sum128 = _mm_add_ps(high, low);
        let sum64 = _mm_hadd_ps(sum128, sum128);
        let sum32 = _mm_hadd_ps(sum64, sum64);
        result += _mm_cvtss_f32(sum32);
        
        i += 8;
    }
    
    // Remainder
    while i < n {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

// Safe wrappers that check for CPU support at runtime
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_squared_avx2(a, b) };
        }
    }
    
    // Fallback to scalar
    crate::distance::scalar::l2_squared_scalar(a, b)
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2_unrolled(a, b) };
        }
    }
    
    crate::distance::scalar::dot_product_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::scalar;
    
    #[test]
    fn test_l2_matches_scalar() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1) + 0.5).collect();
        
        let scalar_result = scalar::l2_squared_scalar(&a, &b);
        let simd_result = l2_squared(&a, &b);
        
        assert!(
            (scalar_result - simd_result).abs() < 1e-3,
            "Results differ: scalar={}, simd={}",
            scalar_result,
            simd_result
        );
    }
    
    #[test]
    fn test_dot_matches_scalar() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001) + 0.1).collect();
        
        let scalar_result = scalar::dot_product_scalar(&a, &b);
        let simd_result = dot_product(&a, &b);
        
        assert!(
            (scalar_result - simd_result).abs() / scalar_result.abs() < 1e-5,
            "Results differ: scalar={}, simd={}",
            scalar_result,
            simd_result
        );
    }
    
    #[test]
    fn test_various_lengths() {
        // Test non-aligned lengths
        for len in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1023, 1024, 1025] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 2.0).collect();
            
            let scalar_result = scalar::dot_product_scalar(&a, &b);
            let simd_result = dot_product(&a, &b);
            
            let relative_error = if scalar_result.abs() > 1e-10 {
                (scalar_result - simd_result).abs() / scalar_result.abs()
            } else {
                (scalar_result - simd_result).abs()
            };
            
            assert!(
                relative_error < 1e-4,
                "Length {}: scalar={}, simd={}, error={}",
                len,
                scalar_result,
                simd_result,
                relative_error
            );
        }
    }
}
```

### Step 3: Distance Module (src/distance/mod.rs)

```rust
//! Distance calculation implementations
//! 
//! This module provides multiple implementations of vector distance functions,
//! from simple scalar baselines to highly optimized SIMD versions.

pub mod scalar;
pub mod simd_avx2;

#[cfg(target_arch = "x86_64")]
pub mod simd_avx512;

/// Distance metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance
    L2,
    /// Cosine distance (1 - cosine_similarity)
    Cosine,
    /// Dot product (for normalized vectors, equivalent to cosine)
    DotProduct,
}

/// Compute distance between two vectors using the specified metric
/// 
/// Automatically dispatches to the fastest available implementation.
#[inline]
pub fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::L2 => simd_avx2::l2_squared(a, b),
        DistanceMetric::DotProduct => -simd_avx2::dot_product(a, b), // Negate so smaller = more similar
        DistanceMetric::Cosine => {
            let dot = simd_avx2::dot_product(a, b);
            let norm_a = simd_avx2::dot_product(a, a).sqrt();
            let norm_b = simd_avx2::dot_product(b, b).sqrt();
            1.0 - (dot / (norm_a * norm_b))
        }
    }
}

/// Batch distance computation - compute distances from query to many vectors
/// 
/// Returns vector of (index, distance) pairs, unsorted.
pub fn batch_distances(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, distance(query, v, metric)))
        .collect()
}

/// Parallel batch distance computation using rayon
pub fn batch_distances_parallel(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;
    
    vectors
        .par_iter()
        .enumerate()
        .map(|(i, v)| (i, distance(query, v, metric)))
        .collect()
}
```

---

## Part 5: Benchmarking Infrastructure

### Create benches/distance_bench.rs

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rand::Rng;

// Import your distance functions
use vectordb::distance::{scalar, simd_avx2};

fn generate_random_vectors(n: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    (a, b)
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    
    for dim in [128, 256, 512, 768, 1024, 1536, 2048].iter() {
        let (a, b) = generate_random_vectors(1, *dim);
        
        group.throughput(Throughput::Bytes((*dim * 4 * 2) as u64)); // 2 vectors, f32 = 4 bytes
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bencher, _| {
            bencher.iter(|| scalar::dot_product_scalar(black_box(&a), black_box(&b)))
        });
        
        group.bench_with_input(BenchmarkId::new("avx2", dim), dim, |bencher, _| {
            bencher.iter(|| simd_avx2::dot_product(black_box(&a), black_box(&b)))
        });
        
        group.bench_with_input(BenchmarkId::new("avx2_unrolled", dim), dim, |bencher, _| {
            bencher.iter(|| unsafe { 
                simd_avx2::dot_product_avx2_unrolled(black_box(&a), black_box(&b)) 
            })
        });
    }
    
    group.finish();
}

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_squared");
    
    for dim in [128, 256, 512, 768, 1024, 1536, 2048].iter() {
        let (a, b) = generate_random_vectors(1, *dim);
        
        group.throughput(Throughput::Bytes((*dim * 4 * 2) as u64));
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bencher, _| {
            bencher.iter(|| scalar::l2_squared_scalar(black_box(&a), black_box(&b)))
        });
        
        group.bench_with_input(BenchmarkId::new("avx2", dim), dim, |bencher, _| {
            bencher.iter(|| simd_avx2::l2_squared(black_box(&a), black_box(&b)))
        });
    }
    
    group.finish();
}

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");
    group.sample_size(50);
    
    let dim = 1024;
    let mut rng = rand::thread_rng();
    
    for num_vectors in [100, 1000, 10000].iter() {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let vectors: Vec<Vec<f32>> = (0..*num_vectors)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::new("sequential", num_vectors),
            &(&query, &vectors),
            |bencher, (q, vecs)| {
                bencher.iter(|| {
                    vectordb::distance::batch_distances(
                        black_box(q),
                        black_box(vecs),
                        vectordb::distance::DistanceMetric::L2,
                    )
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel", num_vectors),
            &(&query, &vectors),
            |bencher, (q, vecs)| {
                bencher.iter(|| {
                    vectordb::distance::batch_distances_parallel(
                        black_box(q),
                        black_box(vecs),
                        vectordb::distance::DistanceMetric::L2,
                    )
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_l2_distance,
    bench_batch_distances,
);
criterion_main!(benches);
```

### Run Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- dot_product

# Generate HTML report
cargo bench -- --save-baseline main

# Compare to baseline after making changes
cargo bench -- --baseline main
```

---

## Part 6: Profiling Your Code

### Using perf (Linux)

```bash
# Install perf
sudo apt install linux-tools-generic

# Build with debug symbols
cargo build --release

# Run with perf
sudo perf record -g ./target/release/vectordb bench

# Analyze
sudo perf report

# Flamegraph
sudo perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

### Using Instruments (macOS)

```bash
# Build with debug symbols
cargo build --release

# Profile with Instruments
xcrun xctrace record --template "Time Profiler" --launch ./target/release/vectordb
```

### Using VTune (Intel CPUs)

Intel VTune is excellent for understanding:
- Cache misses
- Branch mispredictions
- SIMD utilization
- Memory bandwidth

```bash
# Install VTune (free for open source)
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html

vtune -collect hotspots ./target/release/vectordb
vtune -report summary
```

---

## Part 7: First Week Goals Checklist

### Day 1-2: Setup
- [ ] Choose hosting (start local)
- [ ] Set up Rust project with structure
- [ ] Implement scalar distance functions
- [ ] Write basic tests

### Day 3-4: SIMD
- [ ] Implement AVX2 distance functions
- [ ] Verify correctness against scalar
- [ ] Set up benchmarking
- [ ] Measure speedup (expect 4-8x)

### Day 5-7: Basic Index
- [ ] Implement flat (brute-force) index
- [ ] Add insertion and search
- [ ] Benchmark search throughput
- [ ] Profile to find bottlenecks

### Success Metrics for Week 1
- [ ] Scalar vs SIMD: 4-8x speedup
- [ ] Can insert 1M 1024-dim vectors
- [ ] Brute-force search returns correct results
- [ ] Search 1M vectors in < 100ms

---

## Quick Reference: Useful Commands

```bash
# Check CPU features
lscpu | grep -E "avx|sse"

# Check cache sizes
getconf -a | grep CACHE

# Monitor memory bandwidth (needs PCM)
sudo pcm-memory

# Watch CPU utilization per core
htop

# Check NUMA topology
numactl --hardware

# Pin process to specific cores
taskset -c 0-7 ./target/release/vectordb

# Check for memory errors
valgrind --tool=cachegrind ./target/release/vectordb
```

---

## Next Steps After Week 1

Once you have the basics working:

1. **Week 2**: Implement k-means clustering
2. **Week 3-4**: Add binary quantization
3. **Week 5-6**: Memory-mapped storage for SSD tier
4. **Week 7-8**: Build the hierarchical tree structure

Good luck! This is an excellent project for learning systems programming.
