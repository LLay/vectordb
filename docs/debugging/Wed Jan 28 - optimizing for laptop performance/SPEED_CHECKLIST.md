# Speed Optimization Checklist (Scalable Architecture)

**Goal:** Optimize hierarchical index to run well on laptop (2GB available RAM) and scale to powerful hardware  
**Current:** ~3-10ms p99 → **Laptop Target:** 1-3ms p99 → **Server Target:** 0.2-0.5ms p99

**Philosophy:** Build once, scale everywhere. Same code from laptop to production.

---

## Priority Order (Do in This Sequence)

### 🔴 Priority 1: Quantized Tree Navigation (2-3 hours) → Get to 2-3ms p99

- [ ] **1.1** Add binary quantization to tree centroids
  - Quantize all centroid vectors in the index
  - Store both f32 and binary versions
  - Use Hamming for tree traversal
  - Memory: Minimal increase (~10 MB for tree)
  - Expected: 2-3x faster tree traversal

- [ ] **1.2** Implement Hamming-based tree probing
  - Replace L2 distance with Hamming in `nearest_centroid()`
  - Much faster: 10ns vs 30ns per comparison
  - Especially fast when visiting multiple levels
  - Expected: 1-2ms p99

- [ ] **1.3** Add benchmark for quantized tree
  - Test with 1M vectors, 1024-dim
  - Measure tree traversal time separately
  - Target: <100μs for tree traversal

**Time:** 2-3 hours  
**Impact:** 2-3x speedup on tree operations  
**Complexity:** Medium  
**Scales:** Works at any vector count

---

### 🟡 Priority 2: Parallel Probe Optimization (1-2 hours) → Get to 1-2ms p99

- [ ] **2.1** Parallelize candidate leaf scanning
  - After tree traversal, scan multiple leaves in parallel
  - Use rayon for leaf-level parallelism
  - Expected: 2-4x speedup

- [ ] **2.2** Parallel quantized distance computation
  - Compute Hamming distances in parallel within leaves
  - Chunk size: ~1000 vectors per task
  - Minimal overhead with rayon

- [ ] **2.3** Tune probes_per_level parameter
  - Test probes = 1, 2, 3, 5
  - More probes = better recall but slower
  - Find sweet spot for your use case

**Time:** 1-2 hours  
**Impact:** 2-4x speedup on candidate scanning  
**Complexity:** Low (rayon makes it easy)  
**Scales:** More cores = more speedup

---

### 🟢 Priority 3: Memory-Mapped Storage (1-2 hours) → Reduce cold latency

- [ ] **3.1** Create `MmapVectorStore` (storage/mmap.rs)
  - Memory-map full precision vectors
  - Zero-copy access
  - OS handles caching automatically

- [ ] **3.2** Replace in-memory vectors with mmap
  - Keeps quantized vectors in RAM
  - Full precision vectors on disk (mmap)
  - Reduces RAM from 4GB to ~300MB

- [ ] **3.3** Benchmark mmap performance
  - Compare cache hit/miss latency
  - Should be nearly identical when cached
  - 10-100x slower on cold miss (but rare)

**Time:** 1-2 hours  
**Impact:** Enables scaling beyond RAM  
**Complexity:** Medium  
**Scales:** Essential for large datasets

---

### 🔵 Priority 4: Smart Caching Layer (2 hours) → Improve hot path

- [ ] **4.1** Add LRU cache for full-precision vectors
  - Cache size: Adaptive based on available RAM
  - 2GB available: 400MB cache (~100K vectors)
  - 64GB available: 32GB cache (8M vectors)

- [ ] **4.2** Cache hot tree nodes
  - Cache frequently accessed centroids
  - Reduces tree traversal cost further

- [ ] **4.3** Add cache statistics
  - Track hit rate, miss rate
  - Tune cache size based on workload

**Time:** 2 hours  
**Impact:** 2-5x for hot queries  
**Complexity:** Medium  
**Scales:** Larger RAM = larger cache = better

---

### 🟣 Priority 5: Advanced Parallelization (1-2 hours) → Max throughput

- [ ] **5.1** Batch query optimization
  - Process multiple queries in parallel
  - Amortize tree traversal costs
  - Better CPU utilization

- [ ] **5.2** Async I/O for reranking
  - Prefetch full-precision vectors
  - Overlap I/O with computation
  - Reduces wait time on cache misses

- [ ] **5.3** NUMA-aware allocation (server only)
  - Pin memory to NUMA nodes
  - Reduces cross-socket latency
  - Only matters on multi-socket servers

**Time:** 1-2 hours  
**Impact:** 2-3x throughput  
**Complexity:** Medium-High  
**Scales:** Essential for high QPS

---

### ⚪ Priority 6: SIMD Verification & Tuning (1 hour) → Ensure max performance

- [ ] **6.1** Verify NEON Hamming is used
  - Check assembly or profile
  - Should see `vcnt` instruction
  - ~10ns per comparison target

- [ ] **6.2** Verify NEON L2 is used
  - Should see `vfma` instructions
  - ~30ns per distance (1024-dim)

- [ ] **6.3** Add prefetch hints (optional)
  - Manual prefetch for sequential access
  - Marginal gain (~10-20%)

**Time:** 1 hour  
**Impact:** Verification + 10-20% gain  
**Complexity:** Low  
**Scales:** SIMD works at any scale

---

## Verification Steps

After each priority level:

```bash
# Build and test
cargo build --release
cargo test --release

# Benchmark
cargo bench --bench index_bench -- search

# Profile (macOS)
cargo instruments --release --bench index_bench --template Time
```

### Expected Latency After Each Stage

#### Laptop (16GB RAM, 2GB available)
| Stage | p50 | p99 | Time Invested |
|-------|-----|-----|---------------|
| Baseline | 3ms | 10ms | 0 |
| After P1 (quantized tree) | 2ms | 5ms | 2-3 hrs |
| After P2 (parallel) | 1ms | 3ms | 3-5 hrs |
| After P3 (mmap) | 1ms | 2ms | 4-7 hrs |
| After P4 (cache) | 1ms | 2ms | 6-9 hrs |
| After P5 (advanced) | 1ms | 2ms | 7-11 hrs |
| After P6 (tuning) | 1ms | 2ms | 8-12 hrs |

#### Server (64GB RAM, NVMe)
| Stage | p50 | p99 | Notes |
|-------|-----|-----|-------|
| Same code | 0.3ms | 0.8ms | Just better hardware! |
| + All optimizations | 0.2ms | 0.5ms | Production ready |

**Total time investment: ~8-12 hours for 5x laptop speedup + future scalability**

---

## Implementation Order

### Day 1: Foundation (Priorities 1-2)
1. Morning: Quantize tree centroids (P1.1)
2. Afternoon: Hamming tree traversal (P1.2, P1.3)
3. Evening: Add parallel probes (P2)

**Expected: 1-3ms p99 on laptop**

### Day 2: Storage (Priority 3)
1. Morning: Implement mmap storage (P3.1)
2. Afternoon: Integrate with index (P3.2, P3.3)

**Expected: Reduced RAM usage, same performance**

### Day 3: Optimization (Priorities 4-6)
1. Morning: Add caching (P4)
2. Afternoon: Advanced parallelization (P5)
3. Evening: Verify and tune (P6)

**Expected: 1-2ms p99, optimized for scale**

---

## Quick Start (Get Moving Fast)

### Step 1: Add quantized centroids to hierarchical index (1 hour)

```rust
// In src/index/hierarchical.rs

// Add to Node struct
pub struct Node {
    centroid: Vec<f32>,
    centroid_binary: BinaryVector,  // ← Add this
    children: Vec<Node>,
    vectors: Vec<Vec<f32>>,
    // ...
}

// Update build function
impl ClusteredIndex {
    pub fn build(vectors: Vec<Vec<f32>>, ...) -> Self {
        let quantizer = BinaryQuantizer::from_vectors(&vectors);
        
        // Build tree with quantized centroids
        let root = Self::build_tree_quantized(
            vectors,
            &quantizer,
            branching,
            max_leaf,
            metric,
            max_iters,
            0
        );
        
        Self { root, quantizer }
    }
    
    fn build_tree_quantized(...) -> Node {
        // Run k-means
        let (kmeans, assignment) = KMeans::fit(...);
        
        // Quantize centroids
        let centroids_binary = kmeans.centroids.iter()
            .map(|c| quantizer.quantize(c))
            .collect();
        
        // Build nodes with both representations
        // ...
    }
}
```

### Step 2: Use Hamming for tree traversal (30 min)

```rust
// Update search to use Hamming
pub fn search(&self, query: &[f32], k: usize, probes: usize) -> Vec<(usize, f32)> {
    let query_binary = self.quantizer.quantize(query);
    
    // Traverse tree with Hamming distance
    let candidates = self.probe_tree(&query_binary, probes);
    
    // Rerank with full precision
    self.rerank(query, candidates, k)
}

fn probe_tree(&self, query_binary: &BinaryVector, probes: usize) -> Vec<usize> {
    let mut current_nodes = vec![&self.root];
    let mut candidates = Vec::new();
    
    while !current_nodes.is_empty() {
        let mut next_level = Vec::new();
        
        for node in current_nodes {
            if node.is_leaf() {
                candidates.extend(&node.vector_ids);
            } else {
                // Find nearest children using Hamming
                let mut child_dists: Vec<_> = node.children.iter()
                    .map(|child| {
                        let dist = hamming_distance(
                            query_binary, 
                            &child.centroid_binary
                        );
                        (child, dist)
                    })
                    .collect();
                
                child_dists.sort_by_key(|(_, d)| *d);
                
                // Take top 'probes' children
                for (child, _) in child_dists.iter().take(probes) {
                    next_level.push(*child);
                }
            }
        }
        
        current_nodes = next_level;
    }
    
    candidates
}
```

### Step 3: Parallelize leaf scanning (15 min)

```rust
use rayon::prelude::*;

fn scan_candidates_parallel(
    &self,
    query_binary: &BinaryVector,
    candidate_ids: &[usize],
) -> Vec<(usize, u32)> {
    candidate_ids.par_iter()
        .map(|&idx| {
            let dist = hamming_distance(
                query_binary,
                &self.quantized_vectors[idx]
            );
            (idx, dist)
        })
        .collect()
}
```

### Step 4: Test it (5 min)

```bash
cargo test --release
cargo bench --bench index_bench
```

**You're now 2-3x faster!**

---

## Success Metrics

Track these after each optimization:

| Metric | Laptop Target | Server Target | How to Measure |
|--------|---------------|---------------|----------------|
| **p50 latency** | 1-2ms | <0.5ms | `cargo bench` |
| **p99 latency** | 2-3ms | <1ms | `scale_demo` with 100 queries |
| **Throughput** | 500 QPS | 2000+ QPS | Batch queries |
| **Memory** | <2GB | <32GB | Activity Monitor |
| **CPU usage** | 60-80% | 80-100% | htop during query |
| **Cache hit rate** | >80% | >90% | Custom metrics |

---

## Troubleshooting

### "Not hitting speed targets on laptop"
1. Check if you have 2GB actually available (close other apps)
2. Verify Hamming is being used (not L2 in tree)
3. Check parallel execution (`htop` should show all cores busy)
4. Profile with Instruments to find bottleneck

### "Memory over 2GB"
1. Verify only quantized vectors in RAM
2. Full precision should be mmap'd
3. Reduce cache size if needed
4. Check for memory leaks

### "High variance in latency"
1. Cold cache on first queries (expected)
2. Run warm-up queries before measuring
3. OS might be swapping (need more free RAM)
4. SSD might be slow (check with disk benchmark)

---

## Hardware Scaling Reference

### What happens with better hardware (same code):

| Component | Laptop | Workstation | Server |
|-----------|--------|-------------|--------|
| **RAM** | 16GB (2GB free) | 64GB (16GB free) | 256GB (64GB free) |
| **Cache size** | 400MB | 14GB | 60GB |
| **Vectors cached** | 100K | 3.5M | 15M |
| **Storage** | SSD | NVMe Gen3 | NVMe Gen4 |
| **Cold read** | 1-5ms | 0.5-1ms | 0.1-0.5ms |
| **Cores** | 10 (8+2) | 16-32 | 64-128 |
| **Parallel speedup** | 8x | 16-32x | 64-128x |
| **Query latency** | 1-3ms | 0.3-1ms | 0.2-0.5ms |
| **Max vectors** | 50M | 500M | 5B+ |

**Same code = 10-50x better performance on better hardware!**

---

## Next Steps

1. Read `SPEED_OPTIMIZATION.md` for detailed rationale
2. Implement Priority 1 (2-3 hours)
3. Verify latency improved
4. Continue with Priority 2
5. Test on different hardware configs
6. Celebrate scalable architecture!

**Start now → Get to 2-3ms on laptop, <1ms on server with same code!**
