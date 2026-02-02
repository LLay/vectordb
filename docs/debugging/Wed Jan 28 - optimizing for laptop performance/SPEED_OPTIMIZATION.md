# Speed Optimization: Scalable Architecture for 1M+ Vectors

**Current Target:** 1M vectors on laptop (16GB total RAM, 2GB available for app)  
**Future Target:** Scale to billions on powerful hardware  
**Goal:** Fast queries that scale from laptop to production  
**Constraint:** Optimizations must work at both small and large scale

---

## Current Baseline Performance

**1M vectors, 1024-dim, basic index:**
- Build time: ~10-30 seconds
- Query latency (p99): 2-5ms
- Memory: ~5GB (4GB vectors + 1GB index)

**We can do much better!**

---

## Current Hardware (Laptop)

**System:** 16GB total RAM, 10-core Apple Silicon  
**Available for App:** ~2-4GB (other applications running)  
**Storage:** 280GB available

## Memory Budget (2GB Minimum Available)

```
┌─────────────────────────────────────┐
│ 2GB RAM Allocation (Minimum)        │
├─────────────────────────────────────┤
│ Index structure: 150 MB             │  Hierarchical tree
│ Quantized vectors: 128 MB           │  1M × 128 bytes (binary)
│ Working memory: 200 MB              │  Query processing
│ Vector cache: 500 MB                │  Hot full-precision vectors
│ OS buffers: 1000 MB                 │  File cache, mmap
└─────────────────────────────────────┘

With 4GB available:
├─────────────────────────────────────┤
│ Index structure: 150 MB             │
│ Quantized vectors: 128 MB           │
│ Vector cache: 2 GB                  │  ← Much larger cache
│ Working memory: 500 MB              │
│ OS buffers: 1.2 GB                  │
└─────────────────────────────────────┘
```

**Disk:**
- Full precision vectors: 4 GB (1M × 4KB)
- Quantized vectors backup: 128 MB
- Index metadata: ~50 MB
- Available: 276 GB

---

## Optimization Roadmap (Prioritized for Scale)

### Priority 1: Optimize Hierarchical Index (Scales to Billions)
**Impact:** 2-5x speedup, works at any scale  
**Effort:** Medium  
**Target latency:** 2-10ms on laptop, <1ms on powerful hardware

**Why hierarchical?**
- Scales from 1M to 1B+ vectors
- Logarithmic search time (doesn't grow linearly)
- Works with limited RAM (only tree in memory)
- Industry standard (FAISS, Pinecone, Weaviate all use variants)

**Optimizations:**
1. **Keep quantized index in RAM** (128 MB for 1M vectors)
   - Binary centroids for each tree node
   - Fast Hamming distance for tree traversal
   - Scales: 1B vectors = 12.8 GB quantized (fits on server)

2. **Parallel tree probes**
   - Explore multiple branches simultaneously
   - Use all CPU cores for candidate scanning
   - Scales: More cores = more parallelism

3. **Smart caching strategy**
   - LRU cache for hot vectors
   - Adaptive cache size based on available RAM
   - Scales: Larger RAM = larger cache, better hit rate

**Implementation:**
```rust
pub struct ScalableIndex {
    tree: HierarchicalTree,            // Tree structure in RAM
    quantized: Vec<BinaryVector>,      // Quantized vectors (RAM or mmap)
    full_precision: VectorStore,       // Full precision (mmap)
    quantizer: BinaryQuantizer,
    cache: LruCache<usize, Vec<f32>>,  // Adaptive cache
}

impl ScalableIndex {
    pub fn search(&self, query: &[f32], k: usize, probes: usize) -> Vec<(usize, f32)> {
        // Phase 1: Tree traversal with Hamming distance
        let query_bin = self.quantizer.quantize(query);
        let candidate_leaves = self.tree.probe(
            &query_bin, 
            probes,
            |a, b| hamming_distance(a, b)  // Fast!
        );
        
        // Phase 2: Parallel scan of candidate leaves
        let candidates = candidate_leaves.par_iter()
            .flat_map(|leaf| {
                leaf.vectors.iter().enumerate().map(|(i, vec)| {
                    let dist = hamming_distance(&query_bin, vec);
                    (leaf.start_idx + i, dist)
                })
            })
            .collect();
        
        // Phase 3: Rerank with full precision (with caching)
        let mut results = self.rerank_with_cache(&query, candidates, k);
        results.truncate(k);
        results
    }
}
```

**This approach scales:**
- 1M vectors: Tree depth ~4, scan ~2K candidates
- 100M vectors: Tree depth ~6, scan ~20K candidates  
- 1B vectors: Tree depth ~7, scan ~50K candidates
- Query time grows logarithmically, not linearly!

---

### Priority 2: SIMD Optimization (Already Good!)
**Impact:** Already implemented  
**Effort:** Low (verify)  
**Target:** Maintain 4x SIMD speedup

**Current status:**
NEON dot product: ~18ns (1024-dim)  
NEON L2 squared: ~30ns (1024-dim)  
NEON Hamming: ~10ns (1024-bit)

**Additional optimizations:**
1. **Loop unrolling** (already have 4x unroll)
2. **Prefetching:** Add manual prefetch hints
3. **Cache alignment:** Align vectors to 64-byte boundaries

```rust
// Add prefetching to batch operations
#[cfg(target_arch = "aarch64")]
unsafe fn prefetch_vectors(ptrs: &[*const f32]) {
    use std::arch::aarch64::_prefetch;
    for &ptr in ptrs.iter() {
        _prefetch(ptr as *const i8, 0, 3); // Read, high locality
    }
}
```

---

### Priority 3: Parallel Execution (Easy Win)
**Impact:** 5-8x speedup on 10 cores  
**Effort:** Low (already have rayon)  
**Target:** Utilize all cores

**Implementation:**
```rust
use rayon::prelude::*;

// Parallel Hamming distance computation
let candidates: Vec<(usize, u32)> = (0..self.quantized.len())
    .into_par_iter()
    .with_min_len(10_000) // Chunk size
    .map(|i| {
        let dist = hamming_distance(&query_bin, &self.quantized[i]);
        (i, dist)
    })
    .collect();
```

**Why it matters:**
- Single-threaded: 1M vectors × 10ns = 10ms
- 10 cores: 10ms / 8 = **1.2ms** (accounting for overhead)
- With SIMD + parallel: **0.3-0.5ms**

---

### Priority 4: Memory-Mapped Full Precision Vectors
**Impact:** Zero copy I/O, fast reranking  
**Effort:** Medium  
**Target:** 0.1-0.2ms for rerank phase

**Why mmap beats reading:**
- OS handles paging automatically
- Zero-copy access
- Hot pages stay in cache
- Sequential access is ~500 MB/s on SSD

**Implementation:**
```rust
use memmap2::MmapOptions;
use std::fs::File;

pub struct VectorStore {
    mmap: Mmap,
    dimension: usize,
    count: usize,
}

impl VectorStore {
    pub fn new(path: &str, dimension: usize, count: usize) -> Self {
        let file = File::open(path).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        Self { mmap, dimension, count }
    }
    
    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        let offset = idx * self.dimension;
        let ptr = self.mmap.as_ptr() as *const f32;
        unsafe {
            std::slice::from_raw_parts(
                ptr.add(offset),
                self.dimension
            )
        }
    }
}
```

**Expected performance:**
- Cache hit (hot vector): ~50ns access
- Cache miss (cold vector): ~500ns-1μs (one SSD page fetch)
- Reranking 50 candidates: 50 × 30ns + 10 × 1μs = **11.5μs**

---

### Priority 5: Batch Optimization (Production-Ready)
**Impact:** 3-5x throughput for batch queries  
**Effort:** Low  
**Target:** Scales to handle high QPS

**Implementation:**
```rust
pub fn batch_search_parallel(
    &self, 
    queries: &[Vec<f32>], 
    k: usize
) -> Vec<Vec<(usize, f32)>> {
    queries.par_iter()
        .with_min_len(10)
        .map(|q| self.search(q, k, 2))
        .collect()
}
```

**Why it matters:**
- Better CPU utilization
- Amortized tree traversal costs
- OS can optimize I/O patterns
- Scales: Handle 1000s of queries/second

---

### Priority 6: Hot Path Cache (Advanced)
**Impact:** 2-5x speedup for repeated queries  
**Effort:** Medium  
**Target:** <100μs for cached queries

**Implementation:**
```rust
use lru::LruCache;
use std::hash::{Hash, Hasher};

pub struct CachedIndex {
    inner: FlatIndex,
    query_cache: Mutex<LruCache<QueryHash, Vec<(usize, f32)>>>,
    vector_cache: Mutex<LruCache<usize, Vec<f32>>>,
}

impl CachedIndex {
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let query_hash = hash_query(query, k);
        
        // Check query cache
        if let Some(cached) = self.query_cache.lock().get(&query_hash) {
            return cached.clone(); // ~1μs
        }
        
        // Execute query
        let results = self.inner.search(query, k);
        
        // Cache result
        self.query_cache.lock().put(query_hash, results.clone());
        results
    }
}
```

**Cache budget (500 MB for vectors):**
- Can cache 125K full-precision vectors (1024-dim)
- 12.5% of dataset → high hit rate for skewed access

---

### Priority 7: Batch Query Optimization (Bonus)
**Impact:** 2-3x throughput for batch queries  
**Effort:** Medium  
**Target:** 2000+ QPS

**Implementation:**
```rust
pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) 
    -> Vec<Vec<(usize, f32)>> 
{
    queries.par_iter()
        .with_min_len(10) // Batch size
        .map(|q| self.search(q, k))
        .collect()
}
```

**Why it's faster:**
- Better CPU cache utilization
- Amortized overhead
- OS can optimize disk access patterns

---

## Expected Performance After All Optimizations

### Phase-by-Phase Breakdown (1M vectors, 1024-dim, k=10)

**On Laptop (2GB available RAM):**

| Phase | Operation | Time | Notes |
|-------|-----------|------|-------|
| 1 | Quantize query | 1 μs | 1024 floats → bits |
| 2 | Tree traversal (4 levels) | 50 μs | Hamming on quantized centroids |
| 3 | Identify candidate leaves | 20 μs | probes=2 per level |
| 4 | Parallel scan candidates | 500 μs | ~2K vectors, 10 cores |
| 5 | mmap read (50 vectors) | 50 μs | Some cache misses |
| 6 | Full precision L2 (50) | 2 μs | NEON optimized |
| 7 | Final sort | 1 μs | Small set |
| **Total** | | **~624 μs** | **~1-2ms p99 with cache misses** |

### Latency Distribution (Laptop)

```
p50:  1-2 ms     ← Most nodes cached
p95:  2-5 ms     ← Some cache misses
p99:  5-10 ms    ← Cold reads from disk
p999: 20-50 ms   ← OS scheduling + cold cache
```

**On Powerful Hardware (64GB RAM, NVMe SSD):**

```
p50:  0.1-0.3 ms   ← Everything cached
p95:  0.3-0.5 ms   ← Rare misses
p99:  0.5-1 ms     ← NVMe is fast
p999: 1-2 ms       ← Still excellent
```

**Same code, better hardware = 10x faster!**

---

## Implementation Order

### Week 1: Foundation (High Impact)
1. Remove hierarchical index (use flat scan)
2. Load quantized vectors into RAM
3. Implement two-phase search
4. Verify NEON Hamming is used

**Expected:** 1-2ms p99

### Week 2: Parallel + Memory (Medium Effort)
1. Add parallel Hamming scan with rayon
2. Implement mmap for full precision vectors
3. Tune chunk sizes for parallel work

**Expected:** 0.5-1ms p99

### Week 3: Polish (Low Hanging Fruit)
1. Add prefetch hints
2. Align vectors to cache lines
3. Optimize sorting (min-heap vs full sort)

**Expected:** 0.3-0.7ms p99

### Week 4: Advanced (Optional)
1. Add LRU caching
2. Batch query optimization
3. Profile with Instruments

**Expected:** 0.2-0.5ms p99 TARGET!

---

## Benchmarking Command

```rust
// Add to benches/speed_bench.rs
fn bench_1m_flat_quantized(c: &mut Criterion) {
    let vectors = generate_vectors(1_000_000, 1024);
    let index = FlatIndex::new(vectors);
    let query = generate_query(1024);
    
    c.bench_function("1M_flat_quantized", |b| {
        b.iter(|| index.search(black_box(&query), 10))
    });
}
```

Run:
```bash
cargo bench --bench speed_bench
```

---

## Code Structure

```
src/
├── index/
│   ├── flat.rs           ← New! Flat index implementation
│   └── hierarchical.rs   ← Keep for >10M vectors
├── storage/
│   └── mmap.rs           ← New! Memory-mapped vector store
└── cache/
    └── lru.rs            ← New! LRU cache layer
```

---

## Quick Wins (Do First!)

1. **Remove Index** (5 minutes)
   ```rust
   // Just scan everything
   let results = quantized.par_iter().enumerate()...
   ```

2. **Load Quantized in RAM** (10 minutes)
   ```rust
   let quantized: Vec<BinaryVector> = quantizer
       .quantize_batch_parallel(&vectors);
   ```

3. **Use Existing NEON Hamming** (already done!)
   ```rust
   use vectordb::quantization::hamming_distance;
   ```

**These three changes alone get you to ~1ms p99!**

---

## Expected Results (Scale Comparison)

### Laptop (16GB RAM, 2GB available)
| Configuration | p50 | p99 | QPS | RAM | Scales to |
|---------------|-----|-----|-----|-----|-----------|
| **Current (hierarchical)** | 3ms | 10ms | 100 | 2GB | 10M vectors |
| **+ Quantized tree** | 2ms | 5ms | 200 | 2GB | 50M vectors |
| **+ Parallel probes** | 1ms | 3ms | 300 | 2GB | 50M vectors |
| **+ Smart caching** | 1ms | 2ms | 500 | 2GB | 50M vectors |

### Server (64GB RAM, NVMe SSD)
| Configuration | p50 | p99 | QPS | RAM | Scales to |
|---------------|-----|-----|-----|-----|-----------|
| **Same code, more RAM** | 0.3ms | 1ms | 1000 | 16GB | 500M vectors |
| **+ All optimizations** | 0.2ms | 0.5ms | 2000+ | 32GB | 1B+ vectors |

**Key insight: Same architecture scales 100x with better hardware!**

---

## Trade-offs & Design Philosophy

### Why Hierarchical Index for Everything

**Advantages:**
- Scales from 1M to billions of vectors
- Logarithmic growth (depth increases slowly)
- Works with limited RAM (only tree structure needed)
- Mature algorithms (proven at scale)
- Same code runs on laptop and production

**Current Limitations (Laptop):**
- Cold start latency (disk I/O)
- Limited cache size with 2GB RAM
- Slower than optimal for 1M vectors alone

**Future Advantages (Better Hardware):**
- More RAM = bigger cache = lower latency
- NVMe SSD = 10x faster cold reads
- More cores = more parallel probes
- Can scale to billions without rewrite

### Performance Scaling by Hardware

| Hardware | 1M Vectors | 100M Vectors | 1B Vectors |
|----------|-----------|--------------|------------|
| **Laptop (16GB)** | 2-5ms | Too slow | Won't fit |
| **Workstation (64GB)** | 0.5-1ms | 2-5ms | Tight fit |
| **Server (256GB)** | 0.2-0.5ms | 0.5-2ms | 5-10ms |
| **Distributed (N×256GB)** | 0.1-0.3ms | 0.3-1ms | 1-5ms |

**Same hierarchical architecture works at every scale!**

---

## TL;DR

**To optimize hierarchical index for current laptop AND future scale:**

1. **Use binary quantization for tree nodes** (Hamming distance is fast)
2. **Keep quantized index in RAM** (128 MB for 1M vectors)
3. **Parallel tree probes** (use all cores)
4. **Smart caching** (LRU for hot vectors, adapts to available RAM)
5. **mmap full precision** (zero-copy access, OS handles paging)

**Current laptop (2GB available):** 1-3ms p99 latency  
**Future server (64GB RAM):** 0.2-0.5ms p99 latency  
**Same code, scales from millions to billions!**

Next: See `SPEED_CHECKLIST.md` for implementation steps.
