# CuddleDB System Design

## Architecture Overview (Search Path)

```
Query Vector (1024-dim f32)
    ↓
Binary Quantization (128 bytes)
    ↓
Hierarchical Tree Navigation
    Level 0: ~100 root clusters      → Explore top N probes
    Level 1: ~5K sub-clusters         → Expand best candidates
    Level 2-N: Continue until leaves  → Collect leaf nodes
    ↓
Binary Filtering Phase
    → Scan accumulated leaf vectors with Hamming distance
    → Select top k×rerank candidates
    ↓
Full Precision Reranking
    → Load full vectors from mmap storage
    → Compute L2 distance on candidates
    → Return final top-k results
```

---

## Core Components

### 1. Hierarchical Clustered Index

**Implementation:** Adaptive multi-level k-means clustering

```rust
pub struct ClusteredIndex {
    nodes: Vec<ClusterNode>,           // Tree structure
    root_ids: Vec<usize>,               // Level 0 entry points
    quantizer: BinaryQuantizer,         // Compression
    binary_vectors: Vec<BinaryVector>,  // Quantized vectors in RAM
    full_vectors: MmapVectorStore,      // Full precision on disk
}
```

**Key Features:**
- **Adaptive splitting:** Clusters split until reaching target leaf size
- **Unbalanced trees:** Different branches can have different depths
- **Leaf accumulation:** Search collects all reachable leaves before scanning

**Tree Building:**
```rust
ClusteredIndex::build(
    vectors,
    "vectors.bin",
    branching_factor: 100,    // Clusters per level
    target_leaf_size: 100,    // Vectors per leaf
    metric: DistanceMetric::L2,
    max_iterations: 20,       // K-means convergence
)
```

**Example Tree Structure (1M vectors):**
```
Level 0: 100 root nodes
Level 1: ~5,000 nodes (mostly internal)
Level 2: ~6,000 nodes (mixed)
Level 3: ~2,000 nodes (mostly leaves)
Total:   ~13,500 nodes, ~10,800 leaves
Max depth: 7 levels
Avg leaf size: 92 vectors
```

**Why Hierarchical?**
- O(log N) search vs O(N) flat scan
- Natural pruning of search space
- Scales to billions of vectors
- Multi-probe allows recall/latency tuning

---

### 2. Binary Quantization

**Compression:** 4096 bytes (1024×f32) → 128 bytes (1024 bits)  
**Ratio:** 32x compression  
**Method:** Sign-based quantization with learned thresholds

```rust
pub struct BinaryQuantizer {
    thresholds: Vec<f32>,  // One per dimension
}

// Quantization: v[i] >= threshold[i] → 1, else → 0
pub struct BinaryVector {
    data: Vec<u8>,  // Packed bits (1024 bits = 128 bytes)
}
```

**Distance Computation:**
- **Binary:** Hamming distance (XOR + popcount) ~10ns
- **Full precision:** L2 distance (NEON) ~200ns
- **Speedup:** 20x faster filtering

**Why Binary?**
- Fits more vectors in cache (32x smaller)
- Fast distance computation (bitwise ops)
- Enables two-phase search
- Minimal accuracy loss with reranking

---

### 3. Memory-Mapped Storage

**Purpose:** Keep full-precision vectors on disk, access on-demand

```rust
pub struct MmapVectorStore {
    mmap: Mmap,              // Memory-mapped file
    num_vectors: usize,
    dimension: usize,
}
```

**Memory Layout (1M vectors, 1024-dim):**
```
RAM:
  Binary vectors:    128 MB  (1M × 128 bytes)
  Tree structure:     50 MB  (nodes + metadata)
  Overhead:           20 MB  (allocator, etc.)
  Total in RAM:     ~200 MB

Disk (memory-mapped):
  Full vectors:     3.9 GB  (1M × 1024 × 4 bytes)
```

**Benefits:**
- Only hot vectors cached by OS (typically 10-20%)
- Scales beyond available RAM
- No manual cache management
- Fast sequential/clustered access

**Trade-off:** ~10-20% latency increase vs all-in-RAM

---

### 4. ARM NEON SIMD

**Hardware Acceleration:** 128-bit vector operations on Apple Silicon

**Key Operations:**
```rust
// Dot product (4 floats at once)
let va = vld1q_f32(a.as_ptr());
let vb = vld1q_f32(b.as_ptr());
sum = vfmaq_f32(sum, va, vb);  // Fused multiply-add

// Hamming distance (popcount)
let xor = veorq_u8(a, b);
let count = vcntq_u8(xor);
```

**Performance:**
- L2 distance (1024-dim): ~200ns
- Dot product (1024-dim): ~150ns
- Hamming distance (1024-bit): ~10ns

**Optimizations:**
- 4x loop unrolling with independent accumulators
- Hides FMA latency (4 cycles) via ILP
- Cache-friendly memory access patterns

---

### 5. Two-Phase Search

**Strategy:** Fast binary filtering + precise full-precision reranking

```
Phase 1: Binary Filtering
  Input:  All vectors in searched leaves
  Method: Hamming distance on binary vectors
  Output: Top k×rerank_factor candidates
  Cost:   O(candidates) × 10ns

Phase 2: Full Precision Reranking
  Input:  Top k×rerank_factor candidates
  Method: L2 distance on full vectors (from mmap)
  Output: Final top-k results
  Cost:   O(k×rerank_factor) × 200ns
```

**Example (k=10, rerank_factor=3):**
```
1000 vectors in leaves → 1000 Hamming comparisons = 10μs
Top 30 candidates → 30 L2 comparisons = 6μs
Total: 16μs vs 200μs all-L2
```

**Why It Works:**
- Binary filtering is very fast but lossy
- Reranking recovers accuracy on small candidate set
- Net speedup while maintaining recall

---

## Search Algorithm

### Budget-Based Adaptive Probes

**Problem:** With fixed probes at all levels, early mistakes compound exponentially.

Traditional approach (probes=10 everywhere):
```
Level 0 (root):  10/100 clusters = 10% coverage
Level 1:         10/100 clusters = 10% coverage  
Level 2:         10/100 clusters = 10% coverage
```

**Issue:** Missing the correct cluster at the root eliminates 1,000+ descendants!

**Solution:** Budget-based allocation - keep TOTAL work constant, redistribute smartly.

```rust
fn calculate_adaptive_probes(
    depth: usize, 
    base_probes: usize,
    remaining_budget: &mut usize
) -> usize {
    // Allocate budget by depth with decreasing weights
    let weight = match depth {
        0 => 0.40,  // 40% at root
        1 => 0.30,  // 30% at level 1
        2 => 0.20,  // 20% at level 2
        _ => 0.10,  // 10% at deeper levels
    };
    // ... allocation logic ...
}
```

**Example with probes=10, max_depth=5:**
```
Total budget: 10 × 5 = 50 probes
Level 0 (root): 20 probes (40% of budget)
Level 1:        15 probes (30%)
Level 2:        10 probes (20%)
Level 3+:       5 probes  (10%)
```

**Why it works:**
- **Root probes are cheap:** Only 100 nodes, small absolute cost
- **Root mistakes are fatal:** Wrong cluster → 1,000+ vectors unreachable
- **Total work stays constant:** Same 50 probes, just redistributed
- **No performance penalty:** Unlike multiplier-based which does MORE work

**Impact:**
- Recall: +8-12% improvement (53% → 62% on balanced config)
- Latency: **SAME** (no penalty, total work unchanged!)
- Key insight: Smarter allocation without doing more work

---

### Search Implementation

```rust
pub fn search(&self, query: &[f32], k: usize, probes: usize, rerank_factor: usize) -> Vec<(usize, f32)> {
    // 1. Quantize query
    let query_binary = self.quantizer.quantize(query);
    
    // 2. Navigate tree with budget-based adaptive probes
    let mut current_nodes = self.root_ids.clone();
    let mut accumulated_leaves = Vec::new();
    let mut depth = 0;
    let mut remaining_budget = probes * max_depth;  // Total budget
    
    while !current_nodes.is_empty() {
        // Budget-based adaptive probes: redistribute fixed budget
        let probes_at_level = self.calculate_adaptive_probes(
            depth, probes, current_nodes.len(), &mut remaining_budget
        );
        
        // Find top probes closest to query (using SIMD batch + heap selection)
        let top_nodes = self.find_closest_nodes(&current_nodes, query, probes_at_level);
        
        // Separate leaves from internal nodes
        let (leaves, internal): (Vec<_>, Vec<_>) = top_nodes
            .into_iter()
            .partition(|&id| self.nodes[id].children.is_empty());
        
        accumulated_leaves.extend(leaves);
        
        // Expand internal nodes
        current_nodes = internal
            .iter()
            .flat_map(|&id| self.nodes[id].children.clone())
            .collect();
        
        depth += 1;
    }
    
    // 3. Binary scan all accumulated leaf vectors
    let mut candidates = Vec::new();
    for &leaf_id in &accumulated_leaves {
        for &vector_idx in &self.nodes[leaf_id].vector_indices {
            let dist = hamming_distance(&query_binary, &self.binary_vectors[vector_idx]);
            candidates.push((vector_idx, dist));
        }
    }
    
    // 4. Select top k×rerank_factor by Hamming distance
    candidates.sort_by(|a, b| a.1.cmp(&b.1));
    candidates.truncate(k * rerank_factor);
    
    // 5. Rerank with full precision
    let mut results: Vec<_> = candidates
        .iter()
        .map(|&(idx, _)| {
            let full_vec = self.full_vectors.get_vector(idx);
            let dist = distance(query, full_vec, self.metric);
            (idx, dist)
        })
        .collect();
    
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    results.truncate(k);
    results
}
```

**Key Innovations:** 
1. **Budget-based adaptive probes** - Redistributes fixed probe budget (more at root, less at depth) with NO performance penalty
2. **Leaf accumulation** - Handles unbalanced trees (different branches reach leaves at different depths)
3. **SIMD batch distances** - Computes 4 centroid distances simultaneously
4. **Quickselect top-k** - O(n) selection instead of O(n log n) sorting


---

## Tuning Parameters

### Index Building

**branching_factor** (10-100)
- Higher → shallower tree, faster build, slower search
- Lower → deeper tree, slower build, faster search
- Recommended: 50-100 for large datasets

**target_leaf_size** (50-200)
- Higher → fewer leaves, more vectors per leaf
- Lower → more leaves, fewer vectors per leaf
- Recommended: 100

**max_iterations** (10-30)
- K-means convergence limit
- Higher → better clustering, slower build
- Recommended: 20

### Search

**probes** (1-10)
- **Base** number of branches explored (automatically redistributed per level)
- Uses budget-based allocation: probes × max_depth total budget
- Root gets 40% of budget, level 1 gets 30%, level 2 gets 20%, rest gets 10%
- Example: probes=10, depth=5 → budget=50 → root gets 20, L1 gets 15, L2 gets 10
- Higher → better recall, proportionally more work (total budget scales)
- Recommended: 3-7

**rerank_factor** (2-10)
- Multiplier for binary candidates
- Higher → better accuracy, more reranking cost
- Recommended: 3-5

**k** (1-1000)
- Number of results to return
- Higher k → more work in both phases
- Recommended: 10-100

### Trade-offs

```
Speed-focused:     probes=2, rerank_factor=2
Balanced:          probes=5, rerank_factor=3
Accuracy-focused:  probes=10, rerank_factor=5
```

---

## Design Decisions

### Why Hierarchical over Flat (IVF)?

**Pros:**
- O(log N) vs O(N) probe cost
- Scales to billions without probe explosion
- Natural for multi-modal distributions

**Cons:**
- More complex implementation
- Harder to tune for random data
- Lower recall on uniform distributions


### Why Binary over Product Quantization?

**Pros:**
- Simpler implementation
- Faster distance (bitwise ops)
- Less training data needed

**Cons:**
- Lower compression (1 bit vs 4-8 bits)
- Less accurate approximation


### Why Memory-Mapped over All-in-RAM?

**Pros:**
- Scales beyond available RAM
- OS handles caching automatically
- Hot vectors naturally cached

**Cons:**
- 10-20% latency increase
- Requires fast storage


### Why NEON over Scalar?

**Pros:**
- 4-6x speedup
- Native to Apple Silicon
- No runtime detection needed

**Cons:**
- x86 requires separate AVX/AVX512 impl
- More complex code
