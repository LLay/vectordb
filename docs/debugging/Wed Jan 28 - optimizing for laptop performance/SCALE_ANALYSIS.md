# Scale Analysis: 100 Billion Vectors on a Laptop

## Your Hardware
- **RAM:** 16 GB
- **Storage:** 926 GB total (283 GB available)
- **CPU:** 10 cores (Apple Silicon M-series)

---

## The Reality Check: 100 Billion Vectors

### Storage Requirements

**Scenario: 1024-dimensional f32 vectors**

#### Without Quantization (Raw f32)
- Per vector: 1024 dims × 4 bytes = **4 KB**
- 100B vectors: 100,000,000,000 × 4 KB = **400 TB**

**Impossible** - You'd need 400 enterprise SSDs

#### With 32x Binary Quantization
- Per vector: 1024 dims ÷ 8 bits = **128 bytes**
- 100B vectors: 100,000,000,000 × 128 bytes = **12.8 TB**

**Still impossible** - Would need 13-14 enterprise SSDs

#### What About Smaller Dimensions?

| Dimensions | f32 Size | Quantized Size | 100B Vectors (Quantized) |
|------------|----------|----------------|--------------------------|
| 128        | 512 B    | 16 B           | **1.6 TB** (possible!) |
| 256        | 1 KB     | 32 B           | **3.2 TB** |
| 512        | 2 KB     | 64 B           | **6.4 TB** |
| 768        | 3 KB     | 96 B           | **9.6 TB** |
| 1024       | 4 KB     | 128 B          | **12.8 TB** |

**Conclusion:** Only 128-dimensional vectors with quantization can fit 100B vectors on multi-TB storage.

---

## Memory Requirements (RAM)

### Hierarchical Index Structure

With your adaptive clustering index (branching=10, max_leaf=150):

**Tree depth for 100B vectors:**
- Depth = log₁₀(100B / 150) ≈ **9 levels**

**Number of internal nodes:**
- Level 0: 1 (root)
- Level 1: 10
- Level 2: 100
- Level 3: 1,000
- Level 4: 10,000
- Level 5: 100,000
- Level 6: 1,000,000
- Level 7: 10,000,000
- Level 8: 100,000,000
- **Total: ~111 million nodes**

**Memory for centroids (1024-dim):**
- Per centroid: 1024 × 4 bytes = 4 KB
- Total: 111M × 4 KB = **444 GB**

**28x larger than your RAM!**

**Memory for centroids (128-dim):**
- Per centroid: 128 × 4 bytes = 512 bytes
- Total: 111M × 512 B = **56.8 GB**

**Still 3.5x larger than your RAM!**

---

## What CAN You Do on Your Laptop?

### Option 1: Smaller Dataset (Fully In-Memory)

**With 16GB RAM, keeping 50% for index (8GB):**

| Dimensions | Quantized | Max Vectors | Storage | RAM (Index) |
|------------|-----------|-------------|---------|-------------|
| 128        | Yes       | **80M**     | 1.3 GB  | 7.9 GB      |
| 256        | Yes       | **50M**     | 1.6 GB  | 7.8 GB      |
| 512        | Yes       | **30M**     | 1.9 GB  | 7.7 GB      |
| 1024       | Yes       | **20M**     | 2.5 GB  | 7.5 GB      |

**Performance expectations:**
- **Query latency (p99):** 0.5-2ms
- **Queries per second:** 1,000-5,000 QPS
- **Build time:** 10-60 seconds

**This is practical and fast!**

---

### Option 2: Disk-Backed Index (Hybrid Approach)

**Keep index structure in RAM, vectors on SSD:**

With 280GB available storage:

| Dimensions | Quantized | Max Vectors | Storage Needed | RAM (Index) | Query Latency |
|------------|-----------|-------------|----------------|-------------|---------------|
| 128        | Yes       | **2.2B**    | 280 GB         | 15.5 GB     | 5-20ms        |
| 256        | Yes       | **1.1B**    | 280 GB         | 15.5 GB     | 10-30ms       |
| 512        | Yes       | **550M**    | 280 GB         | 15.5 GB     | 20-50ms       |
| 1024       | Yes       | **280M**    | 280 GB         | 15.5 GB     | 40-100ms      |

**Strategy:**
1. Index structure (centroids) in RAM
2. Quantized vectors on SSD with mmap
3. Original vectors in compressed archive (for reranking)

**Performance expectations:**
- **Query latency (p99):** 5-100ms (depending on k and rerank)
- **SSD reads per query:** ~150-500 (with probes=2, leaf_size=150)
- **Throughput:** 100-500 QPS

**This approaches your 100ms p99 goal for ~1 billion vectors!**

---

### Option 3: Cloud/Distributed (For True 100B Scale)

**For 100 billion vectors, you need:**

| Dimensions | Nodes | RAM/Node | Storage/Node | Total RAM | Total Storage |
|------------|-------|----------|--------------|-----------|---------------|
| 128        | 100   | 64 GB    | 2 TB         | 6.4 TB    | 200 TB        |
| 1024       | 200   | 64 GB    | 10 TB        | 12.8 TB   | 2,000 TB      |

**Architecture:**
- Sharded by vector ID ranges
- Each node handles ~1B vectors
- Distributed query fanout
- Aggregate top-k results

**Cost estimate (AWS/GCP):**
- 100-200 × r6i.2xlarge (64GB RAM) instances
- ~$40,000-80,000/month

---

## The Math: Can You Hit 100ms p99 Latency?

### Breakdown of Query Time

**For disk-backed approach (1B vectors, 1024-dim):**

1. **Traverse index tree** (~9 levels):
   - RAM access per level: ~50ns
   - Total: 9 × 50ns = **0.45 μs**

2. **Identify candidate leaves** (probes=2 per level):
   - Centroid comparisons: ~18 × 30ns = **0.54 μs**

3. **Read quantized vectors from SSD**:
   - Probes=2, leaf_size=150 → read ~300 vectors
   - Quantized size: 300 × 128B = 38.4 KB
   - SSD read latency (random): ~100-500 μs
   - **Total: ~500 μs** (0.5ms)

4. **Compute Hamming distances** (300 candidates):
   - Per distance: ~10ns
   - Total: 300 × 10ns = **3 μs**

5. **Rerank with full precision** (rerank_factor=3, k=10 → 30 candidates):
   - Read 30 vectors from SSD: 30 × 4KB = 120KB
   - SSD read: ~200 μs
   - Compute L2: 30 × 30ns = 0.9 μs
   - **Total: ~200 μs** (0.2ms)

**Total Query Time: ~1ms (median) → ~5-10ms (p99 with cache misses)**

### To Reach 100ms p99 with Current Approach:

You can handle:
- **~1 billion vectors** comfortably
- **Up to 5 billion vectors** with tuning (higher probes, better caching)

**Yes, 100ms p99 is achievable for 1-5B vectors on your laptop!**

---

## Optimizations to Push the Limits

### 1. **Better Quantization**
- Product Quantization (PQ): 8x-16x compression (vs 32x binary)
- Trade: Better accuracy, slightly slower than Hamming

### 2. **SSD Optimizations**
- Memory-mapped files (mmap) for zero-copy reads
- Pre-fetch candidate leaves during tree traversal
- Batch reads (reduce SSD round-trips)

### 3. **Caching**
- LRU cache for hot centroids (already in RAM)
- Page cache for frequently accessed vectors
- With 10% working set: ~1.6GB cache covers 100M vectors

### 4. **Index Tuning**
- Increase `max_leaf_size` to 300-500 (fewer leaves to scan)
- Decrease `branching_factor` to 5-7 (deeper tree, less fan-out)
- Use locality-sensitive clustering

### 5. **Hardware Upgrades (Still a Laptop)**
- 64GB RAM → handle 5B vectors fully in-memory
- 4TB SSD → store 30B vectors (128-dim, quantized)
- NVMe Gen4 SSD → 2-3x faster random reads

---

## Recommended Configuration for Your Laptop

### **Target: 1 Billion Vectors, 100ms p99 Latency**

**Specifications:**
- Vectors: 1,000,000,000
- Dimensions: 512
- Quantization: Binary (32x)
- Storage: 32 GB (quantized) + 64 GB (index) = **96 GB total**

**Index Parameters:**
```rust
branching_factor: 8
max_leaf_size: 256
k: 10
probes_per_level: 2
rerank_factor: 3
```

**Expected Performance:**
- Build time: ~30-60 minutes
- RAM usage: 12-14 GB (index + working set)
- Query latency (median): 2-5ms
- Query latency (p99): 20-50ms
- Throughput: 200-500 QPS

**This is production-ready for a laptop-scale vector DB!**

---

## Conclusion

| Scale | Feasible? | Configuration | Expected Latency |
|-------|-----------|---------------|------------------|
| **100M vectors** | Excellent | Fully in-memory | 0.5-2ms (p99) |
| **1B vectors** | Yes | Disk-backed, 512-dim | 20-50ms (p99) |
| **5B vectors** | Possible | 128-dim, optimized | 50-150ms (p99) |
| **100B vectors** | No | Need distributed system | N/A |

**Your 100ms p99 goal is achievable for up to ~2 billion vectors on your laptop with the right optimizations!**

For true 100B scale, you'd need a distributed cluster costing tens of thousands of dollars per month.

---

## Next Steps

1. **Benchmark at scale:** Run `cargo bench --bench index_bench` with larger datasets
2. **Implement mmap storage:** Add memory-mapped file backend for vectors
3. **Add caching layer:** LRU cache for hot vectors and leaves
4. **Tune parameters:** Experiment with branching/leaf size trade-offs
5. **Profile queries:** Find bottlenecks with flamegraphs

Want me to help implement any of these optimizations?
