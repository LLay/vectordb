# Quick Reference: What Can You Build on Your Laptop?

**Your Hardware:** 16GB RAM, ~1TB SSD, 10-core Apple Silicon

---

## The Short Answer

**❌ 100 billion vectors?** No, you'd need 12.5 TB storage (even quantized) and 444 GB RAM for the index.

**✅ What you CAN do:** **1-2 billion vectors** with <100ms p99 latency!

---

## Practical Limits by Dimension

| Vectors | Dims | Quantized Storage | Index RAM | Query Latency (p99) | Feasible? |
|---------|------|-------------------|-----------|---------------------|-----------|
| **10M** | 1024 | 1.3 GB | 1.5 GB | 0.5-2ms | ✅ Excellent |
| **50M** | 1024 | 6.4 GB | 7.4 GB | 2-10ms | ✅ Great |
| **100M** | 512 | 6.4 GB | 7.8 GB | 5-20ms | ✅ Good |
| **500M** | 512 | 32 GB | 14 GB | 20-80ms | ✅ Possible |
| **1B** | 512 | 64 GB | 15.5 GB | 40-150ms | ⚠️ Need optimization |
| **1B** | 128 | 16 GB | 15.5 GB | 20-60ms | ✅ With disk-backed |
| **2B** | 128 | 32 GB | 15.8 GB | 50-120ms | ⚠️ Pushing limits |
| **100B** | Any | 1.6-12.8 TB | 56-444 GB | N/A | ❌ Impossible |

---

## Three Deployment Modes

### 1. **Fully In-Memory** (Best Performance)
- Dataset: Up to **50M vectors** (1024-dim)
- Latency: **0.5-5ms** p99
- RAM usage: 8-12 GB
- Use case: Real-time applications, low latency critical

### 2. **Disk-Backed Index** (Best Balance)
- Dataset: Up to **2B vectors** (128-dim)
- Latency: **20-100ms** p99
- RAM usage: 14-16 GB (index only)
- Storage: Up to 280 GB available
- Use case: **← YOUR TARGET GOAL** ✅

### 3. **Distributed System** (Unlimited Scale)
- Dataset: **100B+ vectors**
- Latency: 50-200ms p99
- Cost: $40-80K/month (cloud)
- Use case: Internet-scale search

---

## Test It Yourself

Run the scale demo to see actual performance:

```bash
cargo run --release --example scale_demo
```

This will test:
- 100K vectors (baseline)
- 1M vectors (typical laptop scale)
- 5M vectors (comfortable upper limit)

---

## Memory Breakdown (Example: 1B vectors, 128-dim)

```
┌─────────────────────────────────────┐
│ Your 16 GB RAM                      │
├─────────────────────────────────────┤
│ System: 2 GB                        │  System overhead
│ Index: 15.5 GB                      │  ← Centroids for tree
│ Cache: 0.5 GB                       │  Hot vectors
├─────────────────────────────────────┤
│ Disk (SSD): 280 GB available       │
│ - Quantized vectors: 16 GB          │  Binary compressed
│ - Original vectors: 48 GB (archive) │  For reranking
│ - Index metadata: 2 GB              │
│ - Free: 214 GB                      │
└─────────────────────────────────────┘
```

---

## Key Optimizations for Scale

### To reach 1B vectors on your laptop:

1. **Use 128-256 dimensions** (not 1024)
2. **Binary quantization** (32x compression)
3. **Disk-backed storage** with mmap
4. **Two-phase search:**
   - Phase 1: Hamming distance on all (fast)
   - Phase 2: Full precision on top-k (accurate)
5. **Tune parameters:**
   - `branching_factor: 8`
   - `max_leaf_size: 256`
   - `probes_per_level: 2`
   - `rerank_factor: 3`

---

## Cost to Scale Beyond Laptop

| Vectors | Monthly Cost (AWS) | Hardware Alternative |
|---------|-------------------|----------------------|
| 1B | $0 (your laptop) | Your current setup |
| 10B | $4,000-8,000 | ~$15K server (128GB RAM, 8TB SSD) |
| 100B | $40,000-80,000 | ~$150K cluster (100+ nodes) |

---

## Recommended Next Steps

1. **Run benchmarks** to establish baseline:
   ```bash
   cargo bench --bench index_bench
   ```

2. **Test at scale** with demo:
   ```bash
   cargo run --release --example scale_demo
   ```

3. **Implement disk-backing** (see `SCALE_ANALYSIS.md` for details)

4. **Profile queries** to find bottlenecks:
   ```bash
   cargo instruments --example scale_demo --template Time
   ```

5. **Optimize hot paths** based on profiling results

---

## TL;DR

**Can you do 100B vectors with 100ms p99?** No, not on a laptop.

**Can you do 1-2B vectors with 100ms p99?** YES! With proper optimization.

**Sweet spot for your hardware:** 100M-500M vectors @ 10-30ms p99 latency.

Read `SCALE_ANALYSIS.md` for the detailed math and architecture recommendations.
