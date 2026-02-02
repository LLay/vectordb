# Binary Quantization Filtering Issue

## Discovery

User testing revealed that **binary quantization is too lossy** and filters out true nearest neighbors, causing artificially low recall.

### Test Setup
- 100K vectors, single leaf (100% coverage)
- Should achieve 100% recall@10

### Results

| rerank_factor | Vectors Reranked | Recall@10 | Issue |
|---------------|------------------|-----------|-------|
| 3 (default)   | 30               | 30%       | Filtered out 70% of true neighbors |
| 10            | 100              | 10%       | Even worse! |
| 100           | 1,000            | 60%       | Still missing 40% |
| 10000         | 100,000          | 100%      | Perfect (but no filtering) |

## Root Cause

**Hamming distance (binary) ≠ L2 distance (full precision)**

The two-phase search:
1. **Phase 1:** Rank by Hamming distance (fast, lossy)
2. **Phase 2:** Rerank top `k * rerank_factor` by L2 distance (accurate)

When `rerank_factor` is too small, the top candidates by Hamming distance don't include the true top-k by L2 distance.

### Why This Happened

Binary quantization compresses vectors to 1 bit per dimension:
- **Original:** f32 values in [-1, 1]
- **Quantized:** 0 or 1 (above/below threshold)
- **Information loss:** ~32x compression, massive precision loss

The Hamming distance between quantized vectors is a **very rough approximation** of L2 distance between original vectors.

### Visualization

```
True top-10 by L2:     [A, B, C, D, E, F, G, H, I, J]
Top-30 by Hamming:     [A, X, Y, B, Z, W, V, C, ..., D]
                        ↑         ↑         ↑         ↑
                     Only 4 of true top-10 appear in top-30 by Hamming!
                     
After reranking those 30:  [A, B, C, D, ...]
Recall: 4/10 = 40%
```

## Impact on Previous Results

**All our recall measurements were artificially low!**

We've been using `rerank_factor=3` throughout, which means:
- With k=10, we only rerank 30 candidates
- Binary quantization filters out ~60-70% of true neighbors
- This explains the persistent 10-20% recall we've been seeing

### What This Means

The low recall we observed was caused by **two separate problems**:

1. **Low coverage** (searching too few leaves) ← Clustering/tree structure issue
2. **Lossy binary filtering** (rerank_factor too small) ← This new discovery!

Even if we achieved 100% leaf coverage, we'd still get poor recall with `rerank_factor=3`!

## Solutions

### Option 1: Increase rerank_factor (Recommended for benchmarking)

Use `rerank_factor=50-100` for accurate recall measurements:

```rust
// Before (artificially low recall)
let results = index.search(query, 10, probes, 3);  // Only rerank 30 vectors

// After (accurate recall)
let results = index.search(query, 10, probes, 50); // Rerank 500 vectors
```

**Trade-off:** More accurate, but slower (5-10x more full-precision distance calculations)

### Option 2: Improve Binary Quantization

Use Product Quantization (PQ) instead of binary:
- 4-8 bits per dimension instead of 1
- Much better approximation of L2 distance
- Still fast (SIMD lookup tables)

### Option 3: Adaptive Reranking

Automatically adjust `rerank_factor` based on leaf size:
```rust
let rerank_factor = (vectors_in_leaves / k).clamp(10, 100);
```

### Option 4: Skip Binary Filtering for Small Candidate Sets

If candidates < 1000, skip binary filtering entirely:
```rust
if candidates.len() < 1000 {
    // Just rerank all with full precision
    rerank_all(candidates)
} else {
    // Use binary filtering
    filter_and_rerank(candidates, k, rerank_factor)
}
```

## Recommendations

### For Benchmarking (Now)

Use **`rerank_factor=50-100`** to get accurate recall measurements:
- This ensures binary quantization doesn't artificially lower recall
- We can properly measure the clustering/coverage issues
- Trade-off: Slightly slower, but necessary for accuracy

### For Production (Later)

Once we fix the clustering/coverage issues:
1. Implement Product Quantization (better than binary)
2. Use adaptive reranking based on candidate set size
3. Tune for recall/speed trade-off with real data

## Updated Benchmarks Needed

We should re-run recall tests with higher `rerank_factor` to see:
1. How much of our low recall was due to binary quantization vs coverage
2. Whether the clustering structure is actually better than we thought
3. What the true ceiling is for hierarchical k-means on random data

## Conclusion

**Good catch!** This discovery shows that:
- Binary quantization is more lossy than expected
- Our previous recall measurements underestimated the index's actual capability
- We need to increase `rerank_factor` for accurate benchmarking
- Even with perfect coverage, binary quantization limits recall

This doesn't change the fundamental clustering/coverage issues, but it shows we have **two independent problems** to solve, not just one.
