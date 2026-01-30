# Adaptive Leaf Sizing Implementation

## Summary

Implemented adaptive branching and cluster merging to achieve target leaf sizes of ~100 vectors, eliminating the problem of thousands of tiny 1-vector leaves.

## Results (100K vectors, target_leaf_size=100)

### Before Adaptive Sizing
- **18,316 leaves**, avg=5.5, median=1, min=1, max=100
- 87% of leaves had 1-10 vectors (15,905 tiny leaves!)
- Build time: 93.7s
- Recall@1000 (probes=50): 89.2%

### After Adaptive Sizing  
- **1,118 leaves**, avg=89.4, median=79, min=30, max=200
- Much more uniform distribution centered around target
- Build time: 42.3s (55% faster!)
- Recall@1000 (probes=50): 32.0% (needs investigation)

## Implementation Details

### 1. Adaptive Branching Factor

Instead of always using `branching_factor`, we calculate the optimal number of clusters based on remaining vectors:

```rust
let target_clusters = (num_vectors as f32 / target_leaf_size as f32).ceil() as usize;
let num_clusters = target_clusters
    .max(2)  // At least 2 clusters to make progress
    .min(branching_factor)  // Don't exceed max branching factor
    .min(num_vectors);  // Can't have more clusters than vectors
```

### 2. Leaf Creation Threshold

Changed from strict `max_leaf_size` to flexible `target_leaf_size`:

```rust
// Create leaf if we're within 2x of target or hit max depth
if num_vectors <= target_leaf_size * 2 || current_depth >= 10 {
    // Create leaf (can be larger than target)
}
```

This allows leaves to be 30-200 vectors instead of forcing strict 1-100 range.

### 3. Small Cluster Merging

Merge clusters smaller than 30% of target into nearest neighbors:

```rust
let min_cluster_size = (target_leaf_size * 3 / 10).max(10);
// Merge small clusters into closest neighbor by centroid distance
// Always merge - recursion will split if result is too large
```

## Benefits

1. **16x fewer leaves** (18,316 → 1,118)
2. **55% faster build** (93.7s → 42.3s)
3. **No more tiny leaves** (min went from 1 → 30)
4. **More uniform distribution** (tight around target)
5. **Shallower tree** (max depth 10 → 5)

## Trade-offs

- **Recall decreased** (89.2% → 32.0% at probes=50)
  - Fewer leaves means each probe covers less of the space
  - Need to investigate if this is due to poor clustering or just fewer leaves
  - May need to adjust probes_per_level to compensate

## Next Steps

1. Investigate recall drop - is it clustering quality or just coverage?
2. Consider adjusting search strategy for larger leaves
3. Benchmark search speed with new leaf sizes
4. Test with different target_leaf_size values (50, 150, 200)
