# Complete Recall Analysis

## Executive Summary

The hierarchical k-means index achieves only **10-20% recall** on random/uniform data, even with aggressive parameters. The root cause is a combination of:

1. **K-means cluster imbalance** creating thousands of tiny leaves
2. **Exponential probe cost** (`probes^depth`) making thorough search impossible
3. **Neighbor scattering** across clusters in random data

**Recommendation:** Implement a **flat IVF (Inverted File) index** for random data, which provides better recall with linear probe cost.

---

## Problem 1: K-means Creates Unbalanced Clusters

### Experiment: Cluster Balance Analysis

We clustered 10,000 random vectors with different branching factors:

| Branching | Min Size | Max Size | Imbalance | Tiny Clusters (≤20) |
|-----------|----------|----------|-----------|---------------------|
| 10        | 937      | 1,035    | 1.1x      | 0 / 10 (0%)         |
| 30        | 293      | 394      | 1.3x      | 0 / 30 (0%)         |
| 50        | 144      | 251      | 1.7x      | 0 / 50 (0%)         |
| **100**   | **2**    | **159**  | **79.5x** | **4 / 100 (4%)**    |

**Key Finding:** With `branching=100`, some clusters get only 2 vectors while others get 159! This 79.5x imbalance cascades through recursive splitting.

### Why This Happens

K-means++ initialization can place centroids close together, especially with high k. These centroids attract very few vectors. On random/uniform data, there's no natural clustering structure to guide better placement.

### Impact on Tree Structure

With `branching=100, max_leaf=20`:
- 4% of clusters at level 1 become leaves immediately (≤20 vectors)
- The remaining 96% recurse and split into 100 more clusters
- At level 2, another 4% of those become tiny leaves
- This continues, creating **thousands of tiny leaves**

Example: 100K vectors, branching=100, max_leaf=20
- Expected: ~5,000 leaves (100K / 20)
- Actual: **80,546 leaves** (avg 1.2 vectors per leaf!)

---

## Problem 2: Exponential Probe Cost

### The Math

With a hierarchical tree:
- **Leaves searched = probes^depth**
- Depth 2: `probes=10` → 100 leaves
- Depth 3: `probes=10` → 1,000 leaves
- Depth 4: `probes=10` → 10,000 leaves

### Coverage Analysis

| Dataset | Branching | Depth | Leaves | probes=10 Searches | Coverage |
|---------|-----------|-------|--------|-------------------|----------|
| 100K    | 10        | 4     | 5,590  | 10,000            | 100%     |
| 100K    | 20        | 3     | 7,544  | 1,000             | 13.3%    |
| 100K    | 30        | 2     | 900    | 100               | 11.1%    |
| 100K    | 100       | 3     | 80,546 | 1,000             | 1.2%     |

**Key Finding:** Even "shallow" trees (depth 3-4) require exponentially more probes to achieve good coverage.

---

## Problem 3: Neighbor Scattering

K-means on random data scatters true nearest neighbors across different clusters. See `KMEANS_RECALL_PROBLEM.md` for detailed analysis.

**Result:** Even if we search many leaves, we miss neighbors because they're in different branches of the tree.

---

## Solution Comparison

### Current: Hierarchical K-means
- **Depth:** 2-4 levels
- **Probe cost:** O(probes^depth) - exponential
- **Recall@10:** 10-20% with reasonable probe counts
- **Best case:** branching=30, max_leaf=200, depth=2
  - 900 leaves, 24% recall with probes=50

### Recommended: Flat IVF
- **Depth:** 1 level (flat)
- **Probe cost:** O(probes) - linear
- **Expected recall@10:** 60-80% with probes=50-100
- **Implementation:** Single k-means clustering, search top-k clusters

### Why IVF is Better for Random Data

1. **Linear probe cost:** `probes=50` searches exactly 50 clusters
2. **No exponential explosion:** Can afford to search more clusters
3. **Better coverage:** 50 out of 1000 clusters = 5% of data
4. **Simpler:** No recursive tree navigation

### When to Use Each

| Data Distribution | Recommended Index | Why |
|-------------------|-------------------|-----|
| Random/Uniform    | Flat IVF          | No natural hierarchy, need broad search |
| Clustered         | Hierarchical      | Natural hierarchy reduces search space |
| Mixed             | Hybrid (IVF + local hierarchy) | Top-level IVF, local refinement |

---

## Implementation Plan

### Phase 1: Flat IVF Index (Recommended)
1. Single k-means clustering (k=1000-5000)
2. Search top-k clusters by centroid distance
3. Scan all vectors in selected clusters
4. Binary quantization for fast scanning
5. Rerank with full precision

**Expected performance:**
- Build: ~5-10s for 100K vectors
- Query: <10ms with probes=50-100
- Recall@10: 60-80%

### Phase 2: Optimize Current Hierarchical Index
1. Use branching=20-30, max_leaf=100-200
2. Limit depth to 2-3 levels
3. Implement cluster rebalancing (merge tiny clusters)
4. Add adaptive probe selection

**Expected performance:**
- Build: ~1-2s for 100K vectors
- Query: <10ms with probes=20-50
- Recall@10: 30-40%

### Phase 3: Hybrid Approach
1. Top-level: IVF with 100-500 clusters
2. Second-level: Local hierarchical refinement
3. Best of both worlds

---

## Conclusion

The hierarchical k-means index is **fundamentally limited** on random data due to:
1. Cluster imbalance creating tiny leaves
2. Exponential probe cost
3. Neighbor scattering

**Next step:** Implement a flat IVF index for better recall with manageable query cost.
