# Why K-Means Hierarchical Index Has Low Recall on Random Data

## The Problem Visualized

### Scenario: Random/Uniform Distribution

Imagine 2D space with random vectors (shown as dots) and k-means clusters (shown as circles):

```
Random Distribution:
┌────────────────────────────────────────┐
│   •      •         •    •         •    │
│        •    •   •         •   •        │
│   •         •        •         •       │
│      •   •      •       •    •    •    │
│   •        •         •      •          │
│        •      •    •    •       •      │
│   •    •         •         •      •    │
│      •       •         •     •         │
└────────────────────────────────────────┘

After K-Means Clustering (k=4):
┌────────────────────────────────────────┐
│ ╔════════════╗   ╔════════════╗        │
│ ║ • • •  •   ║   ║  • • •  •  ║        │
│ ║   •   •    ║   ║     •   •  ║        │
│ ║ • Cluster1 ║   ║  Cluster2  ║        │
│ ╚════════════╝   ╚════════════╝        │
│                                         │
│ ╔════════════╗   ╔════════════╗        │
│ ║  • •  •    ║   ║   •  • •   ║        │
│ ║    •  •  • ║   ║  •   •     ║        │
│ ║ Cluster3   ║   ║  Cluster4  ║        │
│ ╚════════════╝   ╚════════════╝        │
└────────────────────────────────────────┘
```

**The Problem**: K-means creates **arbitrary boundaries**. Vectors near cluster edges have nearest neighbors in **adjacent clusters**!

---

## Example: Why Recall is Low

Let's say we query for vector **Q** (marked with ★):

```
Query Vector Q near cluster boundary:
┌────────────────────────────────────────┐
│ ╔════════════╗ │ ╔════════════╗        │
│ ║ •  •       ║ │ ║ •          ║        │
│ ║   • •      ║ │ ║  •  •      ║        │
│ ║ •     •    ║ │ ║     •      ║        │
│ ║         ★──┼─┼─┼→ • (nearest!)       │
│ ║   • •      ║ │ ║  •         ║        │
│ ║ Cluster 1  ║ │ ║ Cluster 2  ║        │
│ ╚════════════╝ │ ╚════════════╝        │
└────────────────────────────────────────┘
         Q is in Cluster 1
         But Q's nearest neighbor is in Cluster 2!
```

**With `probes=1`**: We only search Cluster 1
- **Found**: Vectors in Cluster 1 (may not be truly nearest)
- **Missed**: Q's actual nearest neighbor in Cluster 2!
- **Recall**: LOW

**With `probes=2`**: We search both Cluster 1 and Cluster 2
- **Found**: True nearest neighbor in Cluster 2
- **Recall**: BETTER ✓

But there might be other close neighbors in Clusters 3 and 4 too!

---

## Why K-Means Fails for Random Data

### 1. **No Natural Clusters**

```
Random data has NO structure:

Uniform Random:          Natural Clusters:
┌─────────────┐         ┌─────────────┐
│ • • • • • • │         │    ●●●      │
│ • • • • • • │         │   ●●●●●     │
│ • • • • • • │         │             │
│ • • • • • • │         │      ○○○    │
│ • • • • • • │         │     ○○○○    │
└─────────────┘         └─────────────┘
 All equidistant          Clear groups
 No grouping              K-means works!
```

### 2. **Arbitrary Boundaries**

K-means **forces** data into clusters by drawing lines:

```
K-means on random data (k=4):

     Forced Boundaries
          │
    ──────┼──────
          │
    ──────┼──────
          │

Every boundary is ARBITRARY!
Vectors near boundaries have
neighbors on BOTH sides.
```

### 3. **Voronoi Diagram Problem**

K-means creates a **Voronoi diagram** - each point belongs to the nearest centroid:

```
Centroids (C1, C2, C3, C4):

     C1          C2
       \        /
        \      /
    ────┼─────┼────  ← Boundary
        /      \
       /        \
     C3          C4

Any vector near the boundary
has neighbors in MULTIPLE cells!
```

---

## The Math Behind It

For random/uniform data:
- **Expected distance to nearest neighbor**: ~ O(n^(-1/d))
- **Expected distance to cluster centroid**: ~ O(1)
- **Cluster radius**: ~ O(1)

**Problem**: Cluster radius >> distance to nearest neighbor

**Result**: Nearest neighbors are scattered across many clusters!

---

## Hierarchical Makes It Worse

With a **hierarchical** structure (tree of clusters), the problem compounds:

```
Level 0:  Split into 10 clusters
Level 1:  Split each into 10 clusters (100 total)
Level 2:  Split each into 10 clusters (1000 total)

At each level, we pick top-K closest clusters.
If we pick wrong at Level 0, we miss ALL descendants!

Tree Navigation (probes=2 at each level):
          Root
         /    \
      C1       C2     ← Pick 2 closest
     /  \     /  \
   C1a C1b C2a C2b   ← Pick 2 closest from C1 and C2
    |   |   |   |
   ...........      ← Only search these leaves!

If Q's true neighbors are in C3, C4, ...
We'll NEVER find them without probing more at Level 0!
```

---

## Contrast: Clustered Data

When data HAS natural clusters:

```
Clustered Data (e.g., image embeddings):

┌────────────────────────────────────────┐
│   ●●●●●                   ○○○○○         │
│   ●●●●●●                 ○○○○○○         │
│    ●●●●                   ○○○○          │
│                                         │
│                                         │
│         ▲▲▲▲▲                           │
│        ▲▲▲▲▲▲▲           ■■■■           │
│         ▲▲▲▲             ■■■■■          │
└────────────────────────────────────────┘

K-means finds REAL clusters!
Nearest neighbors are in SAME cluster!
Recall is HIGH ✓
```

**Why it works**:
- Vectors naturally group together
- K-means boundaries align with **actual** data boundaries
- Nearest neighbors are **within** the same cluster
- `probes=1` finds most neighbors!

---

## Summary

| Data Distribution | K-Means Works? | Why? |
|-------------------|----------------|------|
| **Random/Uniform** | NO | No natural clusters, arbitrary boundaries, neighbors scattered |
| **Clustered** (images, text) | YES | Natural groups, boundaries align with data, neighbors together |
| **Multi-modal** (mixed topics) | YES | Clear separation, k-means finds modes |
| **Gaussian mixture** | YES | K-means designed for this! |

---

## What This Means for Your System

**Current benchmarks use random data** → worst-case scenario!

**Real-world data** (text embeddings, image features):
- Usually HAS structure
- Natural clusters exist
- K-means will work MUCH better
- Recall will be 80-90%+ with moderate probes

**Test**: Let's verify with clustered data...
