# Vector Space Visualization Guide

## Overview

The visualization tools help you understand:
1. **Where vectors are in space** (2D projection)
2. **What the search actually explored** vs **where the true neighbors are**
3. **Tree structure** showing search path vs ground truth location

## Quick Start

```bash
cargo run --release --example visualize_search
```

This generates:
- `vector_space.csv` - 2D projection of all vectors with labels
- `tree_structure.dot` - Tree visualization in Graphviz format

## Understanding the Output

### 1. Coverage Analysis (Terminal)

```
Ground Truth Distribution:
  100 nearest neighbors spread across 1 leaves
  Top leaves with GT vectors:
    ✓ Leaf 20 contains 100 GT vectors

Search Coverage:
  Searched 10 leaves total
  1 leaves contain GT vectors (100%)
  9 leaves searched but no GT (wasted effort)
  0 leaves with GT were MISSED
```

**Key Metrics:**
- **Coverage %**: What fraction of GT-containing leaves were searched
- **Wasted effort**: Leaves searched that don't contain GT
- **Missed leaves**: GT-containing leaves that weren't searched (bad!)

### 2. Vector Space CSV

**Columns:**
- `x, y`: 2D coordinates (random projection)
- `type`: Vector classification
  - `query`: The search query
  - `found_neighbor`: True neighbor that was found ✅
  - `missed_neighbor`: True neighbor that was missed ❌
  - `searched_non_neighbor`: Non-neighbor in searched area
  - `other`: Not searched, not a neighbor
- `in_ground_truth`: 1 if true neighbor, 0 otherwise
- `was_searched`: 1 if in a searched leaf, 0 otherwise
- `leaf_id`: Which leaf contains this vector

**Visualize with Python/matplotlib:**

```bash
python3 << 'EOF'
import pandas as df
import matplotlib.pyplot as plt

df = pd.read_csv('vector_space.csv')

colors = {
    'query': 'red',
    'found_neighbor': 'green',
    'missed_neighbor': 'orange',
    'searched_non_neighbor': 'lightblue',
    'other': 'lightgray'
}

for vtype in df['type'].unique():
    data = df[df['type'] == vtype]
    plt.scatter(data['x'], data['y'], 
                c=colors.get(vtype, 'gray'),
                label=vtype, alpha=0.6, s=20)

plt.legend()
plt.title('Vector Space Search Coverage')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('vector_space.png', dpi=150)
print('✓ Saved to vector_space.png')
EOF
```

**What to look for:**
- **Green dots near query**: Good! Found the true neighbors
- **Orange dots near query**: Bad! Missed nearby neighbors
- **Blue region**: Where the search looked
- If blue region doesn't overlap green dots → search is looking in wrong place!

### 3. Tree Structure DOT

**Visualize online:**
1. Open https://dreampuf.github.io/GraphvizOnline/
2. Paste contents of `tree_structure.dot`
3. View the tree!

**Or render locally:**
```bash
dot -Tpng tree_structure.dot -o tree_structure.png
open tree_structure.png
```

**Node Colors:**
- **Light green (bold)**: Searched leaf containing GT ✅
- **Light blue**: Searched leaf without GT
- **Orange (bold)**: Missed leaf containing GT ❌
- **White**: Other leaves
- **Light gray**: Internal nodes

**What to look for:**
- Bold green leaves: Success! Found GT
- Bold orange leaves: Problem! GT was in unvisited leaves
- Many blue leaves: Searching widely but not finding GT → clustering issue

## Interpreting Results

### Scenario 1: Perfect Coverage
```
Coverage: 100%
Missed: 0
```
✅ Search is working well! All GT-containing leaves were explored.

### Scenario 2: Low Coverage, Clustered GT
```
Coverage: 30%
GT spread across: 10 leaves
Searched: 10 leaves
Overlap: 3 leaves
```
❌ Problem: GT neighbors are scattered across many leaves, but we're only finding a few.
**Solution**: Increase `probes` or improve clustering.

### Scenario 3: High Waste
```
Coverage: 100%
Wasted effort: 90%
```
⚠️ Finding all GT but searching too many irrelevant leaves.
**Solution**: Decrease `probes` or improve clustering to concentrate GT better.

### Scenario 4: Missed Neighbors
```
Missed leaves: 5
Orange dots near query in vector space
```
❌ Critical: True neighbors exist nearby but in unvisited leaves.
**Root cause**: Tree navigation is going down wrong branches.
**Solution**: 
- Check if centroids represent clusters well
- Increase probes at early tree levels
- Consider if data has natural cluster structure

## Advanced: Custom Visualization

The visualization module is available for custom analysis:

```rust
use vectordb::visualization::{
    visualize_vector_space,
    visualize_tree_structure,
    print_coverage_report,
};

// After performing search with stats
let (results, stats) = index.search_with_stats(query, k, probes, rerank_factor);

// Generate visualizations
visualize_vector_space(&index, &vectors, query, &ground_truth, &stats, "output.csv")?;
visualize_tree_structure(&index, &stats, &ground_truth, "tree.dot")?;
print_coverage_report(&index, &ground_truth, &stats);
```

## Tips

1. **Start with small datasets** (1K-10K vectors) for faster iteration
2. **Use clustered data** for realistic testing (not uniform random)
3. **Compare different probe counts** to see coverage vs cost tradeoff
4. **Look for patterns** in missed neighbors - are they always in certain leaves?
5. **Check tree depth** - very deep trees may have navigation issues

## Files Generated

All visualization outputs are gitignored:
- `vector_space.csv`
- `tree_structure.dot`
- `vector_space.png` (if you generate it)
- `tree_structure.png` (if you generate it)
