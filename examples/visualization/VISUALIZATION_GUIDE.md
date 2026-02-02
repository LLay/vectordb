# Vector Space Visualization Guide

## Overview

The visualization tools help understand:
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
- terminal output with coverage analysis

## Output

### 1. Coverage Analysis (Terminal)

**Key Metrics:**
- **Coverage %**: What fraction of GT-containing leaves were searched
- **Wasted effort**: Leaves searched that don't contain GT
- **Missed leaves**: GT-containing leaves that weren't searched (bad!)

### 2. Vector Space CSV

**Columns:**
- `x, y`: 2D coordinates (random projection)
- `type`: Vector classification
  - `query`: The search query
  - `found_neighbor`: True neighbor that was found
  - `missed_neighbor`: True neighbor that was missed
  - `searched_non_neighbor`: Non-neighbor in searched area
  - `other`: Not searched, not a neighbor
- `in_ground_truth`: 1 if true neighbor, 0 otherwise
- `was_searched`: 1 if in a searched leaf, 0 otherwise
- `leaf_id`: Which leaf contains this vector

### 3. Tree Structure DOT

```bash
dot -Tpng tree_structure.dot -o tree_structure.png
open tree_structure.png
```

**Node Colors:**
- **Light green (bold)**: Searched leaf containing GT
- **Light blue**: Searched leaf without GT
- **Orange (bold)**: Missed leaf containing GT
- **White**: Other leaves
- **Light gray**: Internal nodes
