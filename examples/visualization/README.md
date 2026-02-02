# Visualization Tools

This directory contains tools for visualizing vector space and search behavior.

## Quick Start

```bash
# From the vectordb root directory, run:
source .venv/bin/activate
pip install -r requirements.txt
cargo run --release --example observability_demo

# This will automatically generate visualizations in this directory
```

## Files

### Scripts
- **`visualize.py`** - Python script to generate vector_space.png
- **`generate_visualizations.sh`** - Automated script to generate all images
- **`VISUALIZATION_GUIDE.md`** - Complete documentation

### Generated Outputs (gitignored)
- **`vector_space*.csv`** - 2D projections of vectors with labels
- **`vector_space*.png`** - Rendered vector space images
- **`tree_structure*.dot`** - Graphviz tree structures
- **`tree_structure*.png`** - Rendered tree images

## What the Visualizations Show

### Vector Space (vector_space.png)
- **Query**: Red star (large, prominent)
- **Found neighbors**: Green dots (true positives)
- **Missed neighbors**: Orange dots (false negatives)
- **Searched area**: Light blue dots (where search looked)
- **Other vectors**: Gray dots (background)

### Tree Structure (tree_structure.png)
- **Light green (bold)**: Searched leaves containing ground truth
- **Orange (bold)**: Missed leaves containing ground truth
- **Light blue**: Searched leaves without GT
- **White**: Other leaves
- **Light gray**: Internal nodes

## Integration

These tools are used by:
- `examples/observability_demo.rs` - Automatically generates visualizations
- `examples/visualize_search.rs` - Standalone visualization tool

The visualization module is at `src/visualization.rs`.
