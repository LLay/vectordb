# GIST-960M Dataset

**High-dimensional benchmark dataset for vector search evaluation**

## Overview

- **Dimensions:** 960
- **Base vectors:** 1,000,000
- **Learn vectors:** 500,000
- **Query vectors:** 1,000
- **Ground truth:** 100-NN for each query
- **Format:** `.fvecs` (float vectors) and `.ivecs` (int vectors)

## Why GIST?

GIST is ideal for testing high-dimensional vector search algorithms:
- **960 dimensions** - Much higher than SIFT (128D), closer to real embeddings
- **RaBitQ paper tested on GIST-960D** - Can compare directly to published results
- **Standard benchmark** - Widely used in vector search research

## Source

Download from: http://corpus-texmex.irisa.fr/

## File Structure

```
datasets/gist/
├── README.md
├── data/
│   └── gist/
│       ├── gist_base.fvecs       (1M vectors, 960D, 3.6GB)
│       ├── gist_learn.fvecs      (500K vectors, 960D, 1.8GB)
│       ├── gist_query.fvecs      (1K queries, 960D, 3.7MB)
│       └── gist_groundtruth.ivecs (1K x 100-NN)
├── loader.rs
└── mod.rs
```

## Usage

```rust
use gist::loader::{read_fvecs, read_ivecs};

// Load base vectors
let (vectors, dim) = read_fvecs("datasets/gist/data/gist/gist_base.fvecs")?;

// Load queries
let (queries, _) = read_fvecs("datasets/gist/data/gist/gist_query.fvecs")?;

// Load ground truth
let ground_truth = read_ivecs("datasets/gist/data/gist/gist_groundtruth.ivecs")?;
```

## Benchmarks

```bash
# Run GIST comparison benchmark (Binary vs RaBitQ)
cargo bench --bench gist_comparison
```

## Expected Performance

Based on RaBitQ paper results on GIST-960D:
- **RaBitQ distance error:** <5% (vs 20% on SIFT-128D)
- **Expected recall improvement:** 10-20% over binary quantization
- **Why better than SIFT:** Higher dimensions → better concentration effects

## Notes

- GIST-960D is what the RaBitQ paper tested on
- This is the ideal dataset to validate RaBitQ performance
- Much more representative of real-world embeddings (OpenAI, CLIP, etc.)
