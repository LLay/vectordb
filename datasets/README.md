# Vector Datasets

This folder contains pre-generated vector datasets for testing and benchmarking.

## Quick Start

```bash
# Generate standard datasets (10K-100K vectors)
cd datasets
rustc generate_datasets.rs && ./generate_datasets

# Generate ALL datasets including 1M vectors (~3.9 GB)
rustc generate_datasets.rs && ./generate_datasets --large
```

## Available Datasets

| File | Vectors | Dims | Clusters | Size | Type |
|------|---------|------|----------|------|------|
| `data_10k_1024d_10clusters.bin` | 10K | 1024 | 10 | ~39 MB | Gaussian clusters |
| `data_10k_1024d_50clusters.bin` | 10K | 1024 | 50 | ~39 MB | Gaussian clusters |
| `data_10k_1024d_100clusters.bin` | 10K | 1024 | 100 | ~39 MB | Gaussian clusters |
| `data_100k_1024d_100clusters.bin` | 100K | 1024 | 100 | ~391 MB | Gaussian clusters |
| `data_1m_1024d_1000clusters.bin` | 1M | 1024 | 1000 | ~3.9 GB | Gaussian clusters (--large) |
| `data_10k_1024d_random.bin` | 10K | 1024 | - | ~39 MB | Uniform random |

## Dataset Format

Binary format: `[num_vectors: u32][dims: u32][vector1_data][vector2_data]...`

Each vector is stored as consecutive `f32` values (little-endian).

## Generation Details

- **Gaussian Clusters**: Vectors are sampled from Gaussian distributions around random cluster centers
  - Cluster centers: Random points in `[-1, 1]^dims`
  - Standard deviation: 0.2 within each cluster
  - Vectors distributed evenly across clusters
  
- **Uniform Random**: Baseline dataset with uniform random distribution in `[-1, 1]^dims`

## Usage in Code

```rust
use std::fs::File;
use std::io::Read;

fn load_dataset(path: &str) -> (Vec<Vec<f32>>, usize, usize) {
    let mut file = File::open(path).unwrap();
    let mut buffer = [0u8; 8];
    file.read_exact(&mut buffer).unwrap();
    
    let num_vectors = u32::from_le_bytes(buffer[0..4].try_into().unwrap()) as usize;
    let dims = u32::from_le_bytes(buffer[4..8].try_into().unwrap()) as usize;
    
    // ... read vectors
}
```

Or use the helper in `examples/dataset_loader.rs`.

## Tools

- **`generate_datasets.rs`**: Generate all datasets (use `--large` for 1M)
- **`inspect_dataset.rs`**: Inspect dataset contents and check for issues

## Notes

- Large datasets (1M vectors) use streaming generation to minimize memory usage
- All datasets use the same random seed for reproducibility (via `rand::thread_rng()`)
- Binary files are in `.gitignore` - regenerate as needed
