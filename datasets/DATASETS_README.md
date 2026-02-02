# Vector Datasets

Pre-generated vector datasets for testing and benchmarking are stored in the `datasets/` directory.

## Generating Datasets

To regenerate all datasets:

```bash
cargo run --release --example generate_datasets
```

## Available Datasets

### Clustered Datasets (Gaussian Distribution)

These datasets contain vectors grouped into natural clusters using Gaussian distributions. Each cluster has a random center, and vectors are sampled from a Gaussian distribution around that center (σ=0.2).

- **`datasets/data_10k_1024d_10clusters.bin`** - 10,000 vectors, 10 natural clusters (~39 MB)
- **`datasets/data_10k_1024d_50clusters.bin`** - 10,000 vectors, 50 natural clusters (~39 MB)
- **`datasets/data_10k_1024d_100clusters.bin`** - 10,000 vectors, 100 natural clusters (~39 MB)
- **`datasets/data_100k_1024d_100clusters.bin`** - 100,000 vectors, 100 natural clusters (~391 MB)

### Random Dataset (Baseline)

- **`datasets/data_10k_1024d_random.bin`** - 10,000 vectors, uniform random distribution (~39 MB)

## File Format

Binary format with little-endian encoding:

```
[num_vectors: u32][dims: u32][vector1_f32...][vector2_f32...]...
```

## Why Clustered Data?

Hierarchical k-means indices perform **much better** on clustered data than on uniform random data:

| Dataset Type | Probes=10 | Probes=50 |
|--------------|-----------|-----------|
| Random       | 3.5%      | 11%       |
| 100 Clusters | 63%       | **100%**  |

This is because:
1. **Natural Structure**: Real-world embeddings (text, images) tend to have natural clusters
2. **K-means Alignment**: The index's k-means clustering aligns with the data's natural structure
3. **Neighbor Locality**: Similar vectors are grouped together, making tree traversal effective

For production use, always test with realistic data that matches your use case, not uniform random vectors.

## Loading Datasets in Code

```rust
use std::fs::File;
use std::io::{BufReader, Read};

fn load_vectors(filename: &str) -> (Vec<Vec<f32>>, usize) {
    let file = File::open(filename).expect("Failed to open file");
    let mut reader = BufReader::new(file);
    
    // Read header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).unwrap();
    
    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    // Read vectors
    let mut vectors = Vec::with_capacity(num_vectors);
    let mut buffer = vec![0u8; dims * 4];
    
    for _ in 0..num_vectors {
        reader.read_exact(&mut buffer).unwrap();
        
        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        vectors.push(vector);
    }
    
    (vectors, dims)
}
```

## Examples Using Datasets

- **`observability_demo.rs`** - Automatically loads `datasets/data_10k_1024d_100clusters.bin` if available
- **`inspect_dataset.rs`** - Inspects dataset statistics and checks for data quality issues

