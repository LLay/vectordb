# Vector Datasets

This directory contains pre-generated vector datasets for testing and benchmarking.

## Quick Start

Generate all datasets:

```bash
cargo run --release --example generate_datasets
```

## Available Datasets

| File | Vectors | Clusters | Size | Description |
|------|---------|----------|------|-------------|
| `data_10k_1024d_10clusters.bin` | 10K | 10 | 39 MB | Small, few clusters |
| `data_10k_1024d_50clusters.bin` | 10K | 50 | 39 MB | Medium clusters |
| `data_10k_1024d_100clusters.bin` | 10K | 100 | 39 MB | Many clusters |
| `data_100k_1024d_100clusters.bin` | 100K | 100 | 391 MB | Large dataset |
| `data_10k_1024d_random.bin` | 10K | - | 39 MB | Uniform random (baseline) |

**Total size: ~560 MB**

## Dataset Format

Binary format with little-endian encoding:

```
[num_vectors: u32][dims: u32][vector1_f32...][vector2_f32...]...
```

## Why These Datasets?

Real-world vector embeddings (from text, images, etc.) have **natural cluster structure**. These datasets simulate that structure using Gaussian distributions.

### Performance Comparison

| Dataset | Probes=10 | Probes=50 |
|---------|-----------|-----------|
| Random | 3.5% recall | 11% recall |
| 100 Clusters | **63% recall** | **100% recall** |

The hierarchical k-means index performs **dramatically better** on clustered data!

## Usage Examples

See the main [DATASETS_README.md](../DATASETS_README.md) for code examples and more details.

---

**Note**: These `.bin` files are gitignored. Run `generate_datasets` to create them locally.
