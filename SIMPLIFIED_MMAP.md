# Memory-Mapped Storage (Simplified)

The `ClusteredIndex` now **always** uses memory-mapped storage for full-precision vectors. This design simplifies the API while maximizing RAM efficiency.

## Quick Start

```rust
use vectordb::{ClusteredIndex, DistanceMetric};

// Build index - vectors are automatically stored on disk
let index = ClusteredIndex::build(
    vectors,
    "vectors.bin",  // File path for mmap storage
    10,             // branching_factor
    1000,           // max_leaf_size  
    DistanceMetric::L2,
    20,             // max_iterations
)?;

// Query as normal
let results = index.search(&query, 10, 4, 5);
```

## What Changed

### Before (Pluggable Storage)
```rust
// Build in-memory
let mut index = ClusteredIndex::build(vectors, 10, 1000, DistanceMetric::L2, 20);

// Optionally convert to mmap
index.use_mmap_storage("vectors.bin")?;
```

### Now (Always Mmap)
```rust
// Build with mmap (required)
let index = ClusteredIndex::build(
    vectors, 
    "vectors.bin",  // mmap file path
    10, 
    1000, 
    DistanceMetric::L2, 
    20
)?;
```

## Benefits

1. **Simpler API**: No need to decide between storage backends
2. **Consistent Performance**: All code paths use the same storage
3. **RAM Efficient**: Always uses minimal RAM (~97% savings)
4. **OS-Optimized Caching**: Leverages OS page cache automatically

## Architecture

```
ClusteredIndex
├── Tree Structure (RAM)
│   └── Nodes with binary centroids
├── Binary Vectors (RAM, 32x compressed)
│   └── Used for candidate filtering
└── Full Vectors (Memory-Mapped)
    └── MmapVectorStore → vectors.bin on disk
```

## Memory Usage

| Component | Storage | Size (1M @ 768d) |
|-----------|---------|------------------|
| Tree nodes | RAM | ~10 MB |
| Binary vectors | RAM | ~90 MB |
| Full vectors | Mmap (disk) | ~2.9 GB |
| **Total RAM** | | **~100 MB** |

## Performance

- **Cold queries** (first access): 2-5x slower due to disk I/O
- **Warm queries** (OS cached): 1.1-1.5x slower than pure in-memory
- **Hot queries** (recently accessed): Near in-memory performance

The OS automatically caches frequently accessed vectors, so performance improves with repeated queries.

## API Reference

### Build Index
```rust
pub fn build<P: AsRef<Path>>(
    vectors: Vec<Vec<f32>>,
    vector_file: P,              // NEW: required file path
    branching_factor: usize,
    max_leaf_size: usize,
    metric: DistanceMetric,
    max_iterations: usize,
) -> std::io::Result<Self>
```

### Memory Monitoring
```rust
// RAM usage (excludes mmap file)
let ram_mb = index.memory_usage_bytes() as f64 / 1_048_576.0;

// Disk usage (mmap file size)
let disk_mb = index.disk_usage_bytes() as f64 / 1_048_576.0;
```

### Search (Unchanged)
```rust
pub fn search(
    &self,
    query: &[f32],
    k: usize,
    probes_per_level: usize,
    rerank_factor: usize,
) -> Vec<(usize, f32)>
```

## Example

See `examples/mmap_demo.rs` for a complete demonstration:

```bash
cargo run --release --example mmap_demo
```

Expected output:
```
Building hierarchical index with mmap storage...
  Vector file: 290.00 MB

RAM usage: 100.23 MB
Disk usage: 290.00 MB

Cold latency:  2.45 ms
Warm latency:  0.58 ms

RAM saved: 189.77 MB (65.4%)
```

## Error Handling

```rust
match ClusteredIndex::build(vectors, "vectors.bin", 10, 1000, DistanceMetric::L2, 20) {
    Ok(index) => {
        // Use index
    }
    Err(e) => {
        eprintln!("Failed to build index: {}", e);
    }
}
```

Common errors:
- File I/O errors (permissions, disk space)
- Invalid vector dimensions
- Empty vector set

## Migration Guide

If you have existing code using the old API:

1. Add a file path parameter to `build()`
2. Remove calls to `use_mmap_storage()` and `is_using_mmap()`
3. Handle `Result` from `build()` (now returns `io::Result`)

### Before
```rust
let index = ClusteredIndex::build(vectors, 10, 1000, DistanceMetric::L2, 20);
index.use_mmap_storage("vectors.bin").unwrap();
```

### After
```rust
let index = ClusteredIndex::build(
    vectors, 
    "vectors.bin", 
    10, 
    1000, 
    DistanceMetric::L2, 
    20
).unwrap();
```

## Related Documentation

- `MMAP_STORAGE.md` - Detailed technical documentation
- `src/storage/mmap.rs` - Implementation details
- `examples/mmap_demo.rs` - Performance demonstration
