# Memory-Mapped Storage

Memory-mapped storage allows the hierarchical index to store full-precision vectors on disk instead of in RAM, significantly reducing memory usage while maintaining good query performance.

## How It Works

1. **Memory Mapping**: The OS maps the vector file directly into the process's address space
2. **Zero-Copy Access**: Vectors are accessed directly from the mapped region without copying
3. **Automatic Caching**: The OS automatically caches frequently accessed pages in RAM
4. **Lazy Loading**: Only accessed vectors are loaded into RAM

## Usage

### Basic Example

```rust
use vectordb::{ClusteredIndex, DistanceMetric};

// Build index normally (in-memory)
let mut index = ClusteredIndex::build(
    vectors,
    10,    // branching_factor
    1000,  // max_leaf_size
    DistanceMetric::L2,
    20,    // max_iterations
);

// Convert to memory-mapped storage
index.use_mmap_storage("vectors.bin")?;

// Query as normal - the API doesn't change
let results = index.search(&query, k, probes, rerank_factor);
```

### Checking Memory Usage

```rust
// Check current memory usage
let mem_bytes = index.memory_usage_bytes();
println!("Memory usage: {:.2} MB", mem_bytes as f64 / 1_048_576.0);

// Check if using mmap
if index.is_using_mmap() {
    println!("Using memory-mapped storage");
} else {
    println!("Using in-memory storage");
}
```

## Performance Characteristics

### Memory Savings

For 1M vectors with dimension 768:
- **In-memory**: ~2.9 GB RAM (full precision f32 vectors)
- **Mmap**: ~100 MB RAM (quantized vectors + index structure only)
- **Savings**: ~97% memory reduction

### Query Performance

- **Cold queries** (first access): 2-5x slower due to disk I/O
- **Warm queries** (cached by OS): 1.1-1.5x slower than in-memory
- **Hot queries** (recently accessed): Near in-memory performance

### When to Use Mmap

Use memory-mapped storage when:
- ✅ You have limited RAM relative to dataset size
- ✅ Your working set fits in available RAM (OS can cache hot vectors)
- ✅ You can tolerate 10-50% query latency increase
- ✅ You need to scale to larger datasets without buying more RAM

Don't use mmap when:
- ❌ Query latency is critical and you have enough RAM
- ❌ Dataset is very small (<100k vectors)
- ❌ Working set is larger than available RAM (causes thrashing)

## Architecture

```
┌─────────────────────────────────────┐
│      ClusteredIndex                 │
├─────────────────────────────────────┤
│ Tree Structure (in RAM)             │
│  - Nodes                            │
│  - Binary centroids                 │
│                                     │
│ Binary Vectors (in RAM)             │
│  - Pre-quantized, 32x compressed    │
│  - Used for candidate filtering    │
│                                     │
│ Full Vectors (storage backend)      │
│  ┌──────────────┬─────────────────┐ │
│  │  In-Memory   │      Mmap       │ │
│  │              │                 │ │
│  │ Vec<Vec<f32>>│ MmapVectorStore │ │
│  │              │       ↓         │ │
│  │  ~3 GB RAM   │  vectors.bin    │ │
│  │              │   (on disk)     │ │
│  │              │   ~100 MB RAM   │ │
│  └──────────────┴─────────────────┘ │
└─────────────────────────────────────┘
```

## Query Flow with Mmap

1. **Quantized Tree Navigation** (in RAM, fast)
   - Uses binary centroids for tree traversal
   - Hamming distance on quantized query

2. **Candidate Filtering** (in RAM, fast)
   - Uses pre-quantized binary vectors
   - Parallel Hamming distance computation
   - Selects top-k × rerank_factor candidates

3. **Reranking** (mmap, potentially disk I/O)
   - Accesses full precision vectors via mmap
   - OS loads pages on demand
   - Hot vectors cached in RAM by OS
   - Computes exact distances

## OS Page Cache Behavior

The OS maintains a page cache for memory-mapped files:

- **Page Size**: Typically 4-16 KB (varies by OS/architecture)
- **Cache Size**: Uses available free RAM
- **Eviction**: LRU-based when memory pressure occurs
- **Prefetch**: OS may prefetch adjacent pages

### Maximizing Cache Efficiency

1. **Sequential Layout**: Vectors are stored sequentially in the file
2. **Clustered Access**: Hierarchical index queries nearby vectors together
3. **Batch Reranking**: Multiple candidates reranked together
4. **Warm-up**: Run queries to populate cache before measuring performance

## Implementation Details

### File Format

The vector file uses a simple binary format:
```
[vector_0] [vector_1] [vector_2] ... [vector_n]
```

Each vector is `dimension × 4 bytes` (f32):
```
[f32][f32][f32]...[f32]
```

No header or metadata - dimension and count are known by the index.

### Safety

- Memory mapping uses `unsafe` but is well-encapsulated
- File size is validated on open
- Bounds checking on vector access
- Thread-safe (implements `Send + Sync`)

### Error Handling

```rust
match index.use_mmap_storage("vectors.bin") {
    Ok(_) => println!("Successfully converted to mmap"),
    Err(e) => eprintln!("Error: {}", e),
}
```

Common errors:
- File I/O errors (permissions, disk space)
- Already using mmap (can't convert twice)
- File size mismatch (corrupted file)

## Benchmarking Mmap Performance

Use the provided example to measure mmap performance on your hardware:

```bash
cargo run --release --example mmap_demo
```

This will:
1. Build an index with 100k vectors (768d)
2. Measure in-memory query latency
3. Convert to mmap storage
4. Measure cold and warm mmap latency
5. Report memory savings and performance overhead

Expected output:
```
Memory saved: ~290 MB (97%)
Mmap warm latency: 1.2x slower than in-memory
```

## Future Optimizations

Potential improvements for mmap performance:

1. **madvise hints**: Tell OS about access patterns
2. **Prefetch API**: Explicit prefetching of vectors
3. **Custom page layout**: Optimize vector layout for cache locality
4. **Compression**: Compress vectors on disk (with decompression in cache)
5. **Tiered storage**: Hot vectors in RAM, warm in mmap, cold in compressed storage

## Related Documentation

- `DESIGN.md` - Overall architecture
- `src/storage/mmap.rs` - Memory-mapped storage implementation
- `src/index/hierarchical.rs` - Index integration
- `examples/mmap_demo.rs` - Performance demonstration
