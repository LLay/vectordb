# Quick Start: Memory-Mapped Storage

## TL;DR

Save ~97% RAM by storing vectors on disk with memory-mapped files:

```rust
use vectordb::{ClusteredIndex, DistanceMetric};

// Build index (in-memory)
let mut index = ClusteredIndex::build(
    vectors,
    10,      // branching_factor
    1000,    // max_leaf_size  
    DistanceMetric::L2,
    20,      // max_iterations
);

// Convert to mmap (saves RAM)
index.use_mmap_storage("vectors.bin")?;

// Query as normal (1.1-1.5x slower when warm)
let results = index.search(&query, 10, 4, 5);
```

## Memory Savings

| Dataset | In-Memory | Mmap | Savings |
|---------|-----------|------|---------|
| 100K @ 768d | 290 MB | 10 MB | 97% |
| 1M @ 768d | 2.9 GB | 100 MB | 97% |
| 10M @ 768d | 29 GB | 1 GB | 97% |

## Performance Trade-off

```
┌─────────────┬──────────────┬──────────────────┐
│ Access Type │ Latency      │ Description      │
├─────────────┼──────────────┼──────────────────┤
│ In-Memory   │ 1.0x         │ Baseline         │
│ Mmap (cold) │ 2-5x slower  │ First access     │
│ Mmap (warm) │ 1.1-1.5x     │ OS cached        │
└─────────────┴──────────────┴──────────────────┘
```

## When to Use

✅ **Use Mmap When:**
- You have limited RAM (< dataset size)
- Can tolerate 10-50% latency increase
- Need to scale to larger datasets
- Working set fits in available RAM

❌ **Don't Use Mmap When:**
- Latency is critical and RAM is available
- Dataset is very small (<100k vectors)
- Working set > available RAM (causes thrashing)

## Check Status

```rust
// Check memory usage
let mb = index.memory_usage_bytes() as f64 / 1_048_576.0;
println!("RAM usage: {:.2} MB", mb);

// Check storage type
if index.is_using_mmap() {
    println!("Using disk-backed storage");
} else {
    println!("Using in-memory storage");
}
```

## Error Handling

```rust
match index.use_mmap_storage("vectors.bin") {
    Ok(_) => println!("Converted to mmap storage"),
    Err(e) => eprintln!("Error: {}", e),
}
```

Common errors:
- File I/O errors (permissions, disk space)
- Already using mmap (can't convert twice)
- Invalid file (size mismatch on open)

## Example: 1M Vectors on Laptop

```rust
use vectordb::{ClusteredIndex, DistanceMetric};

fn main() -> std::io::Result<()> {
    // Load/generate 1M vectors @ 768d
    let vectors = load_vectors(1_000_000, 768);
    
    println!("Building index...");
    let mut index = ClusteredIndex::build(
        vectors,
        10,
        1000,
        DistanceMetric::L2,
        20,
    );
    
    println!("Memory usage: {:.2} GB", 
        index.memory_usage_bytes() as f64 / 1e9);
    // Output: Memory usage: 2.90 GB
    
    // Convert to mmap to save RAM
    index.use_mmap_storage("vectors_1m.bin")?;
    
    println!("Memory usage: {:.2} MB",
        index.memory_usage_bytes() as f64 / 1e6);
    // Output: Memory usage: 100 MB
    
    // Query works the same
    let query = vec![0.5; 768];
    let results = index.search(&query, 10, 4, 5);
    
    println!("Found {} results", results.len());
    // Output: Found 10 results
    
    Ok(())
}
```

## Benchmarking

Run the demo to measure performance on your hardware:

```bash
cargo run --release --example mmap_demo
```

This will:
1. Build index with 100k vectors
2. Measure in-memory latency
3. Convert to mmap
4. Measure cold and warm mmap latency
5. Report memory savings

Expected output:
```
Memory saved: ~290 MB (97%)
In-Memory latency:  0.45 ms
Mmap warm latency:  0.55 ms (1.2x slower)

Conclusion: Mmap adds ~1.2x overhead when warm, but saves 290 MB RAM
```

## Architecture

The hierarchical index uses a hybrid approach:

```
Query → Binary Tree → Binary Vectors → Full Vectors
        (in RAM)      (in RAM, 32x)    (mmap or RAM)
```

Only the final reranking phase accesses full vectors via mmap.
The OS automatically caches hot vectors, so frequently accessed
vectors stay in RAM.

## Next Steps

- Read `MMAP_STORAGE.md` for detailed documentation
- See `src/storage/mmap.rs` for implementation
- Run `examples/mmap_demo.rs` for benchmarks

## Troubleshooting

**Q: Queries are very slow**
A: First queries are cold (disk I/O). Run multiple times to warm OS cache.

**Q: Still using a lot of RAM**
A: Check `is_using_mmap()`. Tree structure and binary vectors still in RAM.

**Q: Getting "Already using mmap" error**
A: Can't convert twice. Rebuild index if you need to change file path.

**Q: File size doesn't match expectations**
A: File size = `num_vectors × dimension × 4 bytes` (f32)

**Q: Better performance than expected**
A: OS is caching vectors in RAM. This is good! It's working as designed.
