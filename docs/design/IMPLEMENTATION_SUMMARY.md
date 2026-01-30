# Implementation Summary: Memory-Mapped Storage

## Overview

Implemented memory-mapped storage for the hierarchical clustered index, enabling efficient disk-backed vector storage with significant RAM savings while maintaining good query performance.

## What Was Implemented

### 1. Core Memory-Mapped Storage (`src/storage/mmap.rs`)

A zero-copy memory-mapped vector store with:

- **`MmapVectorStore`**: Main storage structure using `memmap2` crate
- **`create()`**: Write vectors to file and create mmap
- **`open()`**: Open existing vector file as mmap
- **`get(idx)`**: Zero-copy vector access via memory mapping
- **Thread Safety**: Implements `Send + Sync` for parallel access
- **Safety**: Encapsulated `unsafe` with proper validation

Key features:
- No copying - vectors accessed directly from mapped memory
- OS-managed caching - hot vectors stay in RAM automatically
- File size validation
- Bounds checking on access

### 2. Index Integration (`src/index/hierarchical.rs`)

Extended `ClusteredIndex` with pluggable storage backend:

**New Abstractions:**
```rust
enum VectorStorage {
    InMemory(Vec<Vec<f32>>),  // Original behavior
    Mmap(MmapVectorStore),     // New disk-backed storage
}
```

**New Methods:**
- `use_mmap_storage(path)`: Convert index to use mmap
- `is_using_mmap()`: Check storage backend type
- `memory_usage_bytes()`: Estimate RAM usage

**Design:**
- Transparent to search algorithm - API unchanged
- Opt-in conversion after index build
- All existing functionality preserved

### 3. Testing

**Unit Tests:**
- `test_create_and_open`: File I/O and reopening
- `test_large_vectors`: Handling realistic sizes (1024d × 100 vectors)
- `test_zero_copy`: Verify pointers are same (no copying)
- `test_out_of_bounds`: Safety checks

**Integration Tests:**
- `test_mmap_storage`: Full workflow (build → convert → query)
- `test_mmap_storage_twice_fails`: Error handling

All 26 tests pass ✅

### 4. Documentation

- **`MMAP_STORAGE.md`**: Comprehensive guide covering:
  - How it works (OS memory mapping)
  - Usage examples
  - Performance characteristics
  - When to use vs. avoid
  - Architecture diagrams
  - File format specification
  - Future optimizations

- **`examples/mmap_demo.rs`**: Runnable performance demonstration
  - Compares in-memory vs. mmap latency
  - Shows cold vs. warm cache behavior
  - Measures memory savings

## Performance Impact

### Memory Savings

For 1M vectors @ 768 dimensions:
```
In-Memory:  ~2,900 MB (full f32 vectors)
Mmap:       ~100 MB (quantized + index only)
Savings:    ~97% RAM reduction
```

### Query Latency

Expected performance:
```
Cold queries:  2-5x slower (disk I/O on first access)
Warm queries:  1.1-1.5x slower (OS cache hit)
Hot queries:   Near in-memory (cached by OS)
```

The overhead comes from:
1. Page fault handling when accessing unmapped pages
2. Indirect access through mmap pointer
3. Potential TLB misses

## Architecture

```
ClusteredIndex
├── Tree Structure (in RAM)
│   └── Nodes with binary centroids
├── Binary Vectors (in RAM, 32x compressed)
│   └── Used for candidate filtering
└── Full Vectors (pluggable storage)
    ├── VectorStorage::InMemory
    │   └── Vec<Vec<f32>> (~3 GB)
    └── VectorStorage::Mmap
        └── MmapVectorStore → vectors.bin (~100 MB RAM)
```

## Query Flow

1. **Tree Navigation** (RAM)
   - Binary centroids
   - Hamming distance

2. **Candidate Filtering** (RAM)
   - Pre-quantized binary vectors
   - Parallel Hamming distance
   - Top-k × rerank_factor selection

3. **Reranking** (Mmap - may trigger disk I/O)
   - Full precision vectors via `VectorStorage::get()`
   - OS loads pages on demand
   - Exact distance computation

## Key Design Decisions

### 1. Pluggable Storage Backend
**Decision:** Abstract storage with enum rather than traits
**Rationale:** 
- Simpler implementation
- Zero-cost abstraction via match/inline
- Easy to add more backends later

### 2. Opt-in Conversion
**Decision:** Build in-memory, then convert to mmap
**Rationale:**
- Index building is already expensive
- Mmap overhead doesn't matter during build
- Users can choose based on their constraints

### 3. Zero-Copy Access
**Decision:** Return `&[f32]` slices directly from mmap
**Rationale:**
- No allocation/copying overhead
- Safe via proper bounds checking
- Maximizes cache efficiency

### 4. No Custom Caching
**Decision:** Rely on OS page cache, not application-level LRU
**Rationale:**
- OS cache is highly optimized
- Avoids memory overhead of tracking
- Simpler implementation
- Can add later if needed

## Usage Example

```rust
// Build index (in-memory)
let mut index = ClusteredIndex::build(
    vectors,
    10,    // branching_factor
    1000,  // max_leaf_size
    DistanceMetric::L2,
    20,    // max_iterations
);

// Convert to mmap to save RAM
index.use_mmap_storage("vectors.bin")?;

// Query as normal - API unchanged
let results = index.search(&query, k, probes, rerank_factor);
```

## Future Optimizations

Potential improvements:
1. **madvise hints**: `MADV_RANDOM`, `MADV_WILLNEED`, etc.
2. **Prefetch API**: Explicit prefetching of likely-needed vectors
3. **Vector layout optimization**: Cluster layout for cache locality
4. **Compression**: Store compressed, decompress on access
5. **Tiered storage**: Hot vectors in RAM, warm in mmap, cold compressed

## Files Changed

**New Files:**
- `src/storage/mmap.rs` - Memory-mapped store implementation
- `examples/mmap_demo.rs` - Performance demonstration
- `MMAP_STORAGE.md` - User documentation
- `IMPLEMENTATION_SUMMARY.md` - This file

**Modified Files:**
- `src/storage/mod.rs` - Export mmap module
- `src/index/hierarchical.rs` - Add VectorStorage abstraction and methods
- `Cargo.toml` - Add `memmap2` dependency (already present)

**Test Coverage:**
- 4 new unit tests in `storage::mmap`
- 2 new integration tests in `index::hierarchical`
- All 26 tests passing

## Dependencies

- `memmap2 = "0.9"` - Memory mapping library
  - Well-maintained
  - Cross-platform (Linux, macOS, Windows)
  - Safe abstractions over `mmap()`

## Next Steps

With memory-mapped storage implemented, the system now supports:
- ✅ Binary quantization (32x compression)
- ✅ Quantized tree navigation (fast traversal)
- ✅ Parallel candidate filtering
- ✅ Parallel reranking
- ✅ Memory-mapped storage (RAM savings)

Remaining optimizations for laptop performance:
1. **LRU Caching**: Application-level cache of hot vectors
2. **Batch Query Optimization**: Amortize tree traversal across queries
3. **SIMD Hamming Distance**: Further optimize binary distance
4. **Index Serialization**: Save/load index to avoid rebuild
5. **Quantization Tuning**: Experiment with PQ, SQ variants

The foundation is solid - all core infrastructure is in place for scaling to 1M+ vectors on laptop hardware.
