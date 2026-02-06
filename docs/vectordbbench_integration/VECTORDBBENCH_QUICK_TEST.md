# VectorDBBench Quick Testing Workflow

> **Note**: For complete integration guide and setup instructions, see [VECTORDBBENCH_INTEGRATION.md](./VECTORDBBENCH_INTEGRATION.md)

This is a quick reference for fast iterative development. For newcomers, start with the integration guide above.

## Fast Iterative Development

### Step 1: Load Data Once (Skip Search)
```bash
vectordbbench rustvectordb --case-type Performance1536D50K \
  --skip-search-serial --skip-search-concurrent
```
- **Duration**: ~4 minutes (239s)
- **What it does**: Loads 50K vectors, builds index, saves to `/tmp/vectordb_bench.bin.staging`
- **Use when**: First run, or after code changes to data loading/indexing

### Step 2: Debug Search (Skip Load)
```bash
vectordbbench rustvectordb --case-type Performance1536D50K \
  --skip-load --skip-drop-old --skip-search-concurrent
```
- **Duration**: ~50 seconds (44s rebuild + 5s search)
- **What it does**: Rebuilds index from staging file, runs search
- **Use when**: Debugging search parameters, testing recall, etc.

### Step 3: Full Benchmark (Load + Search)
```bash
vectordbbench rustvectordb --case-type Performance1536D50K \
  --skip-search-concurrent
```
- **Duration**: ~4.5 minutes (load + optimize + search)
- **What it does**: Complete end-to-end benchmark
- **Use when**: Final testing, comparing against other DBs

## Available Skip Flags

- `--skip-drop-old`: Don't delete existing index (for reuse)
- `--skip-load`: Skip data loading phase
- `--skip-search-serial`: Skip serial search testing
- `--skip-search-concurrent`: Skip concurrent search testing

## Current Performance (50K OpenAI Vectors, 1536D)

- **Load**: 253s (insert) + 80s (optimize) = 333s total
- **Search**: 5.4s for 1000 queries (186 QPS)
  - p99 latency: 12.6ms
  - p95 latency: 6.0ms
  - **Recall**: 95.61% ✅
  - **NDCG**: 96.17%
- **Index**: ~530 leaves, depth=3, 18.7MB RAM

## Parameters

```bash
vectordbbench rustvectordb \
  --branching-factor 100 \      # Tree width
  --target-leaf-size 100 \      # Vectors per leaf
  --probes 100 \                # Clusters to search
  --rerank-factor 10            # Reranking multiplier
```

## Available Datasets

- `Performance1536D50K`: OpenAI 50K (COSINE)
- `Performance1536D500K`: OpenAI 500K (COSINE)
- `Performance1536D5M`: OpenAI 5M (COSINE)
- `Performance768D1M`: Cohere 1M (COSINE)
- `Performance768D10M`: Cohere 10M (COSINE)

## Troubleshooting

### If "Index not optimized" error
- Make sure staging file exists: `/tmp/vectordb_bench.bin.staging`
- Try without `--skip-load` first to build fresh

### If slow data loading (45+ minutes)
- This was fixed! Should now be ~4 minutes for 50K vectors
- O(n²) write issue resolved with append-based staging file

## Next Steps

1. **Test with larger datasets** (Cohere 1M, OpenAI 500K)
2. **Compare against industry DBs** (Milvus, Weaviate, Qdrant)
3. **Optimize search parameters** for recall/latency tradeoff
4. **Profile and optimize** hot paths identified in benchmarks
