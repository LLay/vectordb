# VectorDBBench Integration Guide

## Overview

This Rust VectorDB implementation is integrated with [VectorDBBench](https://github.com/zilliztech/VectorDBBench), an industry-standard benchmarking framework for vector databases. This allows you to:

- **Compare performance** against production databases (Milvus, Weaviate, Qdrant, Pinecone, etc.)
- **Test at scale** with real-world datasets (OpenAI embeddings, Cohere embeddings, etc.)
- **Measure key metrics** like recall, latency (p95/p99), throughput, and memory usage
- **Validate improvements** with standardized test cases

## Quick Start

### 1. Install VectorDBBench

```bash
# Clone VectorDBBench (if not already cloned)
cd ~/projects
git clone https://github.com/zilliztech/VectorDBBench.git

# Create virtual environment for benchmarking
cd ~/workspace/vectordb/vectordb
python3 -m venv venv_bench
source venv_bench/bin/activate

# Install VectorDBBench
cd ~/projects/VectorDBBench
pip install -e .
```

### 2. Build and Install Rust VectorDB Client

```bash
cd ~/workspace/vectordb/vectordb

# Build PyO3 bindings (creates Python module from Rust)
./build_python_bindings.sh release

# Install VectorDB client into VectorDBBench
./install_to_vectordbbench.sh
```

### 3. Run Your First Benchmark

```bash
# Activate the benchmarking environment
source venv_bench/bin/activate

# Run a quick test (50K vectors, 1536 dimensions)
vectordbbench rustvectordb --case-type Performance1536D50K --skip-search-concurrent
```

**Expected Results** (for 50K OpenAI vectors):
- Load time: ~5 minutes
- Recall: ~95%
- p99 latency: ~13ms
- p95 latency: ~6ms

## Integration Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      VectorDBBench                           │
│  (Python framework for standardized vector DB testing)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Python API calls
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Rust VectorDB Client                            │
│    (vectordb_bench/backend/clients/rust_vectordb/)          │
│                                                              │
│  - rust_vectordb.py:  VectorDB interface implementation     │
│  - config.py:         Configuration classes                 │
│  - cli.py:            Command-line interface                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ PyO3 bindings
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Rust VectorDB Core                          │
│              (vectordb Python module)                        │
│                                                              │
│  - src/python_bindings.rs:  PyO3 wrapper (PyVectorDB)       │
│  - src/index/*:              Hierarchical index + RaBitQ    │
│  - src/quantization/*:       Vector quantization            │
│  - src/storage/*:            Memory-mapped storage          │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **PyO3 Bindings** (`src/python_bindings.rs`)
   - Exposes Rust VectorDB to Python as `PyVectorDB` class
   - Handles GIL (Global Interpreter Lock) management for parallelism
   - Implements pickle support for multiprocessing

2. **VectorDBBench Client** (`vectordb_bench/backend/clients/rust_vectordb/`)
   - Implements VectorDBBench's `VectorDB` abstract interface
   - Manages ID mapping (VectorDBBench IDs ↔ array indices)
   - Handles configuration and CLI integration

3. **Multiprocessing Support**
   - Data persisted to staging files (`/tmp/vectordb_bench.bin.staging`)
   - ID mappings saved to pickle files (`/tmp/vectordb_bench.bin.idmap`)
   - Index automatically rebuilt in subprocesses

## Available Commands

### Basic Benchmarks

```bash
# 50K vectors, 1536D (OpenAI embeddings) - ~5 min
vectordbbench rustvectordb --case-type Performance1536D50K --skip-search-concurrent

# 500K vectors, 1536D (OpenAI embeddings) - ~30 min
vectordbbench rustvectordb --case-type Performance1536D500K --skip-search-concurrent

# 1M vectors, 768D (Cohere embeddings) - ~45 min
vectordbbench rustvectordb --case-type Performance768D1M --skip-search-concurrent

# 5M vectors, 1536D (OpenAI embeddings) - ~3 hours
vectordbbench rustvectordb --case-type Performance1536D5M --skip-search-concurrent
```

### Custom Parameters

```bash
vectordbbench rustvectordb \
  --case-type Performance1536D50K \
  --branching-factor 100 \      # Tree width at each level
  --target-leaf-size 100 \      # Vectors per leaf node
  --probes 100 \                # Number of clusters to search
  --rerank-factor 10 \          # Reranking multiplier (k × rerank_factor candidates)
  --skip-search-concurrent
```

### Fast Iteration Workflow

**Load data once:**
```bash
vectordbbench rustvectordb --case-type Performance1536D50K \
  --skip-search-serial --skip-search-concurrent
```

**Iterate on search parameters (~50 seconds per run):**
```bash
vectordbbench rustvectordb --case-type Performance1536D50K \
  --skip-load --skip-drop-old --skip-search-concurrent \
  --probes 50 --rerank-factor 20
```

This workflow is ideal for:
- Testing different search parameters
- Debugging recall issues
- Optimizing latency vs. recall tradeoffs

## Understanding Results

### Metrics Explained

When a benchmark completes, you'll see output like:

```
| DB           | db_label | case                                            | load_dur | qps | latency(p99) | latency(p95) | recall  | max_load_count |
| RustVectorDB |          | Search Performance Test (50K Dataset, 1536 Dim) | 333.15   | 0.0 | 0.0126       | 0.006        | 0.9561  | 0              |
```

**Key Metrics:**

- **load_dur**: Total time to load and index data (seconds)
- **recall**: Percentage of true nearest neighbors found (0.0-1.0)
  - 0.95+ is excellent
  - 0.90-0.95 is good
  - <0.90 needs tuning
- **latency (p99)**: 99th percentile query latency (seconds)
  - The slowest 1% of queries
- **latency (p95)**: 95th percentile query latency (seconds)
  - The slowest 5% of queries
- **qps**: Queries per second (only for concurrent search tests)

### Results Files

Results are saved to:
```
venv_bench/lib/python3.14/site-packages/vectordb_bench/results/RustVectorDB/
```

Each run generates a JSON file with detailed metrics.

## Parameter Tuning Guide

### Balancing Recall vs. Latency

The main tradeoff in ANN search is recall (accuracy) vs. latency (speed):

**Higher Recall** (slower, more accurate):
```bash
--probes 100 --rerank-factor 20
```

**Lower Latency** (faster, less accurate):
```bash
--probes 30 --rerank-factor 5
```

**Balanced** (default):
```bash
--probes 100 --rerank-factor 10
```

### Build-Time Parameters

These affect index construction (slower to build = better quality):

- `--branching-factor`: Tree width
  - Higher (100-200): Shallower tree, better for large datasets
  - Lower (20-50): Deeper tree, better for small datasets

- `--target-leaf-size`: Vectors per leaf
  - Higher (100-200): Fewer leaves, faster build
  - Lower (30-50): More leaves, better locality

## Comparing Against Other Databases

### Running Side-by-Side Comparisons

```bash
# Run Rust VectorDB
vectordbbench rustvectordb --case-type Performance1536D50K --skip-search-concurrent

# Run Milvus (example)
vectordbbench milvus --case-type Performance1536D50K --skip-search-concurrent

# Run Weaviate (example)
vectordbbench weaviate --case-type Performance1536D50K --skip-search-concurrent
```

### Viewing Results

VectorDBBench saves all results and allows comparison through its web UI:

```bash
# Start the VectorDBBench web interface
init_bench

# Open browser to http://localhost:8501
# Navigate to "View Results" to compare runs
```

## Troubleshooting

### "Cannot import vectordb module"

```bash
# Rebuild PyO3 bindings
cd ~/workspace/vectordb/vectordb
./build_python_bindings.sh release
```

### "Index not optimized" error

The staging file may be missing. Run a full load:
```bash
vectordbbench rustvectordb --case-type Performance1536D50K --skip-search-concurrent
```

### Low recall (<0.5)

1. Check that metric type matches dataset (Cosine vs. L2)
2. Increase search parameters:
   ```bash
   --probes 150 --rerank-factor 25
   ```
3. Verify ID mapping file exists: `/tmp/vectordb_bench.bin.idmap`

### Slow data loading (>10 minutes for 50K)

This should be fixed in the current implementation (uses O(n) append-based writes).
If still slow, check:
- Disk I/O performance
- Available disk space in `/tmp/`

### "No ID mapping found" warning

The ID mapping is saved to `/tmp/vectordb_bench.bin.idmap`. If this file is missing:
```bash
# Reload data to regenerate ID mapping
vectordbbench rustvectordb --case-type Performance1536D50K \
  --skip-search-serial --skip-search-concurrent
```

### Python version errors

VectorDBBench requires Python 3.9+. If using Python 3.14:
```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
./build_python_bindings.sh release
```

## Development Workflow

### Making Changes to the Rust Code

```bash
cd ~/workspace/vectordb/vectordb

# 1. Make changes to Rust code (src/*)

# 2. Rebuild PyO3 bindings
./build_python_bindings.sh release

# 3. Test changes (no need to reinstall VectorDBBench client)
source venv_bench/bin/activate
vectordbbench rustvectordb --case-type Performance1536D50K --skip-search-concurrent
```

### Making Changes to the Python Client

```bash
cd ~/workspace/vectordb/vectordb

# 1. Make changes to vectordbbench_client/rust_vectordb.py

# 2. Copy to VectorDBBench installation
./install_to_vectordbbench.sh

# 3. Reinstall VectorDBBench (quick)
cd ~/projects/VectorDBBench
source ~/workspace/vectordb/vectordb/venv_bench/bin/activate
pip install --no-cache-dir -e .

# 4. Test
vectordbbench rustvectordb --case-type Performance1536D50K --skip-search-concurrent
```

### Running Quick Tests

Use the skip-load workflow for fast iteration:

```bash
# Load once (~5 min)
vectordbbench rustvectordb --case-type Performance1536D50K \
  --skip-search-serial --skip-search-concurrent

# Test repeatedly (~50 sec each)
vectordbbench rustvectordb --case-type Performance1536D50K \
  --skip-load --skip-drop-old --skip-search-concurrent \
  --probes 80 --rerank-factor 15  # Try different parameters
```

## Technical Deep Dive

### Multiprocessing Architecture

VectorDBBench uses Python's multiprocessing for parallelism. Our implementation handles this through:

1. **Pickle Support** (`__getstate__`/`__setstate__`):
   - PyO3 objects can't be pickled
   - We serialize configuration, not the Rust object
   - Rebuild index in each subprocess

2. **Persistent Storage**:
   - Vectors: `/tmp/vectordb_bench.bin.staging` (binary format)
   - ID mapping: `/tmp/vectordb_bench.bin.idmap` (pickle format)
   - Index data: `/tmp/vectordb_bench.bin` (memory-mapped)

3. **Subprocess Index Rebuilding**:
   - Each search subprocess loads vectors from staging file
   - Rebuilds index (takes ~45 seconds for 50K vectors)
   - Loads ID mapping for correct result translation

### ID Mapping

VectorDBBench assigns integer IDs to vectors. Our Rust index uses array indices (0, 1, 2...). We maintain a bidirectional mapping:

```
VectorDBBench ID → Array Index → Search Results → VectorDBBench ID
     42015      →      0       →     (0, 0.15)  →     42015
     19382      →      1       →     (1, 0.23)  →     19382
     ...
```

This mapping is:
- Built during `insert_embeddings()`
- Saved to `/tmp/vectordb_bench.bin.idmap`
- Loaded in subprocesses via `__setstate__()`
- Applied in `search_embedding()` to translate results

### Performance Optimizations

1. **GIL Management**: Use `py.allow_threads()` during heavy Rust operations
2. **Append-based Writes**: O(n) staging file writes, not O(n²)
3. **Memory-mapped Storage**: Minimal RAM usage for full-precision vectors
4. **RaBitQ Quantization**: Fast filtering with binary codes + reranking

## Available Datasets

| Dataset | Vectors | Dimensions | Metric | Case Type |
|---------|---------|------------|--------|-----------|
| OpenAI Small | 50K | 1536 | Cosine | Performance1536D50K |
| OpenAI Medium | 500K | 1536 | Cosine | Performance1536D500K |
| OpenAI Large | 5M | 1536 | Cosine | Performance1536D5M |
| Cohere Small | 1M | 768 | Cosine | Performance768D1M |
| Cohere Large | 10M | 768 | Cosine | Performance768D10M |

Datasets are automatically downloaded on first use to `/tmp/vectordb_bench/dataset/`.

## Example: Full Benchmark Run

Here's what a complete benchmark looks like:

```bash
$ vectordbbench rustvectordb --case-type Performance1536D50K --skip-search-concurrent

# Output:
2026-02-05 20:00:00 | INFO: Task submitted: Performance1536D50K
2026-02-05 20:00:01 | INFO: Initializing Rust VectorDB (dim=1536)
2026-02-05 20:00:02 | INFO: Downloading dataset... [====================] 100%
2026-02-05 20:00:15 | INFO: Start inserting embeddings (batch=100)
2026-02-05 20:02:30 | INFO: Inserted 50000 vectors (dur=135s)
2026-02-05 20:02:31 | INFO: Building index...
2026-02-05 20:03:51 | INFO: Index built (dur=80s)
2026-02-05 20:03:52 | INFO: Subprocess: rebuilding index from staging file
2026-02-05 20:04:36 | INFO: Subprocess: index rebuilt (44.2s)
2026-02-05 20:04:37 | INFO: Running serial search (1000 queries)
2026-02-05 20:04:42 | INFO: Search complete (dur=5.3s)

Results:
| DB           | load_dur | recall | p99_latency | p95_latency |
| RustVectorDB | 333.15s  | 0.9561 | 12.6ms      | 6.0ms       |
```

## Further Reading

- **VectorDBBench Documentation**: https://github.com/zilliztech/VectorDBBench
- **PyO3 Guide**: https://pyo3.rs/
- **Rust VectorDB Architecture**: See `ARCHITECTURE.md` (if exists)
- **Quick Testing Guide**: See `VECTORDBBENCH_QUICK_TEST.md`

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Look at example client implementations in VectorDBBench:
   - `vectordb_bench/backend/clients/chroma/`
   - `vectordb_bench/backend/clients/milvus/`
3. Review logs in `/tmp/vectordb_bench_run.log`
4. Open an issue with:
   - Command used
   - Error message
   - Dataset and parameters
   - System info (OS, Python version)

## Summary

With this integration, you can:
- ✅ Run standardized benchmarks against real datasets
- ✅ Compare performance with industry-standard databases
- ✅ Iterate quickly with skip-load workflow
- ✅ Tune parameters for your use case
- ✅ Validate improvements with reliable metrics

The integration uses PyO3 for maximum performance and handles all the complexity of VectorDBBench's multiprocessing architecture transparently.

Happy benchmarking! 🚀
