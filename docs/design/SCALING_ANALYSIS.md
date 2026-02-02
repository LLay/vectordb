# Vector Database Scaling Analysis

Based on benchmark results from 1M vector dataset (1024 dimensions, clustered data).

## Memory Requirements Per Vector

| Component | Size | Notes |
|-----------|------|-------|
| Binary quantized vector | 128 bytes | 1024 bits / 8 |
| Tree metadata (amortized) | ~2 bytes | Node structure, indices |
| **Total RAM per vector** | **~130 bytes** | |
| Full-precision storage | 4,096 bytes | 1024 × f32, stored in S3 |

## 100 Million Vectors

**RAM Requirements:**
- Index in RAM: ~20-25 GB
- Binary vectors: 12.8 GB (100M × 128 bytes)
- Tree nodes (~1.4M): 5.7 GB (centroids)
- Metadata: ~2 GB

**Storage (S3):**
- Full-precision vectors: 390 GB

**Recommended AWS Instance:** `r7iz.8xlarge`
- RAM: 256 GB (10x headroom)
- vCPUs: 32
- Cost: ~$7/hour (~$5K/month)
- Expected latency: ~500 μs average, ~600 μs p99
- Build time: ~60 minutes

## AWS Maximum Single-Machine Capacity

**Largest Instance: `u-24tb1.112xlarge`**
- RAM: 24 TB (24,576 GB)
- vCPUs: 448
- Network: 100 Gbps
- Cost: ~$218/hour (~$160K/month)

**Maximum Vectors:**
- Theoretical: ~193 billion vectors
- Conservative (with overhead): ~150 billion vectors

**Storage Requirements (S3):**
- 150B vectors: 600 TB in S3
- S3 cost: ~$14K/month

## Scaling by Dataset Size

### 1M Vectors (Benchmark Baseline)
- RAM: ~200 MB
- Storage: 3.9 GB
- Instance: t3.medium
- Cost: ~$0.04/hour
- Latency: 475 μs avg, 587 μs p99
- Build time: ~1 minute

### 10M Vectors
- RAM: ~2 GB
- Storage: 39 GB
- Instance: r7iz.large (16 GB RAM)
- Cost: ~$0.45/hour
- Latency: ~500 μs (estimated)
- Build time: ~10 minutes

### 100M Vectors
- RAM: ~20 GB
- Storage: 390 GB
- Instance: r7iz.8xlarge (256 GB RAM)
- Cost: ~$7/hour
- Latency: ~500-600 μs
- Build time: ~60 minutes

### 1B Vectors
- RAM: ~200 GB
- Storage: 3.9 TB
- Instance: r7iz.32xlarge (1 TB RAM)
- Cost: ~$28/hour
- Latency: ~1-2 ms (estimated)
- Build time: ~10 hours

### 10B Vectors
- RAM: ~2 TB
- Storage: 39 TB
- Instance: u-6tb1.112xlarge (6 TB RAM)
- Cost: ~$55/hour
- Latency: ~2-5 ms (estimated)
- Build time: ~100 hours (4+ days)

### 150B Vectors (Theoretical Max)
- RAM: ~20 TB
- Storage: 600 TB
- Instance: u-24tb1.112xlarge (24 TB RAM)
- Cost: ~$218/hour
- Latency: ~10-20 ms (estimated, limited by tree depth)
- Build time: ~2,000 hours (83 days)

## Critical Scaling Challenges

### Tree Depth Problem
- At 1M vectors: depth = 10
- At 100M vectors: depth ≈ 13-15 (estimated)
- With probes=5, depth=15 → 5^15 = 30 billion leaves explored
- This causes exponential search cost

### Build Time
- Linear scaling: 1M = 1 min → 100M = 100 min → 1B = 16 hours
- Becomes prohibitive at 10B+ (days to weeks)

### S3 Access Latency
- Reranking requires reading full-precision vectors from S3
- Need fast S3 access (S3 Express One Zone or mountpoint-s3)
- Alternative: Keep hot vectors in local NVMe

## Optimization Strategies for >1B Vectors

### 1. Sharding
- Distribute vectors across multiple machines
- Each shard handles subset of vector space
- Aggregate results (requires scatter-gather query pattern)

### 2. IVF Index (Instead of Hierarchical)
- Flat, single-level clustering
- No exponential probe explosion
- Better for uniform/random distributions

### 3. Product Quantization
- Compress vectors further (8x-32x compression)
- 128 bytes → 16-32 bytes per vector
- Enables 10x more vectors in same RAM

### 4. GPU Acceleration
- Distance calculations on GPU
- 10-100x speedup for batch operations
- Especially beneficial for reranking phase

### 5. Hybrid Architecture
- Hot tier: Recent/popular vectors in RAM
- Warm tier: Less frequent on NVMe
- Cold tier: Archival in S3

## Cost Comparison

| Scale | Instance | RAM | Compute Cost/Month | S3 Cost/Month | Total/Month |
|-------|----------|-----|-------------------|---------------|-------------|
| 100M | r7iz.8xlarge | 256 GB | $5,000 | $9 | $5,009 |
| 1B | r7iz.32xlarge | 1 TB | $20,000 | $90 | $20,090 |
| 10B | u-6tb1.112xlarge | 6 TB | $40,000 | $900 | $40,900 |
| 150B | u-24tb1.112xlarge | 24 TB | $160,000 | $14,000 | $174,000 |

*S3 costs assume Standard tier ($0.023/GB/month)*

## Recommendations

**For 100M vectors:**
- Single machine is practical and cost-effective
- Use `r7iz.8xlarge` for production
- Sub-millisecond latency achievable
- Build time reasonable (~1 hour)

**For 1B vectors:**
- Single machine still viable
- Consider sharding for better latency
- Build time becomes significant (~10 hours)

**For 10B+ vectors:**
- Single machine possible but not recommended
- Shard across multiple machines
- Consider IVF or product quantization
- Implement incremental building

**For 100B+ vectors:**
- Single machine not practical
- Mandatory distributed architecture
- Consider specialized vector DB (Pinecone, Weaviate, Qdrant)
- Or build custom sharded solution

## Architecture: Index in RAM, Vectors in S3

This hybrid approach enables massive scale on single machines:

**Pros:**
- Index fits in RAM for fast tree traversal
- Binary quantization enables fast filtering
- Only fetch full vectors for top-k reranking
- Minimal S3 reads (k × rerank_factor vectors)

**Cons:**
- S3 latency adds overhead to queries
- Requires fast S3 access (Express or mountpoint)
- Network bandwidth becomes bottleneck

**Optimization:**
- Use S3 Express One Zone for single-digit ms latency
- Or use Mountpoint for S3 with kernel-level caching
- Or hybrid: hot vectors in NVMe, cold in S3
