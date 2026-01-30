# Vector Database Scaling Goals

Three deployment scenarios with detailed specifications, costs, and performance targets.

---

## Goal 1: Development Setup (10M Vectors on Laptop)

**Target:** Run locally on developer laptop for testing and development.

### Hardware Specs
- **Device:** MacBook Pro M2/M3 (typical developer laptop)
- **RAM:** 16-32 GB
- **Storage:** 500 GB SSD (local disk)
- **Cost:** $0/hour (already owned)

### Capacity
- **Vectors:** 10 million
- **Dimensions:** 1024
- **RAM usage:** ~2 GB (index only)
  - Binary vectors: 1.28 GB (10M × 128 bytes)
  - Tree nodes: ~570 MB
  - Overhead: ~200 MB
- **Storage:** 39 GB (full-precision vectors)

### Performance Targets
- **Build time:** ~10 minutes
- **Search latency:** <1 ms average
- **Recall@10:** >95% (with 5 probes)
- **Throughput:** 1,000+ queries/sec (single-threaded)

---

## Goal 2: Budget Production (1.5B Vectors on Cheap AWS)

**Target:** Maximum vectors on a cost-effective AWS instance (<$10/hour).

### Hardware Specs
- **Instance:** AWS `r7iz.8xlarge`
- **RAM:** 256 GB
- **vCPUs:** 32
- **Network:** 12.5 Gbps
- **Cost:** ~$7/hour ($5,000/month)

### Capacity
- **Vectors:** 1.5 billion (maximum capacity)
- **Dimensions:** 1024
- **RAM usage:** ~200 GB (index only)
  - Binary vectors: 192 GB (1.5B × 128 bytes)
  - Tree nodes: ~7 GB
  - Overhead: ~1 GB
- **Storage (S3):** 6 TB (full-precision vectors)
  - Cost: ~$138/month

### Performance Targets
- **Build time:** ~15 hours (one-time or incremental)
- **Search latency:** 0.5-1 ms average, <2 ms p99
- **Recall@10:** >80% (with 5-7 probes)

### Cost Breakdown
| Component | Monthly Cost |
|-----------|-------------|
| Compute (r7iz.8xlarge) | $5,040 |
| S3 Storage (6 TB) | $138 |
| S3 Requests | ~$50 |
| Data Transfer | ~$100 |
| **Total** | **~$5,328/month** |

### Architecture
```
┌─────────────────────────────────────────┐
│  r7iz.8xlarge (256 GB RAM)              │
│  ┌──────────────────────────────────┐   │
│  │  Index in RAM (~200 GB)          │   │
│  │  • Binary vectors (192 GB)       │   │
│  │  • Tree structure (7 GB)         │   │
│  │  • Query cache (1 GB)            │   │
│  └──────────────────────────────────┘   │
│              ↓ mmap                      │
│  ┌──────────────────────────────────┐   │
│  │  S3 Mountpoint (kernel cache)    │   │
│  │  Full-precision vectors (6 TB)   │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
         ↓ network (12.5 Gbps)
┌─────────────────────────────────────────┐
│  S3 Express One Zone                    │
│  • Sub-ms access latency                │
│  • 6 TB stored                          │
└─────────────────────────────────────────┘
```
---

## Goal 3: Enterprise Scale (100B Vectors on Expensive AWS)

**Target:** Maximum single-machine capacity for enterprise workloads.

### Hardware Specs
- **Instance:** AWS `u-18tb1.112xlarge` (High Memory)
- **RAM:** 18 TB
- **vCPUs:** 448
- **Network:** 100 Gbps
- **Cost:** ~$165/hour (~$120,000/month)

### Capacity
- **Vectors:** 100 billion (conservative target)
- **Dimensions:** 1024
- **RAM usage:** ~14 TB (index only)
  - Binary vectors: 12.8 TB (100B × 128 bytes)
  - Tree nodes: ~1 TB
  - Buffer/cache: ~200 GB
- **Storage (S3):** 400 TB (full-precision vectors)
  - Cost: ~$9,200/month

### Performance Targets
- **Build time:** ~1,400 hours (58 days) OR sharded parallel build
- **Search latency:** 2-5 ms average, <10 ms p99
- **Recall@10:** >70% (with optimal probes)
- **Throughput:** 20,000+ queries/sec (fully parallel)
- **Availability:** 99.99% uptime (4-nines)

### Cost Breakdown
| Component | Monthly Cost |
|-----------|-------------|
| Compute (u-18tb1.112xlarge) | $118,800 |
| S3 Storage (400 TB) | $9,200 |
| S3 Express Zone | $12,000 |
| Data Transfer | $2,000 |
| Monitoring/Backup | $1,000 |
| **Total** | **~$143,000/month** |


### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  u-18tb1.112xlarge (18 TB RAM, 448 vCPU)                    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Shard Manager (coordinator)                       │     │
│  └────────────────────────────────────────────────────┘     │
│       ↓ parallel queries to 20 shards                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Shard 0  │  │ Shard 1  │  │   ...    │  │ Shard 19 │   │
│  │ 5B vecs  │  │ 5B vecs  │  │          │  │ 5B vecs  │   │
│  │ 700 GB   │  │ 700 GB   │  │          │  │ 700 GB   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       ↓              ↓             ↓              ↓         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  S3 Mountpoint (NVMe cache: 4 TB hot data)          │  │
│  │  Full-precision vectors (400 TB in S3)              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                    ↓ 100 Gbps network
┌─────────────────────────────────────────────────────────────┐
│  S3 Express One Zone (multiple buckets for sharding)        │
│  • Sub-ms latency                                           │
│  • 400 TB total storage                                     │
│  • Parallel access across shards                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparison Table

| Goal | Vectors | RAM | Cost/Month | Latency (p99) | Recall@10 | Build Time |
|------|---------|-----|------------|---------------|-----------|------------|
| **Development** | 10M | 16 GB | $0 | <2 ms | >95% | 10 min |
| **Budget Prod** | 1.5B | 256 GB | $5,300 | <2 ms | >80% | 15 hrs |
| **Enterprise** | 100B | 18 TB | $143,000 | <10 ms | >70% | 7 days* |

*With parallel sharded building

---


