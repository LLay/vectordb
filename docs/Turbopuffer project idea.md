## **Turbopuffer Core Architecture Summary**

Turbopuffer achieves 200ms p99 over 100B vectors through:

1. **Hierarchical clustering (SPFresh)** \- Multi-level tree with \~100x branching factor to narrow search space  
2. **Binary quantization (RaBitQ)** \- 16-32x compression (1 bit/dimension) with error bounds for reranking  
3. **Memory hierarchy alignment** \- Quantized centroids in L3 cache, quantized data vectors in DRAM, full-precision vectors on NVMe  
4. **Distributed sharding** \- Random sharding across storage-dense VMs

---

## **Ideas for Improvements**

### **1\. Quantization Improvements**

**Product Quantization (PQ) or Additive Quantization hybrid**

* RaBitQ is 1-bit per dimension. You could explore 2-bit or 4-bit quantization (e.g., using PQ or ScaNN-style anisotropic quantization) for better recall at moderate compression  
* Implement a tiered approach: 1-bit for initial filtering, 4-bit for reranking candidates before fetching full precision  
* This could reduce the "1% rerank" phase they mention even further

**Learned quantization**

* Train a small neural network to learn optimal quantization boundaries per-cluster rather than using fixed binary thresholds  
* Could significantly improve recall without changing compression ratio

### **2\. Hardware Explorations**

**CXL Memory Pooling**

* CXL (Compute Express Link) is becoming available and allows memory pooling across nodes  
* You could explore using CXL-attached memory as a "tier 2.5" between DRAM and NVMe with \~100GB/s bandwidth but larger capacity  
* This is bleeding edge but a great learning opportunity

**GPU/TPU Acceleration for Distance Computations**

* They mention being compute-bound on binary quantized distance calculations  
* GPUs excel at this \- even a modest GPU could potentially handle the VPOPCNTDQ equivalent much faster  
* Explore hybrid CPU/GPU where GPU handles batch distance computations while CPU handles tree traversal

**FPGA for Custom Distance Kernels**

* FPGAs can implement custom bit manipulation logic that outperforms even AVX-512  
* AWS F1 instances or Intel Agilex could be interesting platforms  
* Implement a streaming architecture that processes quantized vectors as they flow from memory

**Persistent Memory (Intel Optane successors / CXL PM)**

* Explore using persistent memory as a faster-than-SSD tier  
* Could eliminate the NVMe latency for the reranking phase entirely

### **3\. Index Structure Improvements**

**Adaptive Branching Factor**

* They use fixed 100x branching. What if you made it adaptive based on cluster density?  
* Dense regions get higher branching (more children), sparse regions get lower  
* Could improve cache utilization in skewed distributions

**Graph-based hybrid (HNSW \+ Clustering)**

* Combine their SPFresh clustering with small HNSW graphs within each leaf cluster  
* The graph helps with local navigation when clusters aren't perfectly spherical  
* DiskANN does something similar

**Dynamic Tree Rebalancing**

* Implement online clustering updates that rebalance the tree without full rebuilds  
* Track cluster quality metrics and trigger partial rebuilds when recall degrades

### **4\. Systems-Level Improvements**

**Prefetching and Speculative Execution**

* At each tree level, speculatively prefetch the top-N most likely next-level clusters while computing current distances  
* Hide memory/disk latency behind computation  
* Requires predicting which clusters will be visited \- could use a small ML model

**Query Batching and Reordering**

* Batch incoming queries and reorder them to maximize cache hits  
* If two queries will visit the same clusters, process them together  
* Implement "query routing" that sends similar queries to the same shard

**Tiered Caching with LRU \+ Frequency**

* Implement ARC or 2Q caching policies instead of simple LRU  
* Some clusters are "hot" (frequent) vs "warm" (recent) \- treat them differently

**io\_uring for Async I/O**

* If they're not already using it, io\_uring can significantly reduce syscall overhead for the scatter-gather reads during reranking  
* Implement completion batching for the many small reads

### **5\. Algorithmic Improvements**

**Early Termination with Confidence Bounds**

* Use RaBitQ's error bounds more aggressively \- if you're confident the current top-k won't change, stop early  
* Implement progressive refinement that returns approximate results quickly, then refines

**Learned Index Routing**

* Train a small neural network to predict which clusters to probe given a query  
* Could be more accurate than nearest-centroid routing and reduce probes needed  
* Similar to learned indexes for B-trees

**Dimensionality Reduction Before Quantization**

* Apply PCA or random projection to reduce dimensions before binary quantization  
* Could improve the quality of 1-bit representations  
* Trade-off: extra computation vs better recall

### **6\. Low-Level Optimizations (Great for Learning)**

**Custom Memory Allocators**

* Implement a slab allocator optimized for fixed-size vector allocations  
* Reduce fragmentation and improve cache line utilization

**NUMA-Aware Data Placement**

* On multi-socket systems, ensure vectors are placed on the same NUMA node as the core processing them  
* Can provide 20-40% memory bandwidth improvement

**Huge Pages**

* Use 2MB or 1GB huge pages for the DRAM-resident quantized vectors  
* Reduces TLB misses significantly for large working sets

**Custom AVX-512 Kernels**

* Write hand-tuned assembly for the hot paths  
* Explore different instruction scheduling to maximize pipeline utilization  
* Compare Intel vs AMD performance (Zen 4 has excellent AVX-512 now)

## **Getting Started**

[vectordb-getting-started.md.pdf](https://drive.google.com/file/d/15gLma4NUlJgnxWMU9i_vTDWd6uyznOVH/view?usp=drive_link)

## **Resources**

https://modal.com/gpu-glossary/perf/arithmetic-intensity