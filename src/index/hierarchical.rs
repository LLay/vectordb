//! Hierarchical clustered index with binary quantization
//! 
//! Uses multi-level clustering (tree structure) for fast search:
//! - Level 0 (root): ~10 clusters
//! - Level 1: ~10 sub-clusters per parent = ~100 total
//! - Leaf level: Vectors grouped by final cluster
//! 
//! Binary quantization at each level for fast filtering + precise reranking.

use crate::clustering::KMeans;
use crate::distance::{distance, DistanceMetric};
use crate::quantization::{BinaryQuantizer, BinaryVector, hamming_distance};
use crate::storage::MmapVectorStore;
use std::path::Path;

/// A node in the hierarchical tree
#[derive(Debug, Clone)]
struct ClusterNode {
    /// Node ID
    #[allow(dead_code)]
    id: usize,
    /// Binary quantized centroid
    binary_centroid: BinaryVector,
    /// Full precision centroid (for reranking)
    #[allow(dead_code)]
    full_centroid: Vec<f32>,
    /// Children node IDs (empty for leaf nodes)
    children: Vec<usize>,
    /// Vector indices (only for leaf nodes)
    vector_indices: Vec<usize>,
}

/// Hierarchical clustered index with binary quantization
/// 
/// Uses memory-mapped storage for full precision vectors to minimize RAM usage.
/// Hot vectors are automatically cached by the OS.
pub struct ClusteredIndex {
    /// All nodes in the tree (indexed by node ID)
    nodes: Vec<ClusterNode>,
    /// Root node IDs (level 0)
    root_ids: Vec<usize>,
    /// Binary quantizer
    quantizer: BinaryQuantizer,
    /// Pre-quantized binary vectors (for fast candidate scanning)
    binary_vectors: Vec<BinaryVector>,
    /// Full precision vectors (memory-mapped for RAM efficiency)
    full_vectors: MmapVectorStore,
    /// Distance metric
    metric: DistanceMetric,
    /// Dimensionality
    dimension: usize,
    /// Maximum tree depth (deepest leaf)
    max_depth: usize,
    /// Maximum leaf size (vectors per leaf)
    max_leaf_size: usize,
}

impl ClusteredIndex {
    /// Build a hierarchical clustered index with adaptive splitting
    /// 
    /// Vectors are stored in a memory-mapped file for RAM efficiency.
    /// The OS automatically caches hot vectors.
    /// 
    /// # Arguments
    /// * `vectors` - The vectors to index
    /// * `vector_file` - Path where vectors will be stored on disk
    /// * `branching_factor` - Number of clusters at each level (~10-20 typical)
    /// * `max_leaf_size` - Maximum vectors per leaf (splits if larger, ~100-200 typical)
    /// * `metric` - Distance metric to use
    /// * `max_iterations` - Maximum k-means iterations
    pub fn build<P: AsRef<Path>>(
        vectors: Vec<Vec<f32>>,
        vector_file: P,
        branching_factor: usize,
        max_leaf_size: usize,
        metric: DistanceMetric,
        max_iterations: usize,
    ) -> std::io::Result<Self> {
        assert!(!vectors.is_empty(), "Cannot build index from empty vectors");
        assert!(branching_factor >= 2, "Branching factor must be >= 2");
        assert!(max_leaf_size >= 10, "Max leaf size must be >= 10");
        
        let dimension = vectors[0].len();
        let num_vectors = vectors.len();

        println!("Building adaptive hierarchical index...");
        println!("  Vectors: {}", num_vectors);
        println!("  Branching factor: {}", branching_factor);
        println!("  Max leaf size: {}", max_leaf_size);

        // Create binary quantizer
        let quantizer = BinaryQuantizer::from_vectors(&vectors);

        // Pre-quantize all vectors for fast candidate scanning
        println!("  Quantizing {} vectors...", num_vectors);
        let binary_vectors = quantizer.quantize_batch_parallel(&vectors);
        println!("  Quantization complete: {} bytes per vector", 
                 binary_vectors[0].bits.len() * 8);

        // Write vectors to disk and create memory-mapped store
        println!("  Writing {} vectors to disk...", vectors.len());
        let full_vectors = MmapVectorStore::create(&vector_file, &vectors)?;
        let size_mb = full_vectors.size_bytes() as f64 / 1_048_576.0;
        println!("  Vector file: {:.2} MB", size_mb);

        // Build tree recursively with adaptive splitting
        let mut nodes = Vec::new();
        let mut next_node_id = 0;
        let mut max_depth_reached = 0;
        
        let root_ids = Self::build_recursive(
            &vectors,
            &quantizer,
            (0..num_vectors).collect(),
            0,
            &mut max_depth_reached,
            max_leaf_size,
            branching_factor,
            metric,
            max_iterations,
            &mut nodes,
            &mut next_node_id,
        );

        println!("  Max depth: {}", max_depth_reached);
        println!("  Total nodes: {}", nodes.len());
        println!("  Root nodes: {}", root_ids.len());

        // Calculate leaf statistics
        let leaf_sizes: Vec<usize> = nodes
            .iter()
            .filter(|n| n.children.is_empty())
            .map(|n| n.vector_indices.len())
            .collect();
        
        if !leaf_sizes.is_empty() {
            let avg_leaf = leaf_sizes.iter().sum::<usize>() as f64 / leaf_sizes.len() as f64;
            let max_leaf = *leaf_sizes.iter().max().unwrap();
            let min_leaf = *leaf_sizes.iter().min().unwrap();
            println!("  Leaf sizes: min={}, max={}, avg={:.1}", min_leaf, max_leaf, avg_leaf);
        }

        Ok(Self {
            nodes,
            root_ids,
            quantizer,
            binary_vectors,
            full_vectors,
            metric,
            dimension,
            max_depth: max_depth_reached,
            max_leaf_size,
        })
    }
    

    /// Recursively build tree levels with adaptive splitting
    /// 
    /// Algorithm:
    /// 1. Base case: If few enough vectors (≤max_leaf_size), create leaf node
    /// 2. Recursive case: Cluster vectors into k groups, recurse on each group
    /// 
    /// This creates a tree where:
    /// - Internal nodes guide search via cluster centroids
    /// - Leaf nodes store actual vector indices
    /// - Depth adapts to data distribution (denser regions → deeper tree)
    #[allow(clippy::too_many_arguments)]
    fn build_recursive(
        all_vectors: &[Vec<f32>],
        quantizer: &BinaryQuantizer,
        indices: Vec<usize>,
        current_depth: usize,
        max_depth_reached: &mut usize,
        max_leaf_size: usize,
        branching_factor: usize,
        metric: DistanceMetric,
        max_iterations: usize,
        nodes: &mut Vec<ClusterNode>,
        next_node_id: &mut usize,
    ) -> Vec<usize> {
        // Track maximum depth reached
        *max_depth_reached = (*max_depth_reached).max(current_depth);
        
        if indices.is_empty() {
            return Vec::new();
        }

        // Base case: Create leaf node if conditions are met
        if Self::should_be_leaf(indices.len(), current_depth, max_leaf_size, branching_factor) {
            let node_id = *next_node_id;
            *next_node_id += 1;

            let leaf = Self::create_leaf_from_indices(indices, all_vectors, quantizer, node_id);
            nodes.push(leaf);
            return vec![node_id];
        }

        // Recursive case: Cluster and split
        let num_clusters = branching_factor.min(indices.len());
        let (kmeans, assignment) = Self::cluster_subset(
            &indices,
            all_vectors,
            num_clusters,
            metric,
            max_iterations,
        );

        // Quantize centroids for fast search
        let binary_centroids = quantizer.quantize_batch(&kmeans.centroids);

        // Build child nodes for each cluster
        let mut node_ids = Vec::new();
        
        for cluster_id in 0..num_clusters {
            let cluster_indices = Self::extract_cluster_indices(&indices, &assignment, cluster_id);
            
            if cluster_indices.is_empty() {
                continue; // Skip empty clusters
            }

            // Recursively build subtree for this cluster
            let children = Self::build_recursive(
                all_vectors,
                quantizer,
                cluster_indices,
                current_depth + 1,
                max_depth_reached,
                max_leaf_size,
                branching_factor,
                metric,
                max_iterations,
                nodes,
                next_node_id,
            );

            // Create internal node for this cluster
            let node_id = *next_node_id;
            *next_node_id += 1;

            let node = Self::create_internal_from_cluster(
                node_id,
                cluster_id,
                &binary_centroids,
                &kmeans.centroids,
                children,
            );

            nodes.push(node);
            node_ids.push(node_id);
        }

        node_ids
    }

    /// Search with adaptive hierarchical clustering
    /// 
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `probes_per_level` - Number of clusters to explore at each level
    /// * `rerank_factor` - How many binary candidates to rerank
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        probes_per_level: usize,
        rerank_factor: usize,
    ) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dimension, "Query dimension mismatch");

        // Quantize query
        let query_binary = self.quantizer.quantize(query);

        // Start from root and traverse down
        let mut current_nodes = self.root_ids.clone();
        
        // Traverse tree until we reach leaves (adaptive depth)
        loop {
            if current_nodes.is_empty() {
                break;
            }

            // Find nearest nodes at this level (parallel if many nodes)
            use rayon::prelude::*;
            let mut node_distances: Vec<(usize, u32)> = if current_nodes.len() > 10 {
                // Parallel distance computation for many nodes
                current_nodes
                    .par_iter()
                    .map(|&node_id| {
                        let node = &self.nodes[node_id];
                        let dist = hamming_distance(&query_binary, &node.binary_centroid);
                        (node_id, dist)
                    })
                    .collect()
            } else {
                // Sequential for few nodes (less overhead)
                current_nodes
                    .iter()
                    .map(|&node_id| {
                        let node = &self.nodes[node_id];
                        let dist = hamming_distance(&query_binary, &node.binary_centroid);
                        (node_id, dist)
                    })
                    .collect()
            };

            node_distances.sort_by_key(|x| x.1);

            // Take top probes_per_level nodes
            let top_nodes: Vec<usize> = node_distances
                .iter()
                .take(probes_per_level)
                .map(|(id, _)| *id)
                .collect();

            // Check if these are leaf nodes
            let first_node = &self.nodes[top_nodes[0]];
            if first_node.children.is_empty() {
                // Reached leaf level - collect candidates
                return self.search_leaves(&top_nodes, query, &query_binary, k, rerank_factor);
            }

            // Not leaves yet - expand to children for next level
            current_nodes = top_nodes
                .iter()
                .flat_map(|&node_id| self.nodes[node_id].children.clone())
                .collect();
        }

        Vec::new()
    }

    /// Search within leaf nodes - two-phase algorithm
    /// 
    /// Phase 1: Fast filtering with binary quantization (Hamming distance)
    /// Phase 2: Precise reranking with full vectors (Euclidean/Cosine distance)
    /// 
    /// This two-phase approach is ~10-20x faster than brute force full-precision search
    fn search_leaves(
        &self,
        leaf_ids: &[usize],
        query: &[f32],
        query_binary: &BinaryVector,
        k: usize,
        rerank_factor: usize,
    ) -> Vec<(usize, f32)> {
        let rerank_k = (k * rerank_factor).min(10000);

        // Phase 1: Fast binary filtering (Hamming distance on 32x compressed vectors)
        let binary_candidates = self.collect_leaf_candidates(leaf_ids, query_binary);
        
        if binary_candidates.is_empty() {
            return Vec::new();
        }

        // Select top candidates for reranking (e.g., top 3*k if rerank_factor=3)
        let top_candidates = self.select_top_candidates(binary_candidates, rerank_k);

        // Phase 2: Precise reranking (full precision distance on filtered candidates)
        let reranked = self.rerank_with_full_precision(&top_candidates, query);

        // Return final top-k results (sorted by distance)
        self.finalize_results(reranked, k)
    }

    /// Open an existing index from disk
    /// 
    /// # Arguments
    /// * `vector_file` - Path to the memory-mapped vector file
    /// * `dimension` - Vector dimensionality
    /// * `count` - Number of vectors
    /// 
    /// Note: This is a placeholder. Full serialization of the index structure
    /// (nodes, quantizer, etc.) is not yet implemented.
    pub fn open<P: AsRef<Path>>(
        vector_file: P,
        dimension: usize,
        count: usize,
    ) -> std::io::Result<MmapVectorStore> {
        MmapVectorStore::open(vector_file, dimension, count)
    }
    
    /// Get memory usage estimate in bytes (RAM only, excludes mmap file)
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = 0;
        
        // Nodes (rough estimate)
        total += std::mem::size_of::<ClusterNode>() * self.nodes.len();
        
        // Binary vectors (stored in RAM)
        total += self.binary_vectors.iter()
            .map(|bv| bv.bits.len() * 8)
            .sum::<usize>();
        
        // Note: full_vectors are mmap'd, so not counted in RAM usage
        // (OS will cache hot pages automatically)
        
        total
    }
    
    /// Get disk usage in bytes (mmap file size)
    pub fn disk_usage_bytes(&self) -> usize {
        self.full_vectors.size_bytes()
    }
    
    /// Check if a node should be a leaf (base case for recursion)
    #[inline]
    fn should_be_leaf(
        cluster_size: usize,
        current_depth: usize,
        max_leaf_size: usize,
        branching_factor: usize,
    ) -> bool {
        const MAX_TREE_DEPTH: usize = 15; // Safety limit to prevent infinite recursion
        
        cluster_size <= max_leaf_size 
            || current_depth >= MAX_TREE_DEPTH
            || cluster_size < branching_factor
    }

    /// Create a leaf node from a set of vector indices
    #[inline]
    fn create_leaf_from_indices(
        indices: Vec<usize>,
        all_vectors: &[Vec<f32>],
        quantizer: &BinaryQuantizer,
        node_id: usize,
    ) -> ClusterNode {
        // Compute centroid from the vectors in this leaf
        let subset_vectors: Vec<Vec<f32>> = indices
            .iter()
            .map(|&idx| all_vectors[idx].clone())
            .collect();
        
        let centroid = compute_centroid(&subset_vectors);
        let binary_centroid = quantizer.quantize(&centroid);
        
        ClusterNode {
            id: node_id,
            binary_centroid,
            full_centroid: centroid,
            children: Vec::new(),
            vector_indices: indices,
        }
    }

    /// Cluster vectors and return k-means result
    #[inline]
    fn cluster_subset(
        indices: &[usize],
        all_vectors: &[Vec<f32>],
        num_clusters: usize,
        metric: DistanceMetric,
        max_iterations: usize,
    ) -> (KMeans, crate::clustering::ClusterAssignment) {
        let subset_vectors: Vec<Vec<f32>> = indices
            .iter()
            .map(|&idx| all_vectors[idx].clone())
            .collect();

        KMeans::fit(&subset_vectors, num_clusters, metric, max_iterations)
    }

    /// Extract indices belonging to a specific cluster
    #[inline]
    fn extract_cluster_indices(
        indices: &[usize],
        assignment: &crate::clustering::ClusterAssignment,
        cluster_id: usize,
    ) -> Vec<usize> {
        indices
            .iter()
            .enumerate()
            .filter(|(i, _)| assignment.assignments[*i] == cluster_id)
            .map(|(_, &idx)| idx)
            .collect()
    }

    /// Create an internal node from cluster information
    #[inline]
    fn create_internal_from_cluster(
        node_id: usize,
        cluster_id: usize,
        binary_centroids: &[BinaryVector],
        full_centroids: &[Vec<f32>],
        children: Vec<usize>,
    ) -> ClusterNode {
        ClusterNode {
            id: node_id,
            binary_centroid: binary_centroids[cluster_id].clone(),
            full_centroid: full_centroids[cluster_id].clone(),
            children,
            vector_indices: Vec::new(), // Internal nodes don't store vectors
        }
    }

    /// Compute Hamming distances for a set of vector indices (adaptive parallelization)
    /// 
    /// Returns Vec<(vector_idx, hamming_distance)> pairs.
    /// 
    /// Adaptive strategy:
    /// - >100 vectors: Parallel (overhead is worth it)
    /// - ≤100 vectors: Sequential (avoid rayon overhead ~10-50μs)
    #[inline]
    fn compute_hamming_distances(
        &self,
        vector_indices: &[usize],
        query_binary: &BinaryVector,
    ) -> Vec<(usize, u32)> {
        use rayon::prelude::*;
        
        if vector_indices.len() > 100 {
            // Parallel: Process vectors across multiple cores
            // Each thread computes Hamming distance independently
            // Rayon's work-stealing balances the load automatically
            vector_indices
                .par_iter()
                .map(|&idx| (idx, hamming_distance(query_binary, &self.binary_vectors[idx])))
                .collect()
        } else {
            // Sequential: Faster for small workloads
            // Avoids parallel overhead (thread spawning, synchronization)
            vector_indices
                .iter()
                .map(|&idx| (idx, hamming_distance(query_binary, &self.binary_vectors[idx])))
                .collect()
        }
    }

    /// Collect candidates from multiple leaves (adaptive parallelization)
    #[inline]
    fn collect_leaf_candidates(
        &self,
        leaf_ids: &[usize],
        query_binary: &BinaryVector,
    ) -> Vec<(usize, u32)> {
        use rayon::prelude::*;
        
        if leaf_ids.len() > 1 {
            // Multiple leaves - parallelize across leaves
            leaf_ids
                .par_iter()
                .flat_map(|&leaf_id| {
                    let leaf = &self.nodes[leaf_id];
                    self.compute_hamming_distances(&leaf.vector_indices, query_binary)
                })
                .collect()
        } else if let Some(&leaf_id) = leaf_ids.first() {
            // Single leaf - adaptive parallelization within
            let leaf = &self.nodes[leaf_id];
            self.compute_hamming_distances(&leaf.vector_indices, query_binary)
        } else {
            Vec::new()
        }
    }

    /// Select top-k candidates by distance (partial sort for efficiency)
    #[inline]
    fn select_top_candidates(&self, mut candidates: Vec<(usize, u32)>, k: usize) -> Vec<(usize, u32)> {
        if candidates.len() <= k {
            candidates.sort_by_key(|x| x.1);
            candidates
        } else {
            // Partial sort: only sort enough to get top-k (faster than full sort)
            candidates.select_nth_unstable_by(k - 1, |a, b| a.1.cmp(&b.1));
            candidates.truncate(k);
            candidates
        }
    }

    /// Rerank candidates with full precision distance (parallel)
    #[inline]
    fn rerank_with_full_precision(
        &self,
        candidates: &[(usize, u32)],
        query: &[f32],
    ) -> Vec<(usize, f32)> {
        use rayon::prelude::*;
        
        candidates
            .par_iter()
            .map(|(idx, _)| {
                let full_vec = self.full_vectors.get(*idx);
                let dist = distance(query, full_vec, self.metric);
                (*idx, dist)
            })
            .collect()
    }

    /// Sort and return top-k results
    #[inline]
    fn finalize_results(&self, mut results: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
        if results.len() <= k {
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            results
        } else {
            // Partial sort for top-k
            results.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut top_k = results[..k].to_vec();
            top_k.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            top_k
        }
    }

    /// Get total number of vectors
    pub fn len(&self) -> usize {
        self.full_vectors.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.full_vectors.is_empty()
    }

    /// Get dimensionality
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get maximum depth of tree
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Get number of nodes in tree
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get maximum leaf size
    pub fn max_leaf_size(&self) -> usize {
        self.max_leaf_size
    }
}

/// Compute centroid of a set of vectors
fn compute_centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let dim = vectors[0].len();
    let mut centroid = vec![0.0; dim];
    
    for vector in vectors {
        for (i, &val) in vector.iter().enumerate() {
            centroid[i] += val;
        }
    }
    
    let n = vectors.len() as f32;
    for val in &mut centroid {
        *val /= n;
    }
    
    centroid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_build() {
        use std::fs;
        
        let mut vectors = Vec::new();
        
        // Create 100 vectors
        for i in 0..100 {
            vectors.push(vec![i as f32, 0.0, 0.0]);
        }

        let test_file = "test_hier_build.bin";
        let index = ClusteredIndex::build(vectors, test_file, 5, 50, DistanceMetric::L2, 10).unwrap();

        assert_eq!(index.len(), 100);
        assert!(index.max_depth() >= 1);
        assert!(index.num_nodes() > 0);
        
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_hierarchical_search() {
        use std::fs;
        
        let mut vectors = Vec::new();
        
        // Cluster 1: around [0, 0]
        for i in 0..50 {
            vectors.push(vec![i as f32 * 0.01, 0.0]);
        }
        // Cluster 2: around [10, 10]
        for i in 0..50 {
            vectors.push(vec![10.0 + i as f32 * 0.01, 10.0]);
        }

        let test_file = "test_hier_search.bin";
        let index = ClusteredIndex::build(vectors, test_file, 5, 30, DistanceMetric::L2, 10).unwrap();

        // Query near first cluster
        let query = vec![0.25, 0.0];
        let results = index.search(&query, 5, 2, 3);

        assert_eq!(results.len(), 5);
        
        // Results should be from first cluster
        for (idx, _) in &results {
            assert!(*idx < 50, "Expected index from first cluster, got {}", idx);
        }
        
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_adaptive_splitting() {
        // Create non-uniform distribution
        let mut vectors = Vec::new();
        
        // Dense cluster: 500 vectors tightly packed
        for i in 0..500 {
            vectors.push(vec![i as f32 * 0.001, 0.0]);
        }
        
        // Sparse cluster: 50 vectors spread out
        for i in 0..50 {
            vectors.push(vec![100.0 + i as f32 * 1.0, 0.0]);
        }

        let max_leaf_size = 100;
        let test_file = "test_adaptive_old.bin";
        let index = ClusteredIndex::build(
            vectors,
            test_file,
            5, 
            max_leaf_size, 
            DistanceMetric::L2, 
            10
        ).unwrap();
        
        // Check that leaves respect max_leaf_size
        for node in &index.nodes {
            if node.children.is_empty() { // It's a leaf
                assert!(
                    node.vector_indices.len() <= max_leaf_size,
                    "Leaf has {} vectors, max is {}",
                    node.vector_indices.len(),
                    max_leaf_size
                );
            }
        }
        
        std::fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_different_branching_factors() {
        use std::fs;
        
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| vec![i as f32, 0.0])
            .collect();

        for branching in [3, 5, 10] {
            let test_file = format!("test_branch_{}_vectors.bin", branching);
            let index = ClusteredIndex::build(vectors.clone(), &test_file, branching, 50, DistanceMetric::L2, 10).unwrap();
            
            assert_eq!(index.len(), 200);
            
            let query = vec![100.0, 0.0];
            let results = index.search(&query, 10, 2, 3);
            assert_eq!(results.len(), 10);
            
            fs::remove_file(&test_file).ok();
        }
    }
    
    #[test]
    fn test_mmap_storage() {
        use std::fs;
        
        // Create vectors with distinct values
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| {
                (0..128).map(|j| (i * 128 + j) as f32).collect()
            })
            .collect();
        let test_file = "test_mmap_vectors.bin";
        
        // Build index (now uses mmap by default)
        let index = ClusteredIndex::build(
            vectors.clone(),
            test_file,
            10,
            100,
            DistanceMetric::L2,
            20,
        ).unwrap();
        
        // Check memory and disk usage
        let mem_usage = index.memory_usage_bytes();
        let disk_usage = index.disk_usage_bytes();
        
        println!("RAM usage: {:.2} MB", mem_usage as f64 / 1_048_576.0);
        println!("Disk usage: {:.2} MB", disk_usage as f64 / 1_048_576.0);
        
        // Disk usage should be much larger (full vectors)
        assert!(disk_usage > mem_usage);
        
        // Query should work - just verify we get results
        let query = vectors[500].clone();
        let results = index.search(&query, 10, 4, 5);
        assert_eq!(results.len(), 10);
        
        // Verify results are sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i-1].1, "Results should be sorted by distance");
        }
        
        // Verify we can access the vectors via mmap
        assert!(results[0].1 >= 0.0, "Distance should be non-negative");
        
        // Cleanup
        fs::remove_file(test_file).ok();
    }
}
