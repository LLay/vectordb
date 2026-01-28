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

/// Storage backend for full precision vectors
enum VectorStorage {
    /// In-memory storage (fast, but uses more RAM)
    InMemory(Vec<Vec<f32>>),
    /// Memory-mapped storage (slower cold reads, but saves RAM)
    Mmap(MmapVectorStore),
}

impl VectorStorage {
    #[inline]
    fn get(&self, idx: usize) -> &[f32] {
        match self {
            VectorStorage::InMemory(vectors) => &vectors[idx],
            VectorStorage::Mmap(store) => store.get(idx),
        }
    }
    
    #[allow(dead_code)]
    fn len(&self) -> usize {
        match self {
            VectorStorage::InMemory(vectors) => vectors.len(),
            VectorStorage::Mmap(store) => store.len(),
        }
    }
    
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Hierarchical clustered index with binary quantization
pub struct ClusteredIndex {
    /// All nodes in the tree (indexed by node ID)
    nodes: Vec<ClusterNode>,
    /// Root node IDs (level 0)
    root_ids: Vec<usize>,
    /// Binary quantizer
    quantizer: BinaryQuantizer,
    /// Pre-quantized binary vectors (for fast candidate scanning)
    binary_vectors: Vec<BinaryVector>,
    /// Lookup table: original_index → full precision vector
    full_vectors: VectorStorage,
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
    /// # Arguments
    /// * `vectors` - The vectors to index
    /// * `branching_factor` - Number of clusters at each level (~10-20 typical)
    /// * `max_leaf_size` - Maximum vectors per leaf (splits if larger, ~100-200 typical)
    /// * `metric` - Distance metric to use
    /// * `max_iterations` - Maximum k-means iterations
    pub fn build(
        vectors: Vec<Vec<f32>>,
        branching_factor: usize,
        max_leaf_size: usize,
        metric: DistanceMetric,
        max_iterations: usize,
    ) -> Self {
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

        // Store full vectors for reranking (in-memory by default)
        let full_vectors = VectorStorage::InMemory(vectors.clone());

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

        Self {
            nodes,
            root_ids,
            quantizer,
            binary_vectors,
            full_vectors,
            metric,
            dimension,
            max_depth: max_depth_reached,
            max_leaf_size,
        }
    }
    
    /// Convert index to use memory-mapped storage
    /// 
    /// This saves RAM by storing full precision vectors on disk
    /// instead of in memory. The OS will cache hot vectors automatically.
    /// 
    /// # Arguments
    /// * `vector_file_path` - Where to store the vectors on disk
    pub fn use_mmap_storage<P: AsRef<Path>>(&mut self, vector_file_path: P) -> std::io::Result<()> {
        // Extract vectors from current storage
        let vectors = match &self.full_vectors {
            VectorStorage::InMemory(vecs) => vecs.clone(),
            VectorStorage::Mmap(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "Already using mmap storage",
                ));
            }
        };
        
        println!("Converting to memory-mapped storage...");
        println!("  Writing {} vectors to disk...", vectors.len());
        
        // Create memory-mapped store
        let mmap_store = MmapVectorStore::create(vector_file_path, &vectors)?;
        
        let size_mb = mmap_store.size_bytes() as f64 / 1_048_576.0;
        println!("  File size: {:.2} MB", size_mb);
        
        // Replace storage
        self.full_vectors = VectorStorage::Mmap(mmap_store);
        
        println!("  Conversion complete!");
        
        Ok(())
    }
    
    /// Check if index is using memory-mapped storage
    pub fn is_using_mmap(&self) -> bool {
        matches!(self.full_vectors, VectorStorage::Mmap(_))
    }
    
    /// Get memory usage estimate in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = 0;
        
        // Nodes (rough estimate)
        total += std::mem::size_of::<ClusterNode>() * self.nodes.len();
        
        // Binary vectors
        total += self.binary_vectors.iter()
            .map(|bv| bv.bits.len() * 8)
            .sum::<usize>();
        
        // Full vectors (only if in-memory)
        if let VectorStorage::InMemory(vectors) = &self.full_vectors {
            total += vectors.iter()
                .map(|v| v.len() * 4)
                .sum::<usize>();
        }
        
        total
    }
    
    /// Recursively build tree levels with adaptive splitting
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
        const MAX_TREE_DEPTH: usize = 15; // Safety limit to prevent infinite recursion
        
        // Update max depth tracker
        *max_depth_reached = (*max_depth_reached).max(current_depth);
        
        let cluster_size = indices.len();
        
        // Get vectors for this subset
        let subset_vectors: Vec<Vec<f32>> = indices
            .iter()
            .map(|&idx| all_vectors[idx].clone())
            .collect();

        if subset_vectors.is_empty() {
            return Vec::new();
        }

        // Decide if this should be a leaf node
        // Leaf if: small enough OR hit max depth OR too small to split
        let is_leaf = cluster_size <= max_leaf_size 
                   || current_depth >= MAX_TREE_DEPTH
                   || cluster_size < branching_factor;

        if is_leaf {
            // Create single leaf node with all vectors
            let node_id = *next_node_id;
            *next_node_id += 1;
            
            // Use cluster center as centroid
            let centroid = compute_centroid(&subset_vectors);
            let binary_centroid = quantizer.quantize(&centroid);
            
            let node = ClusterNode {
                id: node_id,
                binary_centroid,
                full_centroid: centroid,
                children: Vec::new(),
                vector_indices: indices,
            };
            
            nodes.push(node);
            return vec![node_id];
        }

        // Not a leaf - split into clusters
        let num_clusters = branching_factor.min(cluster_size);
        
        // Run k-means on this subset
        let (kmeans, assignment) = KMeans::fit(
            &subset_vectors,
            num_clusters,
            metric,
            max_iterations,
        );

        // Quantize centroids
        let binary_centroids = quantizer.quantize_batch(&kmeans.centroids);

        // Create nodes for this level
        let mut node_ids = Vec::new();
        
        for cluster_id in 0..num_clusters {
            let node_id = *next_node_id;
            *next_node_id += 1;

            // Get indices in this cluster
            let cluster_indices: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|(i, _)| assignment.assignments[*i] == cluster_id)
                .map(|(_, &idx)| idx)
                .collect();

            if cluster_indices.is_empty() {
                continue;
            }

            // Recursively build children (they decide if they're leaves)
            let children = Self::build_recursive(
                all_vectors,
                quantizer,
                cluster_indices.clone(),
                current_depth + 1,
                max_depth_reached,
                max_leaf_size,
                branching_factor,
                metric,
                max_iterations,
                nodes,
                next_node_id,
            );

            let node = ClusterNode {
                id: node_id,
                binary_centroid: binary_centroids[cluster_id].clone(),
                full_centroid: kmeans.centroids[cluster_id].clone(),
                children,
                vector_indices: Vec::new(), // Internal nodes don't store vectors
            };

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

            // Find nearest nodes at this level
            let mut node_distances: Vec<(usize, u32)> = current_nodes
                .iter()
                .map(|&node_id| {
                    let node = &self.nodes[node_id];
                    let dist = hamming_distance(&query_binary, &node.binary_centroid);
                    (node_id, dist)
                })
                .collect();

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

    /// Search within leaf nodes with parallel Hamming distance computation
    fn search_leaves(
        &self,
        leaf_ids: &[usize],
        query: &[f32],
        query_binary: &BinaryVector,
        k: usize,
        rerank_factor: usize,
    ) -> Vec<(usize, f32)> {
        use rayon::prelude::*;
        
        let rerank_k = (k * rerank_factor).min(10000);

        // Collect all vector indices from leaves
        let mut all_indices = Vec::new();
        for &leaf_id in leaf_ids {
            let leaf = &self.nodes[leaf_id];
            all_indices.extend_from_slice(&leaf.vector_indices);
        }

        // Parallel Hamming distance computation (using pre-quantized vectors)
        let mut binary_candidates: Vec<(usize, u32)> = all_indices
            .par_iter()
            .map(|&vector_idx| {
                let hamming_dist = hamming_distance(query_binary, &self.binary_vectors[vector_idx]);
                (vector_idx, hamming_dist)
            })
            .collect();

        if binary_candidates.is_empty() {
            return Vec::new();
        }

        // Select top rerank_k candidates
        if binary_candidates.len() <= rerank_k {
            binary_candidates.sort_by_key(|x| x.1);
        } else {
            binary_candidates.select_nth_unstable_by(rerank_k - 1, |a, b| a.1.cmp(&b.1));
            binary_candidates.truncate(rerank_k);
        }

        // Rerank with full precision (parallel)
        let mut reranked: Vec<(usize, f32)> = binary_candidates
            .par_iter()
            .map(|(original_idx, _)| {
                let full_vec = self.full_vectors.get(*original_idx);
                let dist = distance(query, full_vec, self.metric);
                (*original_idx, dist)
            })
            .collect();

        // Return final top-k
        if reranked.len() <= k {
            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return reranked;
        }

        reranked.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut results = reranked[..k].to_vec();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
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
        let mut vectors = Vec::new();
        
        // Create 100 vectors
        for i in 0..100 {
            vectors.push(vec![i as f32, 0.0, 0.0]);
        }

        let index = ClusteredIndex::build(vectors, 5, 50, DistanceMetric::L2, 10);

        assert_eq!(index.len(), 100);
        assert!(index.max_depth() >= 1);
        assert!(index.num_nodes() > 0);
    }

    #[test]
    fn test_hierarchical_search() {
        let mut vectors = Vec::new();
        
        // Cluster 1: around [0, 0]
        for i in 0..50 {
            vectors.push(vec![i as f32 * 0.01, 0.0]);
        }
        // Cluster 2: around [10, 10]
        for i in 0..50 {
            vectors.push(vec![10.0 + i as f32 * 0.01, 10.0]);
        }

        let index = ClusteredIndex::build(vectors, 5, 30, DistanceMetric::L2, 10);

        // Query near first cluster
        let query = vec![0.25, 0.0];
        let results = index.search(&query, 5, 2, 3);

        assert_eq!(results.len(), 5);
        
        // Results should be from first cluster
        for (idx, _) in &results {
            assert!(*idx < 50, "Expected index from first cluster, got {}", idx);
        }
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
        let index = ClusteredIndex::build(
            vectors, 
            5, 
            max_leaf_size, 
            DistanceMetric::L2, 
            10
        );
        
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
    }

    #[test]
    fn test_different_branching_factors() {
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| vec![i as f32, 0.0])
            .collect();

        for branching in [3, 5, 10] {
            let index = ClusteredIndex::build(vectors.clone(), branching, 50, DistanceMetric::L2, 10);
            
            assert_eq!(index.len(), 200);
            
            let query = vec![100.0, 0.0];
            let results = index.search(&query, 10, 2, 3);
            assert_eq!(results.len(), 10);
        }
    }
    
    #[test]
    fn test_mmap_storage() {
        use std::fs;
        
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| vec![i as f32; 128])
            .collect();
        let test_file = "test_mmap_vectors.bin";
        
        // Build index
        let mut index = ClusteredIndex::build(
            vectors.clone(),
            10,
            100,
            DistanceMetric::L2,
            20,
        );
        
        // Get memory usage before mmap
        let mem_before = index.memory_usage_bytes();
        assert!(!index.is_using_mmap());
        
        // Convert to mmap
        index.use_mmap_storage(test_file).unwrap();
        assert!(index.is_using_mmap());
        
        // Memory usage should be lower
        let mem_after = index.memory_usage_bytes();
        assert!(mem_after < mem_before);
        
        // Query should still work
        let query = vectors[0].clone();
        let results = index.search(&query, 10, 4, 5);
        assert_eq!(results.len(), 10);
        
        // First result should be the query itself
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.001); // Near-zero distance
        
        // Cleanup
        fs::remove_file(test_file).ok();
    }
    
    #[test]
    fn test_mmap_storage_twice_fails() {
        use std::fs;
        
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![i as f32; 32])
            .collect();
        let test_file = "test_mmap_twice.bin";
        
        let mut index = ClusteredIndex::build(
            vectors.clone(),
            5,
            50,
            DistanceMetric::L2,
            10,
        );
        
        // First conversion should succeed
        assert!(index.use_mmap_storage(test_file).is_ok());
        
        // Second conversion should fail
        let result = index.use_mmap_storage("test_mmap_twice_2.bin");
        assert!(result.is_err());
        
        // Cleanup
        fs::remove_file(test_file).ok();
    }
}
