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
pub struct ClusteredIndex {
    /// All nodes in the tree (indexed by node ID)
    nodes: Vec<ClusterNode>,
    /// Root node IDs (level 0)
    root_ids: Vec<usize>,
    /// Binary quantizer
    quantizer: BinaryQuantizer,
    /// Lookup table: original_index → full precision vector
    full_vectors: Vec<Vec<f32>>,
    /// Distance metric
    metric: DistanceMetric,
    /// Dimensionality
    dimension: usize,
    /// Number of levels in hierarchy
    num_levels: usize,
}

impl ClusteredIndex {
    /// Build a hierarchical clustered index
    /// 
    /// # Arguments
    /// * `vectors` - The vectors to index
    /// * `branching_factor` - Number of clusters at each level (~10-20 typical)
    /// * `metric` - Distance metric to use
    /// * `max_iterations` - Maximum k-means iterations
    pub fn build(
        vectors: Vec<Vec<f32>>,
        branching_factor: usize,
        metric: DistanceMetric,
        max_iterations: usize,
    ) -> Self {
        assert!(!vectors.is_empty(), "Cannot build index from empty vectors");
        assert!(branching_factor >= 2, "Branching factor must be >= 2");
        
        let dimension = vectors[0].len();
        let num_vectors = vectors.len();

        println!("Building hierarchical index...");
        println!("  Vectors: {}", num_vectors);
        println!("  Branching factor: {}", branching_factor);

        // Create binary quantizer
        let quantizer = BinaryQuantizer::from_vectors(&vectors);

        // Calculate number of levels needed
        let num_levels = ((num_vectors as f64).log(branching_factor as f64).ceil() as usize).max(2);
        println!("  Levels: {}", num_levels);

        // Store full vectors for reranking
        let full_vectors = vectors.clone();

        // Build tree recursively
        let mut nodes = Vec::new();
        let mut next_node_id = 0;
        
        let root_ids = Self::build_recursive(
            &vectors,
            &quantizer,
            (0..num_vectors).collect(),
            0,
            num_levels,
            branching_factor,
            metric,
            max_iterations,
            &mut nodes,
            &mut next_node_id,
        );

        println!("  Total nodes: {}", nodes.len());
        println!("  Root nodes: {}", root_ids.len());

        Self {
            nodes,
            root_ids,
            quantizer,
            full_vectors,
            metric,
            dimension,
            num_levels,
        }
    }

    /// Recursively build tree levels
    #[allow(clippy::too_many_arguments)]
    fn build_recursive(
        all_vectors: &[Vec<f32>],
        quantizer: &BinaryQuantizer,
        indices: Vec<usize>,
        current_level: usize,
        num_levels: usize,
        branching_factor: usize,
        metric: DistanceMetric,
        max_iterations: usize,
        nodes: &mut Vec<ClusterNode>,
        next_node_id: &mut usize,
    ) -> Vec<usize> {
        let num_clusters = branching_factor.min(indices.len());
        
        // Get vectors for this subset
        let subset_vectors: Vec<Vec<f32>> = indices
            .iter()
            .map(|&idx| all_vectors[idx].clone())
            .collect();

        if subset_vectors.is_empty() {
            return Vec::new();
        }

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

            // Check if this is a leaf node
            let is_leaf = current_level == num_levels - 1 || cluster_indices.len() <= branching_factor;

            let node = if is_leaf {
                // Leaf node - store vector indices
                ClusterNode {
                    id: node_id,
                    binary_centroid: binary_centroids[cluster_id].clone(),
                    full_centroid: kmeans.centroids[cluster_id].clone(),
                    children: Vec::new(),
                    vector_indices: cluster_indices,
                }
            } else {
                // Internal node - recursively build children
                let children = Self::build_recursive(
                    all_vectors,
                    quantizer,
                    cluster_indices,
                    current_level + 1,
                    num_levels,
                    branching_factor,
                    metric,
                    max_iterations,
                    nodes,
                    next_node_id,
                );

                ClusterNode {
                    id: node_id,
                    binary_centroid: binary_centroids[cluster_id].clone(),
                    full_centroid: kmeans.centroids[cluster_id].clone(),
                    children,
                    vector_indices: Vec::new(),
                }
            };

            nodes.push(node);
            node_ids.push(node_id);
        }

        node_ids
    }

    /// Search with hierarchical clustering
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
        
        // Traverse tree level by level
        for _ in 0..self.num_levels {
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
                // Leaf level - collect candidates
                return self.search_leaves(&top_nodes, query, &query_binary, k, rerank_factor);
            }

            // Expand to children for next level
            current_nodes = top_nodes
                .iter()
                .flat_map(|&node_id| self.nodes[node_id].children.clone())
                .collect();
        }

        Vec::new()
    }

    /// Search within leaf nodes
    fn search_leaves(
        &self,
        leaf_ids: &[usize],
        query: &[f32],
        query_binary: &BinaryVector,
        k: usize,
        rerank_factor: usize,
    ) -> Vec<(usize, f32)> {
        let rerank_k = (k * rerank_factor).min(10000);

        // Collect all binary candidates from leaves
        let mut binary_candidates = Vec::new();
        
        for &leaf_id in leaf_ids {
            let leaf = &self.nodes[leaf_id];
            for &vector_idx in &leaf.vector_indices {
                let binary_vec = self.quantizer.quantize(&self.full_vectors[vector_idx]);
                let hamming_dist = hamming_distance(query_binary, &binary_vec);
                binary_candidates.push((vector_idx, hamming_dist));
            }
        }

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

        // Rerank with full precision
        let mut reranked = Vec::with_capacity(rerank_k);
        
        for (original_idx, _) in binary_candidates {
            let full_vec = &self.full_vectors[original_idx];
            let dist = distance(query, full_vec, self.metric);
            reranked.push((original_idx, dist));
        }

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

    /// Get number of levels
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Get number of nodes in tree
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
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

        let index = ClusteredIndex::build(vectors, 5, DistanceMetric::L2, 10);

        assert_eq!(index.len(), 100);
        assert!(index.num_levels() >= 2);
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

        let index = ClusteredIndex::build(vectors, 5, DistanceMetric::L2, 10);

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
    fn test_different_branching_factors() {
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| vec![i as f32, 0.0])
            .collect();

        for branching in [3, 5, 10] {
            let index = ClusteredIndex::build(vectors.clone(), branching, DistanceMetric::L2, 10);
            
            assert_eq!(index.len(), 200);
            
            let query = vec![100.0, 0.0];
            let results = index.search(&query, 10, 2, 3);
            assert_eq!(results.len(), 10);
        }
    }
}
