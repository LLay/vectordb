//! Hierarchical clustered index with RaBitQ quantization
//! 
//! Uses multi-level clustering (tree structure) for fast search:
//! - Level 0 (root): ~10 clusters
//! - Level 1: ~10 sub-clusters per parent = ~100 total
//! - Leaf level: Vectors grouped by final cluster
//! 
//! RaBitQ quantization at each level for fast filtering + precise reranking.
//! RaBitQ provides unbiased distance estimation with theoretical guarantees.

use crate::clustering::KMeans;
use crate::distance::{distance, DistanceMetric};
use crate::quantization::{RaBitQQuantizer, RaBitQVector};
use crate::storage::MmapVectorStore;
use std::path::Path;
use std::time::Instant;

/// A node in the hierarchical tree
#[derive(Debug, Clone)]
pub struct ClusterNodeRaBitQ {
    /// Node ID
    #[allow(dead_code)]
    pub id: usize,
    /// RaBitQ quantized centroid
    #[allow(dead_code)]
    pub rabitq_centroid: RaBitQVector,
    /// Full precision centroid (for reranking)
    pub full_centroid: Vec<f32>,
    /// Children node IDs (empty for leaf nodes)
    pub children: Vec<usize>,
    /// Vector indices (only for leaf nodes)
    pub vector_indices: Vec<usize>,
}

/// Search statistics for observability
#[derive(Debug, Clone)]
pub struct SearchStatsRaBitQ {
    pub total_leaves: usize,
    pub leaves_searched: usize,
    pub leaves_searched_ids: Vec<usize>,
    pub total_vectors: usize,
    pub vectors_scanned_rabitq: usize,
    pub vectors_reranked_full: usize,
    pub tree_depth: usize,
    pub probes_per_level: Vec<usize>,
}

/// Hierarchical clustered index with RaBitQ quantization
/// 
/// Uses memory-mapped storage for full precision vectors to minimize RAM usage.
/// Hot vectors are automatically cached by the OS.
pub struct ClusteredIndexRaBitQ {
    /// All nodes in the tree (indexed by node ID)
    pub nodes: Vec<ClusterNodeRaBitQ>,
    /// Root node IDs (level 0)
    pub root_ids: Vec<usize>,
    /// RaBitQ quantizer
    pub quantizer: RaBitQQuantizer,
    /// Pre-quantized RaBitQ vectors (for fast candidate scanning)
    pub rabitq_vectors: Vec<RaBitQVector>,
    /// Full precision vectors (memory-mapped for RAM efficiency)
    pub full_vectors: MmapVectorStore,
    /// Distance metric
    metric: DistanceMetric,
    /// Dimensionality
    dimension: usize,
    /// Maximum tree depth (deepest leaf)
    max_depth: usize,
    /// Maximum leaf size (vectors per leaf)
    max_leaf_size: usize,
}

impl ClusteredIndexRaBitQ {
    /// Build a hierarchical clustered index with RaBitQ quantization
    /// 
    /// # Arguments
    /// * `vectors` - The vectors to index
    /// * `vector_file` - Path where vectors will be stored on disk
    /// * `branching_factor` - Number of clusters at each level
    /// * `target_leaf_size` - Target vectors per leaf
    /// * `metric` - Distance metric to use
    /// * `max_iterations` - Maximum k-means iterations
    pub fn build<P: AsRef<Path>>(
        vectors: Vec<Vec<f32>>,
        vector_file: P,
        branching_factor: usize,
        target_leaf_size: usize,
        metric: DistanceMetric,
        max_iterations: usize,
    ) -> std::io::Result<Self> {
        assert!(!vectors.is_empty(), "Cannot build index from empty vectors");
        assert!(branching_factor >= 2, "Branching factor must be >= 2");
        assert!(target_leaf_size >= 10, "Target leaf size must be >= 10");
        
        let dimension = vectors[0].len();
        let num_vectors = vectors.len();
        
        println!("Building adaptive hierarchical index with RaBitQ...");
        println!("  Vectors: {}", num_vectors);
        println!("  Branching factor: {}", branching_factor);
        println!("  Target leaf size: {}", target_leaf_size);
        
        // Create RaBitQ quantizer
        eprintln!("[RaBitQ Index] Creating quantizer for D={}...", dimension);
        let quantizer = RaBitQQuantizer::new(dimension);
        
        // Quantize all vectors using RaBitQ
        eprintln!("[RaBitQ Index] Quantizing {} vectors...", num_vectors);
        let rabitq_vectors = quantizer.quantize_batch(&vectors);
        
        // Save full-precision vectors to disk
        let file_size = num_vectors * dimension * 4;
        println!("  Vector file: {:.2} MB", file_size as f64 / 1024.0 / 1024.0);
        
        let full_vectors = MmapVectorStore::create(&vector_file, &vectors)?;
        
        // Build tree structure
        println!("Building tree...");
        let build_start = Instant::now();
        
        let mut all_nodes = Vec::new();
        let mut next_node_id = 0;
        
        // Start with all vector indices
        let all_indices: Vec<usize> = (0..num_vectors).collect();
        
        // Build tree recursively
        let root_ids = Self::build_tree_recursive(
            &all_indices,
            &vectors,
            &rabitq_vectors,
            &quantizer,
            branching_factor,
            target_leaf_size,
            metric,
            max_iterations,
            0, // depth
            &mut all_nodes,
            &mut next_node_id,
        );
        
        // Calculate max depth
        let max_depth = Self::calculate_max_depth(&all_nodes, &root_ids);
        
        let build_time = build_start.elapsed();
        println!("Build time: {:?}", build_time);
        
        // Print tree statistics
        Self::print_tree_stats(&all_nodes, &root_ids, max_depth);
        
        Ok(ClusteredIndexRaBitQ {
            nodes: all_nodes,
            root_ids,
            quantizer,
            rabitq_vectors,
            full_vectors,
            metric,
            dimension,
            max_depth,
            max_leaf_size: target_leaf_size * 2,
        })
    }
    
    /// Recursively build tree structure
    #[allow(clippy::too_many_arguments)]
    fn build_tree_recursive(
        indices: &[usize],
        full_vectors: &[Vec<f32>],
        rabitq_vectors: &[RaBitQVector],
        quantizer: &RaBitQQuantizer,
        branching_factor: usize,
        target_leaf_size: usize,
        metric: DistanceMetric,
        max_iterations: usize,
        depth: usize,
        all_nodes: &mut Vec<ClusterNodeRaBitQ>,
        next_node_id: &mut usize,
    ) -> Vec<usize> {
        if indices.is_empty() {
            return vec![];
        }
        
        let num_vectors = indices.len();
        
        // Base case: Create leaf node if we're close to target size or too deep
        // This is critical - we create the leaf and RETURN immediately
        if num_vectors <= target_leaf_size * 2 || depth >= 10 {
            let node_id = *next_node_id;
            *next_node_id += 1;
            
            // Compute centroid for this leaf
            let mut centroid = vec![0.0; full_vectors[0].len()];
            for &idx in indices {
                for (i, &val) in full_vectors[idx].iter().enumerate() {
                    centroid[i] += val;
                }
            }
            for val in &mut centroid {
                *val /= num_vectors as f32;
            }
            
            let rabitq_centroid = quantizer.quantize(&centroid);
            let leaf = ClusterNodeRaBitQ {
                id: node_id,
                rabitq_centroid,
                full_centroid: centroid,
                children: vec![],
                vector_indices: indices.to_vec(),
            };
            all_nodes.push(leaf);
            return vec![node_id];
        }
        
        // Adaptive branching: Choose branching factor to target target_leaf_size per leaf
        // If we have N vectors and want leaves of ~target_leaf_size, split into ~(N / target_leaf_size) clusters
        let target_clusters = (num_vectors as f32 / target_leaf_size as f32).ceil() as usize;
        let adaptive_k = target_clusters
            .max(2)  // At least 2 clusters to make progress
            .min(branching_factor)  // Don't exceed max branching factor
            .min(num_vectors);  // Can't have more clusters than vectors
        
        // Extract vectors for this subset
        let subset: Vec<Vec<f32>> = indices.iter()
            .map(|&i| full_vectors[i].clone())
            .collect();
        
        // Cluster the subset
        let (kmeans, assignment) = KMeans::fit(&subset, adaptive_k, metric, max_iterations);
        let assignments = assignment.assignments;
        let centroids = kmeans.centroids;
        
        // Collect all non-empty clusters with their indices
        let mut clusters: Vec<(usize, Vec<usize>)> = (0..adaptive_k)
            .filter_map(|cluster_id| {
                let cluster_indices: Vec<usize> = indices.iter()
                    .enumerate()
                    .filter_map(|(i, &idx)| {
                        if assignments[i] == cluster_id {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect();
                
                if cluster_indices.is_empty() {
                    None
                } else {
                    Some((cluster_id, cluster_indices))
                }
            })
            .collect();
        
        // Merge small clusters (< 30% of target) into neighbors to reduce fragmentation
        let min_cluster_size = (target_leaf_size * 3 / 10).max(10);
        let mut i = 0;
        while i < clusters.len() {
            if clusters[i].1.len() < min_cluster_size && clusters.len() > 1 {
                // Find the nearest cluster to merge with
                let small_cluster = clusters.remove(i);
                let small_centroid = &centroids[small_cluster.0];
                
                // Find closest cluster by centroid distance
                let (best_idx, _) = clusters
                    .iter()
                    .enumerate()
                    .map(|(idx, (cid, _))| {
                        let dist = distance(small_centroid, &centroids[*cid], metric);
                        (idx, dist)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                
                // Merge into the closest cluster
                clusters[best_idx].1.extend(small_cluster.1);
            } else {
                i += 1;
            }
        }
        
        // Build nodes from merged clusters
        let mut node_ids = Vec::new();
        
        for (cluster_id, cluster_indices) in clusters {
            // Recursively build subtree - let recursion decide if it should be a leaf
            let children = Self::build_tree_recursive(
                &cluster_indices,
                full_vectors,
                rabitq_vectors,
                quantizer,
                branching_factor,
                target_leaf_size,
                metric,
                max_iterations,
                depth + 1,
                all_nodes,
                next_node_id,
            );
            
            // If recursion returned only 1 child, skip creating an internal node
            // and use the child directly (avoids single-child internal nodes)
            if children.len() == 1 {
                node_ids.push(children[0]);
            } else {
                // Create internal node for this cluster
                let node_id = *next_node_id;
                *next_node_id += 1;
                
                let rabitq_centroid = quantizer.quantize(&centroids[cluster_id]);
                let node = ClusterNodeRaBitQ {
                    id: node_id,
                    rabitq_centroid,
                    full_centroid: centroids[cluster_id].clone(),
                    children,
                    vector_indices: vec![],
                };
                all_nodes.push(node);
                node_ids.push(node_id);
            }
        }
        
        node_ids
    }
    
    /// Calculate maximum depth of tree
    fn calculate_max_depth(nodes: &[ClusterNodeRaBitQ], root_ids: &[usize]) -> usize {
        fn depth_recursive(nodes: &[ClusterNodeRaBitQ], node_id: usize) -> usize {
            let node = &nodes[node_id];
            if node.children.is_empty() {
                1
            } else {
                1 + node.children.iter()
                    .map(|&child_id| depth_recursive(nodes, child_id))
                    .max()
                    .unwrap_or(0)
            }
        }
        
        root_ids.iter()
            .map(|&root_id| depth_recursive(nodes, root_id))
            .max()
            .unwrap_or(0)
    }
    
    /// Print tree statistics
    fn print_tree_stats(nodes: &[ClusterNodeRaBitQ], root_ids: &[usize], max_depth: usize) {
        let leaves: Vec<&ClusterNodeRaBitQ> = nodes.iter()
            .filter(|n| n.children.is_empty())
            .collect();
        
        if leaves.is_empty() {
            println!("  No leaves in tree!");
            return;
        }
        
        let leaf_sizes: Vec<usize> = leaves.iter().map(|n| n.vector_indices.len()).collect();
        let total_vecs: usize = leaf_sizes.iter().sum();
        let avg_size = total_vecs as f32 / leaves.len() as f32;
        let min_size = *leaf_sizes.iter().min().unwrap();
        let max_size = *leaf_sizes.iter().max().unwrap();
        
        println!("  Max depth: {}", max_depth);
        println!("  Total nodes: {}", nodes.len());
        println!("  Root nodes: {}", root_ids.len());
        println!("  Leaves: {} total, avg size: {:.1}, min: {}, max: {}",
                 leaves.len(), avg_size, min_size, max_size);
        
        // Leaf size distribution
        let mut sorted_sizes = leaf_sizes.clone();
        sorted_sizes.sort_unstable();
        let p25 = sorted_sizes[sorted_sizes.len() / 4];
        let p50 = sorted_sizes[sorted_sizes.len() / 2];
        let p75 = sorted_sizes[sorted_sizes.len() * 3 / 4];
        let p90 = sorted_sizes[sorted_sizes.len() * 9 / 10];
        
        println!("  Leaf size distribution: min={}, p25={}, median={}, p75={}, p90={}, max={}",
                 min_size, p25, p50, p75, p90, max_size);
        
        // Histogram
        if max_size > min_size {
            let num_bins = 10;
            let bin_width = (max_size - min_size + num_bins - 1) / num_bins; // Round up
            let mut bins = vec![0; num_bins];
            
            for &size in &leaf_sizes {
                let bin_idx = ((size - min_size) / bin_width).min(num_bins - 1);
                bins[bin_idx] += 1;
            }
            
            let max_count = *bins.iter().max().unwrap();
            let scale = 8; // Max bar width
            
            println!("  Histogram:");
            for (i, &count) in bins.iter().enumerate() {
                if count > 0 {
                    let bin_start = min_size + i * bin_width;
                    let bin_end = (bin_start + bin_width - 1).min(max_size);
                    let bar_len = (count * scale + max_count / 2) / max_count;
                    let bar = "█".repeat(bar_len);
                    println!("     {:3}-{:3}: {} {}", bin_start, bin_end, bar, count);
                }
            }
        }
    }
    
    /// Search for k nearest neighbors
    /// 
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `probes_per_level` - Number of nodes to explore at each level
    /// * `rerank_factor` - Multiplier for RaBitQ candidates (e.g., 10 means top 10k RaBitQ → top k full precision)
    /// 
    /// # Returns
    /// Vector of (index, distance) pairs for k nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        probes_per_level: usize,
        rerank_factor: usize,
    ) -> Vec<(usize, f32)> {
        self.search_internal(query, k, probes_per_level, rerank_factor, false).0
    }
    
    /// Internal search with optional statistics
    fn search_internal(
        &self,
        query: &[f32],
        k: usize,
        probes_per_level: usize,
        rerank_factor: usize,
        collect_stats: bool,
    ) -> (Vec<(usize, f32)>, Option<SearchStatsRaBitQ>) {
        assert_eq!(query.len(), self.dimension, "Query dimension mismatch");
        
        let mut stats = if collect_stats {
            Some(SearchStatsRaBitQ {
                total_leaves: self.nodes.iter().filter(|n| n.children.is_empty()).count(),
                leaves_searched: 0,
                leaves_searched_ids: Vec::new(),
                total_vectors: self.rabitq_vectors.len(),
                vectors_scanned_rabitq: 0,
                vectors_reranked_full: 0,
                tree_depth: self.max_depth,
                probes_per_level: Vec::new(),
            })
        } else {
            None
        };
        
        // Start from root nodes
        let mut current_nodes = self.root_ids.clone();
        
        // Traverse tree level by level
        while !current_nodes.is_empty() {
            // Check if all current nodes are leaves
            let all_leaves = current_nodes.iter()
                .all(|&node_id| self.nodes[node_id].children.is_empty());
            
            if all_leaves {
                break;
            }
            
            // Calculate adaptive probes for this level
            let adaptive_probes = std::cmp::min(probes_per_level, current_nodes.len());
            
            if let Some(ref mut s) = stats {
                s.probes_per_level.push(adaptive_probes);
            }
            
            // Compute distances to all current node centroids
            let mut node_distances: Vec<(usize, f32)> = current_nodes.iter()
                .map(|&node_id| {
                    let node = &self.nodes[node_id];
                    let dist = distance(&node.full_centroid, query, self.metric);
                    (node_id, dist)
                })
                .collect();
            
            // Select top nodes (partial sort is faster than full sort)
            if node_distances.len() > adaptive_probes {
                node_distances.select_nth_unstable_by(adaptive_probes, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                node_distances.truncate(adaptive_probes);
            }
            
            // Collect children from top nodes
            let mut next_nodes = Vec::new();
            for (node_id, _) in node_distances {
                let node = &self.nodes[node_id];
                if !node.children.is_empty() {
                    next_nodes.extend(&node.children);
                } else {
                    // Keep leaves in the list
                    next_nodes.push(node_id);
                }
            }
            
            current_nodes = next_nodes;
        }
        
        // Now current_nodes contains only leaf nodes
        // Collect all vector indices from these leaves
        let mut candidate_indices = Vec::new();
        for &node_id in &current_nodes {
            let node = &self.nodes[node_id];
            candidate_indices.extend(&node.vector_indices);
            
            if let Some(ref mut s) = stats {
                s.leaves_searched += 1;
                s.leaves_searched_ids.push(node_id);
            }
        }
        
        if candidate_indices.is_empty() {
            return (vec![], stats);
        }
        
        if let Some(ref mut s) = stats {
            s.vectors_scanned_rabitq = candidate_indices.len();
        }
        
        // Phase 1: Fast filtering with RaBitQ
        let candidate_refs: Vec<&RaBitQVector> = candidate_indices.iter()
            .map(|&idx| &self.rabitq_vectors[idx])
            .collect();
        
        let rabitq_distances = self.quantizer.estimate_distances_batch_fast(
            &candidate_refs,
            query,
        );
        
        // Create (index, distance) pairs and select top candidates for reranking
        let rerank_count = std::cmp::min(k * rerank_factor, candidate_indices.len());
        let mut rabitq_results: Vec<(usize, f32)> = candidate_indices.iter()
            .zip(rabitq_distances.iter())
            .map(|(&idx, &dist)| (idx, dist))
            .collect();
        
        // Partial sort to get top rerank_count candidates
        if rabitq_results.len() > rerank_count {
            rabitq_results.select_nth_unstable_by(rerank_count, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            rabitq_results.truncate(rerank_count);
        }
        
        if let Some(ref mut s) = stats {
            s.vectors_reranked_full = rabitq_results.len();
        }
        
        // Phase 2: Precise reranking with full precision
        let mut full_results: Vec<(usize, f32)> = rabitq_results.iter()
            .map(|(idx, _)| {
                let vec = self.full_vectors.get(*idx);
                let dist = distance(&vec, query, self.metric);
                (*idx, dist)
            })
            .collect();
        
        // Final sort to get top k
        let final_k = std::cmp::min(k, full_results.len());
        if full_results.len() > final_k {
            full_results.select_nth_unstable_by(final_k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            full_results.truncate(final_k);
        }
        
        full_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        (full_results, stats)
    }
    
    /// Search with statistics for debugging
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        probes_per_level: usize,
        rerank_factor: usize,
    ) -> (Vec<(usize, f32)>, SearchStatsRaBitQ) {
        let (results, stats) = self.search_internal(query, k, probes_per_level, rerank_factor, true);
        (results, stats.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rabitq_index_build_and_search() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.9, 0.1, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.9, 0.1],
        ];
        
        let index = ClusteredIndexRaBitQ::build(
            vectors.clone(),
            "test_rabitq_index.bin",
            2,
            20,
            DistanceMetric::L2,
            10,
        ).expect("Failed to build index");
        
        // Search for nearest neighbor
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 2, 2, 10);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // First vector should be closest
        
        // Clean up
        std::fs::remove_file("test_rabitq_index.bin").ok();
    }
}
