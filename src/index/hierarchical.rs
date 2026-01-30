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
use std::time::Instant;

/// A node in the hierarchical tree
#[derive(Debug, Clone)]
struct ClusterNode {
    /// Node ID
    #[allow(dead_code)]
    id: usize,
    /// Binary quantized centroid
    #[allow(dead_code)]
    binary_centroid: BinaryVector,
    /// Full precision centroid (for reranking)
    full_centroid: Vec<f32>,
    /// Children node IDs (empty for leaf nodes)
    children: Vec<usize>,
    /// Vector indices (only for leaf nodes)
    vector_indices: Vec<usize>,
}

/// Search statistics for observability
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub total_leaves: usize,
    pub leaves_searched: usize,
    pub total_vectors: usize,
    pub vectors_scanned_binary: usize,
    pub vectors_reranked_full: usize,
    pub tree_depth: usize,
    pub probes_per_level: Vec<usize>,
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
    /// * `branching_factor` - Number of clusters at each level
    ///   - High values (50-100) create shallow, wide trees (better for large datasets)
    ///   - Low values (10-20) create deep, narrow trees (better for hierarchical data)
    /// * `max_leaf_size` - Maximum vectors per leaf (splits if larger, ~20-30 recommended)
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
        let binary_vectors = quantizer.quantize_batch_parallel(&vectors);

        // Write vectors to disk and create memory-mapped store
        let full_vectors = MmapVectorStore::create(&vector_file, &vectors)?;
        let size_mb = full_vectors.size_bytes() as f64 / 1_048_576.0;
        println!("  Vector file: {:.2} MB", size_mb);

        // Build tree recursively with adaptive splitting
        let mut nodes = Vec::new();
        let mut next_node_id = 0;
        let mut max_depth_reached = 0;

        println!("Building tree...");
        let build_start = Instant::now();
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
        let build_time = build_start.elapsed();
        println!("Build time: {:?}", build_time);
        println!("  Max depth: {}", max_depth_reached);
        println!("  Total nodes: {}", nodes.len());
        println!("  Root nodes: {}", root_ids.len());

        // Calculate leaf statistics
        let leaf_sizes: Vec<usize> = nodes
            .iter()
            .filter(|n| n.children.is_empty())
            .map(|n| n.vector_indices.len())
            .collect();
        
        let num_leaves = leaf_sizes.len();
        if !leaf_sizes.is_empty() {
            let avg_leaf = leaf_sizes.iter().sum::<usize>() as f64 / leaf_sizes.len() as f64;
            let max_leaf = *leaf_sizes.iter().max().unwrap();
            let min_leaf = *leaf_sizes.iter().min().unwrap();
            println!("  Leaves: {} total, avg size: {:.1}, min: {}, max: {}", 
                     num_leaves, avg_leaf, min_leaf, max_leaf);
            
            // Print leaf size distribution
            Self::print_leaf_distribution(&leaf_sizes);
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
        target_leaf_size: usize,
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

        // Base case: Create leaf node if we're close to target size or too deep
        let num_vectors = indices.len();
        if num_vectors <= target_leaf_size * 2 || current_depth >= 10 {
            let node_id = *next_node_id;
            *next_node_id += 1;

            let leaf = Self::create_leaf_from_indices(indices, all_vectors, quantizer, node_id);
            nodes.push(leaf);
            return vec![node_id];
        }

        // Adaptive branching: Choose branching factor to target target_leaf_size per leaf
        // If we have N vectors and want leaves of ~target_leaf_size, split into ~(N / target_leaf_size) clusters
        let target_clusters = (num_vectors as f32 / target_leaf_size as f32).ceil() as usize;
        let num_clusters = target_clusters
            .max(2)  // At least 2 clusters to make progress
            .min(branching_factor)  // Don't exceed max branching factor
            .min(num_vectors);  // Can't have more clusters than vectors
        
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
        
        // Collect all non-empty clusters with their indices
        let mut clusters: Vec<(usize, Vec<usize>)> = (0..num_clusters)
            .filter_map(|cluster_id| {
                let cluster_indices = Self::extract_cluster_indices(&indices, &assignment, cluster_id);
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
                let small_centroid = &kmeans.centroids[small_cluster.0];
                
                // Find closest cluster by centroid distance
                let (best_idx, _) = clusters
                    .iter()
                    .enumerate()
                    .map(|(idx, (cid, _))| {
                        let dist = distance(small_centroid, &kmeans.centroids[*cid], metric);
                        (idx, dist)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                
                // Always merge small clusters - we'll recursively split if too large
                clusters[best_idx].1.extend(small_cluster.1);
            } else {
                i += 1;
            }
        }
        
        // Build nodes from merged clusters
        for (_cluster_id, cluster_indices) in clusters {
            // Recursively build subtree - let recursion decide if it should be a leaf
            let children = Self::build_recursive(
                all_vectors,
                quantizer,
                cluster_indices,
                current_depth + 1,
                max_depth_reached,
                target_leaf_size,
                branching_factor,
                metric,
                max_iterations,
                nodes,
                next_node_id,
            );

            // If recursion returned only 1 child, skip creating an internal node
            // and use the child directly (avoids single-child internal nodes)
            if children.len() == 1 {
                node_ids.push(children[0]);
            } else {
                // Create internal node for this cluster
                // Use the original cluster_id for centroid lookup
                let node_id = *next_node_id;
                *next_node_id += 1;

                let node = Self::create_internal_from_cluster(
                    node_id,
                    _cluster_id,
                    &binary_centroids,
                    &kmeans.centroids,
                    children,
                );

                nodes.push(node);
                node_ids.push(node_id);
            }
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

        // Quantize query for leaf-level filtering
        let query_binary = self.quantizer.quantize(query);

        // Accumulate all leaves we encounter during traversal
        let mut accumulated_leaves = Vec::new();
        
        // Start from root and traverse down using FULL PRECISION centroids
        let mut current_nodes = self.root_ids.clone();
        
        // Traverse tree, accumulating leaves along the way
        loop {
            if current_nodes.is_empty() {
                break;
            }

            // Find nearest nodes using FULL PRECISION centroids (accurate routing)
            use rayon::prelude::*;
            let mut node_distances: Vec<(usize, f32)> = if current_nodes.len() > 10 {
                // Parallel distance computation for many nodes
                current_nodes
                    .par_iter()
                    .map(|&node_id| {
                        let node = &self.nodes[node_id];
                        let dist = distance(query, &node.full_centroid, self.metric);
                        (node_id, dist)
                    })
                    .collect()
            } else {
                // Sequential for few nodes (less overhead)
                current_nodes
                    .iter()
                    .map(|&node_id| {
                        let node = &self.nodes[node_id];
                        let dist = distance(query, &node.full_centroid, self.metric);
                        (node_id, dist)
                    })
                    .collect()
            };

            node_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Take top probes_per_level nodes
            let top_nodes: Vec<usize> = node_distances
                .iter()
                .take(probes_per_level)
                .map(|(id, _)| *id)
                .collect();

            // Separate leaf nodes from internal nodes (tree may be unbalanced)
            let (leaf_nodes, internal_nodes): (Vec<usize>, Vec<usize>) = top_nodes
                .into_iter()
                .partition(|&node_id| self.nodes[node_id].children.is_empty());
            
            // Accumulate any leaves we found at this level
            accumulated_leaves.extend(leaf_nodes);
            
            // Continue traversing internal nodes to find deeper leaves
            current_nodes = internal_nodes
                .iter()
                .flat_map(|&node_id| self.nodes[node_id].children.clone())
                .collect();
        }

        // Search all accumulated leaves (both shallow and deep)
        if accumulated_leaves.is_empty() {
            Vec::new()
        } else {
            self.search_leaves(&accumulated_leaves, query, &query_binary, k, rerank_factor)
        }
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
    
    /// Count the actual number of leaf nodes in the tree
    pub fn count_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.children.is_empty()).count()
    }
    
    /// Print a visualization of the tree structure (truncated if too wide)
    pub fn print_tree_structure(&self, max_width: usize) {
        println!("\n=== Tree Structure ===");
        
        let mut level = 0;
        let mut current_level: Vec<usize> = self.root_ids.clone();
        
        while !current_level.is_empty() && level <= self.max_depth {
            let num_nodes = current_level.len();
            let num_leaves = current_level.iter()
                .filter(|&&id| self.nodes[id].children.is_empty())
                .count();
            let num_internal = num_nodes - num_leaves;
            
            print!("Level {}: ", level);
            
            if num_nodes <= max_width {
                for &node_id in &current_level {
                    let node = &self.nodes[node_id];
                    if node.children.is_empty() {
                        print!("[{}] ", node.vector_indices.len());
                    } else {
                        print!("({}) ", node.children.len());
                    }
                }
                println!();
            } else {
                println!("{} nodes ({} internal, {} leaves)", num_nodes, num_internal, num_leaves);
                
                // Show first few and last few
                let show = (max_width / 2).min(5);
                for i in 0..show.min(num_nodes) {
                    let node = &self.nodes[current_level[i]];
                    if node.children.is_empty() {
                        print!("[{}] ", node.vector_indices.len());
                    } else {
                        print!("({}) ", node.children.len());
                    }
                }
                if num_nodes > show * 2 {
                    print!("... ");
                }
                for i in (num_nodes.saturating_sub(show))..num_nodes {
                    let node = &self.nodes[current_level[i]];
                    if node.children.is_empty() {
                        print!("[{}] ", node.vector_indices.len());
                    } else {
                        print!("({}) ", node.children.len());
                    }
                }
                println!();
            }
            
            // Collect next level
            let mut next_level = Vec::new();
            for &node_id in &current_level {
                next_level.extend_from_slice(&self.nodes[node_id].children);
            }
            current_level = next_level;
            level += 1;
        }
        
        println!("Legend: [n] = leaf with n vectors, (n) = internal node with n children\n");
    }
    
    /// Print leaf size distribution histogram
    fn print_leaf_distribution(leaf_sizes: &[usize]) {
        let mut sorted = leaf_sizes.to_vec();
        sorted.sort_unstable();
        
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let median = sorted[sorted.len() / 2];
        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[sorted.len() * 3 / 4];
        let p90 = sorted[sorted.len() * 9 / 10];
        
        println!("  Leaf size distribution: min={}, p25={}, median={}, p75={}, p90={}, max={}", 
                 min, p25, median, p75, p90, max);
        
        // Create histogram
        let num_bins = 10;
        let range = max - min + 1;
        let bin_size = (range as f64 / num_bins as f64).ceil() as usize;
        
        if bin_size > 0 {
            let mut bins = vec![0; num_bins];
            for &size in sorted.iter() {
                let bin = ((size - min) / bin_size).min(num_bins - 1);
                bins[bin] += 1;
            }
            
            println!("  Histogram:");
            for (i, &count) in bins.iter().enumerate() {
                let start = min + i * bin_size;
                let end = (start + bin_size - 1).min(max);
                let bar = "█".repeat((count * 40 / sorted.len()).max(1).min(40));
                println!("    {:3}-{:3}: {} {}", start, end, bar, count);
            }
        }
    }
    
    /// Search with statistics tracking
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        probes: usize,
        rerank_factor: usize,
    ) -> (Vec<(usize, f32)>, SearchStats) {
        assert_eq!(query.len(), self.dimension);
        
        let total_leaves = self.count_leaves();
        let query_binary = self.quantizer.quantize(query);
        
        // Track statistics
        let mut stats = SearchStats {
            total_leaves,
            leaves_searched: 0,
            total_vectors: self.binary_vectors.len(),
            vectors_scanned_binary: 0,
            vectors_reranked_full: 0,
            tree_depth: self.max_depth,
            probes_per_level: Vec::new(),
        };
        
        // Navigate tree (same logic as search() but with stats tracking)
        use rayon::prelude::*;
        let mut current_nodes = self.root_ids.clone();
        
        loop {
            stats.probes_per_level.push(current_nodes.len());
            
            if current_nodes.is_empty() {
                break;
            }
            
            // Compute distances
            let mut node_distances: Vec<(usize, f32)> = if current_nodes.len() > 10 {
                current_nodes
                    .par_iter()
                    .map(|&node_id| {
                        let node = &self.nodes[node_id];
                        let dist = distance(query, &node.full_centroid, self.metric);
                        (node_id, dist)
                    })
                    .collect()
            } else {
                current_nodes
                    .iter()
                    .map(|&node_id| {
                        let node = &self.nodes[node_id];
                        let dist = distance(query, &node.full_centroid, self.metric);
                        (node_id, dist)
                    })
                    .collect()
            };
            
            node_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            let top_nodes: Vec<usize> = node_distances
                .iter()
                .take(probes)
                .map(|(id, _)| *id)
                .collect();
            
            // Check if leaves
            let first_node = &self.nodes[top_nodes[0]];
            if first_node.children.is_empty() {
                // Reached leaves
                stats.leaves_searched = top_nodes.len();
                
                // Count vectors in leaves
                for &leaf_id in &top_nodes {
                    stats.vectors_scanned_binary += self.nodes[leaf_id].vector_indices.len();
                }
                
                // Search leaves and track reranking
                let candidates_with_dist = self.collect_leaf_candidates(&top_nodes, &query_binary);
                
                let rerank_count = (k * rerank_factor).min(candidates_with_dist.len());
                stats.vectors_reranked_full = rerank_count;
                
                let mut top_candidates = candidates_with_dist;
                top_candidates.sort_by_key(|x| x.1);
                top_candidates.truncate(rerank_count);
                
                let reranked = self.rerank_with_full_precision(&top_candidates, query);
                let results = self.finalize_results(reranked, k);
                
                return (results, stats);
            }
            
            // Expand to children
            current_nodes = top_nodes
                .iter()
                .flat_map(|&node_id| self.nodes[node_id].children.clone())
                .collect();
        }
        
        (Vec::new(), stats)
    }
    
    /// Print search statistics
    pub fn print_search_stats(&self, stats: &SearchStats, probes: usize) {
        println!("\n=== Search Statistics ===");
        println!("Tree traversal:");
        println!("  Depth: {}", stats.tree_depth);
        println!("  Probes requested: {}", probes);
        for (level, &count) in stats.probes_per_level.iter().enumerate() {
            println!("    Level {}: {} nodes explored", level, count);
        }
        
        println!("\nLeaf coverage:");
        println!("  Total leaves: {}", stats.total_leaves);
        println!("  Leaves searched: {}", stats.leaves_searched);
        println!("  Coverage: {:.2}%", 
                 stats.leaves_searched as f64 / stats.total_leaves as f64 * 100.0);
        
        println!("\nVector processing:");
        println!("  Total vectors: {}", stats.total_vectors);
        println!("  Binary scanned: {} ({:.2}%)", 
                 stats.vectors_scanned_binary,
                 stats.vectors_scanned_binary as f64 / stats.total_vectors as f64 * 100.0);
        println!("  Full precision reranked: {} ({:.2}%)", 
                 stats.vectors_reranked_full,
                 stats.vectors_reranked_full as f64 / stats.total_vectors as f64 * 100.0);
        println!("  Final results returned: varies by k\n");
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
