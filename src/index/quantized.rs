//! Quantized clustered index using binary vectors
//! 
//! Uses binary quantization for fast filtering, then reranks with full precision.
//! Achieves 32x compression + faster distance computation.

use crate::clustering::KMeans;
use crate::distance::{distance, DistanceMetric};
use crate::quantization::{BinaryQuantizer, BinaryVector, hamming_distance};

/// A clustered index that uses binary quantization for fast search
pub struct QuantizedClusteredIndex {
    /// K-means clustering
    kmeans: KMeans,
    /// Binary quantizer
    quantizer: BinaryQuantizer,
    /// Binary quantized centroids
    binary_centroids: Vec<BinaryVector>,
    /// Binary vectors in each cluster
    binary_clusters: Vec<Vec<(usize, BinaryVector)>>, // (original_index, binary_vector)
    /// Lookup table: original_index → full precision vector (O(1) reranking)
    full_vectors: Vec<Vec<f32>>,
    /// Distance metric
    metric: DistanceMetric,
    /// Dimensionality
    dimension: usize,
}

impl QuantizedClusteredIndex {
    /// Build a quantized clustered index
    /// 
    /// # Arguments
    /// * `vectors` - The vectors to index
    /// * `num_clusters` - Number of clusters to create
    /// * `metric` - Distance metric to use
    /// * `max_iterations` - Maximum k-means iterations
    pub fn build(
        vectors: Vec<Vec<f32>>,
        num_clusters: usize,
        metric: DistanceMetric,
        max_iterations: usize,
    ) -> Self {
        assert!(!vectors.is_empty(), "Cannot build index from empty vectors");
        
        let dimension = vectors[0].len();
        let num_clusters = num_clusters.min(vectors.len());

        println!("Building quantized clustered index with {} clusters...", num_clusters);

        // Create binary quantizer from data
        let quantizer = BinaryQuantizer::from_vectors(&vectors);

        // Run k-means clustering
        let (kmeans, assignment) = KMeans::fit(&vectors, num_clusters, metric, max_iterations);

        // Quantize centroids
        println!("Quantizing centroids...");
        let binary_centroids = quantizer.quantize_batch(&kmeans.centroids);

        // Group binary vectors by cluster and store full vectors in lookup table
        let mut binary_clusters: Vec<Vec<(usize, BinaryVector)>> = vec![Vec::new(); num_clusters];
        let num_vectors = vectors.len();
        let mut full_vectors = Vec::with_capacity(num_vectors);
        
        println!("Quantizing {} vectors...", num_vectors);
        let binary_vectors = quantizer.quantize_batch_parallel(&vectors);
        
        // Store full vectors in order (original_idx = position in this vec)
        for vector in vectors {
            full_vectors.push(vector);
        }
        
        // Group binary vectors by cluster
        for (original_idx, (binary_vec, &cluster_id)) in binary_vectors.into_iter()
            .zip(assignment.assignments.iter())
            .enumerate()
        {
            binary_clusters[cluster_id].push((original_idx, binary_vec));
        }

        // Print statistics
        let sizes: Vec<usize> = binary_clusters.iter().map(|c| c.len()).collect();
        let avg_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        let max_size = sizes.iter().max().unwrap();
        let min_size = sizes.iter().min().unwrap();
        
        println!("Cluster sizes: min={}, max={}, avg={:.1}", min_size, max_size, avg_size);
        println!("Compression: {} bytes → {} bytes ({}x)", 
            dimension * 4, 
            (dimension + 7) / 8,
            dimension * 4 / ((dimension + 7) / 8).max(1)
        );

        Self {
            kmeans,
            quantizer,
            binary_centroids,
            binary_clusters,
            full_vectors,
            metric,
            dimension,
        }
    }

    /// Search with binary quantization + reranking
    /// 
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `num_probes` - Number of clusters to search
    /// * `rerank_factor` - How many binary candidates to rerank (e.g., 10 means rerank 10*k)
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        num_probes: usize,
        rerank_factor: usize,
    ) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dimension, "Query dimension mismatch");

        let num_probes = num_probes.min(self.kmeans.k);
        let rerank_k = (k * rerank_factor).min(10000); // Cap rerank candidates

        // Quantize query
        let query_binary = self.quantizer.quantize(query);

        // 1. Find nearest clusters using binary centroids (fast!)
        let mut cluster_distances: Vec<(usize, u32)> = self
            .binary_centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| (i, hamming_distance(&query_binary, centroid)))
            .collect();

        cluster_distances.sort_by_key(|x| x.1);

        // 2. Fast: Add all binary vectors in top clusters to candidates (with computed hamming distance)
        let mut binary_candidates = Vec::new();
        
        for (cluster_id, _) in cluster_distances.iter().take(num_probes) {
            for (original_idx, binary_vec) in &self.binary_clusters[*cluster_id] {
                let hamming_dist = hamming_distance(&query_binary, binary_vec);
                binary_candidates.push((*original_idx, hamming_dist));
            }
        }

        // 3. Select top rerank_k candidates from binary search
        if binary_candidates.is_empty() {
            return Vec::new();
        }

        if binary_candidates.len() <= rerank_k {
            binary_candidates.sort_by_key(|x| x.1);
        } else {
            binary_candidates.select_nth_unstable_by(rerank_k - 1, |a, b| a.1.cmp(&b.1));
            binary_candidates.truncate(rerank_k);
        }

        // 4. Rerank top candidates with full precision (O(1) lookup!)
        let mut reranked = Vec::with_capacity(rerank_k);
        
        for (original_idx, _) in binary_candidates {
            let full_vec = &self.full_vectors[original_idx];
            let dist = distance(query, full_vec, self.metric);
            reranked.push((original_idx, dist));
        }

        // 5. Return final top-k
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

    /// Get number of clusters
    pub fn num_clusters(&self) -> usize {
        self.kmeans.k
    }

    /// Get total number of vectors
    pub fn len(&self) -> usize {
        self.full_vectors.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get dimensionality
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_index_build() {
        let mut vectors = Vec::new();
        
        // Create 100 vectors
        for i in 0..100 {
            vectors.push(vec![i as f32, 0.0, 0.0]);
        }

        let index = QuantizedClusteredIndex::build(vectors, 5, DistanceMetric::L2, 10);

        assert_eq!(index.num_clusters(), 5);
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_quantized_search() {
        let mut vectors = Vec::new();
        
        // Cluster 1: around [0, 0]
        for i in 0..50 {
            vectors.push(vec![i as f32 * 0.01, 0.0]);
        }
        // Cluster 2: around [10, 10]
        for i in 0..50 {
            vectors.push(vec![10.0 + i as f32 * 0.01, 10.0]);
        }

        let index = QuantizedClusteredIndex::build(vectors, 2, DistanceMetric::L2, 10);

        // Query near first cluster
        let query = vec![0.25, 0.0];
        let results = index.search(&query, 5, 1, 3);

        assert_eq!(results.len(), 5);
        
        // Results should be from first cluster
        for (idx, _) in &results {
            assert!(*idx < 50, "Expected index from first cluster, got {}", idx);
        }
    }

    #[test]
    fn test_rerank_factor() {
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| vec![i as f32, 0.0])
            .collect();

        let index = QuantizedClusteredIndex::build(vectors, 4, DistanceMetric::L2, 10);

        let query = vec![100.0, 0.0];
        
        // Lower rerank factor = faster but potentially lower accuracy
        let results_low = index.search(&query, 10, 2, 2);
        
        // Higher rerank factor = slower but better accuracy
        let results_high = index.search(&query, 10, 2, 5);

        assert_eq!(results_low.len(), 10);
        assert_eq!(results_high.len(), 10);
    }
}
