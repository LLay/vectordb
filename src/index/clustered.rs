//! Clustered index using k-means
//! 
//! Partitions vectors into k clusters using k-means.
//! Search first finds nearest cluster(s), then searches within those clusters.
//! Much faster than flat index for large datasets (O(k + n/k) vs O(n)).

use crate::clustering::KMeans;
use crate::distance::{distance, DistanceMetric};
use rayon::prelude::*;

/// A clustered index that uses k-means for faster search
pub struct ClusteredIndex {
    /// K-means clustering
    kmeans: KMeans,
    /// Vectors in each cluster
    clusters: Vec<Vec<(usize, Vec<f32>)>>, // (original_index, vector)
    /// Distance metric to use
    metric: DistanceMetric,
    /// Dimensionality
    dimension: usize,
}

impl ClusteredIndex {
    /// Build a clustered index from vectors
    /// 
    /// # Arguments
    /// * `vectors` - The vectors to index
    /// * `num_clusters` - Number of clusters to create (typically sqrt(n))
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

        println!("Building clustered index with {} clusters...", num_clusters);

        // Run k-means clustering
        let (kmeans, assignment) = KMeans::fit(&vectors, num_clusters, metric, max_iterations);

        // Group vectors by cluster
        let mut clusters: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); num_clusters];
        
        for (original_idx, (vector, &cluster_id)) in vectors.into_iter()
            .zip(assignment.assignments.iter())
            .enumerate()
        {
            clusters[cluster_id].push((original_idx, vector));
        }

        // Print cluster statistics
        let sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
        let avg_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        let max_size = sizes.iter().max().unwrap();
        let min_size = sizes.iter().min().unwrap();
        
        println!("Cluster sizes: min={}, max={}, avg={:.1}", min_size, max_size, avg_size);

        Self {
            kmeans,
            clusters,
            metric,
            dimension,
        }
    }

    /// Search for k nearest neighbors
    /// 
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `num_probes` - Number of clusters to search (default: 1, more = better recall)
    pub fn search(&self, query: &[f32], k: usize, num_probes: usize) -> Vec<(usize, f32)> {
        assert_eq!(
            query.len(),
            self.dimension,
            "Query dimension mismatch"
        );

        let num_probes = num_probes.min(self.kmeans.k);

        // Find nearest clusters to query
        let mut cluster_distances: Vec<(usize, f32)> = self
            .kmeans
            .centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| (i, distance(query, centroid, self.metric)))
            .collect();

        // Sort by distance to get nearest clusters
        cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Search within top num_probes clusters
        let mut candidates = Vec::new();
        
        for (cluster_id, _) in cluster_distances.iter().take(num_probes) {
            for (original_idx, vector) in &self.clusters[*cluster_id] {
                let dist = distance(query, vector, self.metric);
                candidates.push((*original_idx, dist));
            }
        }

        // Return top k
        if candidates.len() <= k {
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return candidates;
        }

        // Partial sort to get top k
        candidates.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut results = candidates[..k].to_vec();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    /// Parallel search across multiple probes
    pub fn search_parallel(&self, query: &[f32], k: usize, num_probes: usize) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dimension, "Query dimension mismatch");

        let num_probes = num_probes.min(self.kmeans.k);

        // Find nearest clusters
        let mut cluster_distances: Vec<(usize, f32)> = self
            .kmeans
            .centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| (i, distance(query, centroid, self.metric)))
            .collect();

        cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Search clusters in parallel
        let candidates: Vec<(usize, f32)> = cluster_distances
            .iter()
            .take(num_probes)
            .flat_map(|(cluster_id, _)| {
                self.clusters[*cluster_id]
                    .par_iter()
                    .map(|(original_idx, vector)| {
                        let dist = distance(query, vector, self.metric);
                        (*original_idx, dist)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Return top k
        let mut results = candidates;
        if results.len() <= k {
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return results;
        }

        results.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut top_k = results[..k].to_vec();
        top_k.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        top_k
    }

    /// Get number of clusters
    pub fn num_clusters(&self) -> usize {
        self.kmeans.k
    }

    /// Get total number of vectors
    pub fn len(&self) -> usize {
        self.clusters.iter().map(|c| c.len()).sum()
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
    fn test_clustered_index_build() {
        let mut vectors = Vec::new();
        
        // Create 100 vectors around two points
        for i in 0..50 {
            vectors.push(vec![i as f32 * 0.1, 0.0, 0.0]);
        }
        for i in 0..50 {
            vectors.push(vec![10.0 + i as f32 * 0.1, 0.0, 0.0]);
        }

        let index = ClusteredIndex::build(vectors, 2, DistanceMetric::L2, 10);

        assert_eq!(index.num_clusters(), 2);
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_clustered_index_search() {
        let mut vectors = Vec::new();
        
        // Cluster 1: around [0, 0]
        for i in 0..50 {
            vectors.push(vec![i as f32 * 0.01, 0.0]);
        }
        // Cluster 2: around [10, 10]
        for i in 0..50 {
            vectors.push(vec![10.0 + i as f32 * 0.01, 10.0]);
        }

        let index = ClusteredIndex::build(vectors, 2, DistanceMetric::L2, 10);

        // Query near first cluster
        let query = vec![0.25, 0.0];
        let results = index.search(&query, 5, 1);

        assert_eq!(results.len(), 5);
        
        // All results should be from first cluster (indices 0-49)
        for (idx, _) in &results {
            assert!(*idx < 50, "Expected index from first cluster, got {}", idx);
        }
    }

    #[test]
    fn test_multiprobe_search() {
        let mut vectors = Vec::new();
        
        for i in 0..100 {
            vectors.push(vec![i as f32, 0.0]);
        }

        let index = ClusteredIndex::build(vectors, 4, DistanceMetric::L2, 10);

        let query = vec![50.0, 0.0];
        
        // Single probe might miss some close vectors
        let results_1 = index.search(&query, 10, 1);
        
        // Multiple probes should find better results
        let results_4 = index.search(&query, 10, 4);

        assert_eq!(results_1.len(), 10);
        assert_eq!(results_4.len(), 10);
        
        // Multi-probe should have equal or better (lower) average distance
        let avg_dist_1: f32 = results_1.iter().map(|(_, d)| d).sum::<f32>() / 10.0;
        let avg_dist_4: f32 = results_4.iter().map(|(_, d)| d).sum::<f32>() / 10.0;
        
        assert!(avg_dist_4 <= avg_dist_1);
    }
}
