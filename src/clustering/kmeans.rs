//! K-means clustering implementation
//! 
//! Groups vectors into k clusters based on similarity.
//! Used to build hierarchical indices for faster search.

use crate::distance::{distance, DistanceMetric};
use rand::seq::SliceRandom;
use rayon::prelude::*;

/// K-means clustering result
#[derive(Debug, Clone)]
pub struct KMeans {
    /// Cluster centroids
    pub centroids: Vec<Vec<f32>>,
    /// Number of clusters
    pub k: usize,
    /// Dimensionality
    pub dimension: usize,
    /// Distance metric used
    pub metric: DistanceMetric,
}

/// Assignment of vectors to clusters
#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    /// Which cluster each vector belongs to (index)
    pub assignments: Vec<usize>,
    /// Distance from each vector to its assigned centroid
    pub distances: Vec<f32>,
}

impl KMeans {
    /// Initialize k-means with k-means++ algorithm
    /// 
    /// Selects initial centroids that are far apart from each other
    pub fn init_plusplus(
        vectors: &[Vec<f32>],
        k: usize,
        metric: DistanceMetric,
    ) -> Self {
        assert!(!vectors.is_empty(), "Cannot cluster empty vector set");
        assert!(k > 0 && k <= vectors.len(), "k must be between 1 and number of vectors");

        let dimension = vectors[0].len();
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // 1. Choose first centroid randomly
        let first = vectors.choose(&mut rng).unwrap().clone();
        centroids.push(first);

        // 2. For each remaining centroid, choose with probability proportional to distance squared
        for _ in 1..k {
            // Compute distance from each vector to nearest centroid
            let distances: Vec<f32> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| distance(v, c, metric))
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                })
                .collect();

            // Sample proportional to squared distance
            let total: f32 = distances.iter().map(|d| d * d).sum();
            let mut target = rand::random::<f32>() * total;

            let mut chosen_idx = 0;
            for (i, d) in distances.iter().enumerate() {
                target -= d * d;
                if target <= 0.0 {
                    chosen_idx = i;
                    break;
                }
            }

            centroids.push(vectors[chosen_idx].clone());
        }

        Self {
            centroids,
            k,
            dimension,
            metric,
        }
    }

    /// Run k-means clustering
    /// 
    /// Returns the final cluster assignment after convergence
    pub fn fit(
        vectors: &[Vec<f32>],
        k: usize,
        metric: DistanceMetric,
        max_iterations: usize,
    ) -> (Self, ClusterAssignment) {
        let mut kmeans = Self::init_plusplus(vectors, k, metric);
        let mut assignment = kmeans.assign(vectors);

        for _ in 0..max_iterations {
            // Update centroids
            let new_centroids = kmeans.update_centroids(vectors, &assignment);
            
            // Check for convergence (centroids didn't change)
            let converged = new_centroids
                .iter()
                .zip(kmeans.centroids.iter())
                .all(|(new, old)| {
                    new.iter()
                        .zip(old.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-6)
                });

            kmeans.centroids = new_centroids;

            if converged {
                break;
            }

            // Reassign vectors
            assignment = kmeans.assign(vectors);
        }

        (kmeans, assignment)
    }

    /// Assign vectors to nearest centroids
    pub fn assign(&self, vectors: &[Vec<f32>]) -> ClusterAssignment {
        let (assignments, distances): (Vec<_>, Vec<_>) = vectors
            .par_iter()
            .map(|v| {
                // Find nearest centroid
                let (best_idx, best_dist) = self
                    .centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, distance(v, c, self.metric)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                (best_idx, best_dist)
            })
            .unzip();

        ClusterAssignment {
            assignments,
            distances,
        }
    }

    /// Update centroids based on current assignment
    fn update_centroids(
        &self,
        vectors: &[Vec<f32>],
        assignment: &ClusterAssignment,
    ) -> Vec<Vec<f32>> {
        let mut new_centroids = vec![vec![0.0; self.dimension]; self.k];
        let mut counts = vec![0; self.k];

        // Sum up vectors in each cluster
        for (vector, &cluster_id) in vectors.iter().zip(assignment.assignments.iter()) {
            for (i, &val) in vector.iter().enumerate() {
                new_centroids[cluster_id][i] += val;
            }
            counts[cluster_id] += 1;
        }

        // Compute means
        for (centroid, count) in new_centroids.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                for val in centroid.iter_mut() {
                    *val /= *count as f32;
                }
            }
        }

        // Handle empty clusters by reinitializing with random vectors
        for (i, count) in counts.iter().enumerate() {
            if *count == 0 {
                let random_idx = rand::random::<usize>() % vectors.len();
                new_centroids[i] = vectors[random_idx].clone();
            }
        }

        new_centroids
    }

    /// Find the nearest centroid for a query vector
    pub fn nearest_centroid(&self, query: &[f32]) -> (usize, f32) {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, distance(query, c, self.metric)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }

    /// Get vectors grouped by cluster
    pub fn get_clusters(&self, vectors: &[Vec<f32>]) -> Vec<Vec<usize>> {
        let assignment = self.assign(vectors);
        let mut clusters = vec![Vec::new(); self.k];

        for (i, &cluster_id) in assignment.assignments.iter().enumerate() {
            clusters[cluster_id].push(i);
        }

        clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_simple() {
        // Create two clear clusters
        let mut vectors = Vec::new();
        
        // Cluster 1: around [0, 0]
        for _ in 0..10 {
            vectors.push(vec![
                rand::random::<f32>() * 0.5,
                rand::random::<f32>() * 0.5,
            ]);
        }
        
        // Cluster 2: around [10, 10]
        for _ in 0..10 {
            vectors.push(vec![
                10.0 + rand::random::<f32>() * 0.5,
                10.0 + rand::random::<f32>() * 0.5,
            ]);
        }

        let (kmeans, assignment) = KMeans::fit(&vectors, 2, DistanceMetric::L2, 10);

        assert_eq!(kmeans.k, 2);
        assert_eq!(assignment.assignments.len(), 20);

        // Check that vectors in same region are assigned to same cluster
        let first_cluster = assignment.assignments[0];
        for i in 1..10 {
            assert_eq!(assignment.assignments[i], first_cluster);
        }

        let second_cluster = assignment.assignments[10];
        assert_ne!(first_cluster, second_cluster); // Should be different clusters
        for i in 11..20 {
            assert_eq!(assignment.assignments[i], second_cluster);
        }
    }

    #[test]
    fn test_nearest_centroid() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
        ];

        let (kmeans, _) = KMeans::fit(&vectors, 2, DistanceMetric::L2, 10);

        // Query near [0, 0]
        let (nearest, _dist) = kmeans.nearest_centroid(&[0.5, 0.5]);
        
        // Should find centroid near [0.5, 0.5] region (vectors 0 and 1)
        let centroid = &kmeans.centroids[nearest];
        assert!(centroid[0] < 5.0 && centroid[1] < 5.0);
    }

    #[test]
    fn test_get_clusters() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![1.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let (kmeans, _) = KMeans::fit(&vectors, 2, DistanceMetric::L2, 10);
        let clusters = kmeans.get_clusters(&vectors);

        assert_eq!(clusters.len(), 2);
        
        // Each cluster should have 2 vectors
        assert_eq!(clusters[0].len() + clusters[1].len(), 4);
    }
}
