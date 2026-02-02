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
    /// Internal k-means fitting implementation
    /// 
    /// Used by both `fit` and `fit_verbose` to ensure identical logic
    fn fit_internal(
        vectors: &[Vec<f32>],
        k: usize,
        metric: DistanceMetric,
        max_iterations: usize,
        verbose: bool,
    ) -> (Self, ClusterAssignment) {
        let start = std::time::Instant::now();
        
        if verbose {
            eprintln!("K-means clustering: {} vectors, k={}", vectors.len(), k);
            eprintln!("Using k-means++ initialization...");
        }
        
        let mut kmeans = Self::init_plusplus(vectors, k, metric);
        let mut assignment = kmeans.assign(vectors);
        let mut prev_assignments = assignment.assignments.clone();
        
        // Cap iterations - gains diminish quickly after ~10 iterations
        let effective_max_iters = max_iterations.min(12);

        for iteration in 0..effective_max_iters {
            let iter_start = if verbose { Some(std::time::Instant::now()) } else { None };
            
            // Update centroids
            let new_centroids = kmeans.update_centroids(vectors, &assignment);
            kmeans.centroids = new_centroids;

            // Reassign vectors
            assignment = kmeans.assign(vectors);
            
            // Check convergence based on cluster reassignments
            let changed = assignment.assignments
                .iter()
                .zip(prev_assignments.iter())
                .filter(|(a, b)| a != b)
                .count();
            
            let change_rate = changed as f32 / vectors.len() as f32;
            
            if verbose {
                let iter_time = iter_start.unwrap().elapsed();
                eprintln!("  Iteration {}: {:.2}% vectors reassigned, time={:.2}ms", 
                         iteration + 1, change_rate * 100.0, iter_time.as_secs_f64() * 1000.0);
            }

            // Early stopping: if < 2% of vectors changed clusters
            // Real-world data like SIFT rarely gets better after this point
            if change_rate < 0.02 {
                if verbose {
                    eprintln!("K-means converged after {} iterations ({:.2}s)", 
                             iteration + 1, start.elapsed().as_secs_f64());
                }
                break;
            }
            
            prev_assignments = assignment.assignments.clone();
        }
        
        if verbose && start.elapsed().as_secs_f64() > 1.0 {
            eprintln!("K-means total time: {:.2}s", start.elapsed().as_secs_f64());
        }

        (kmeans, assignment)
    }
    
    /// Run k-means clustering with verbose output
    /// 
    /// Same as `fit` but prints convergence information
    pub fn fit_verbose(
        vectors: &[Vec<f32>],
        k: usize,
        metric: DistanceMetric,
        max_iterations: usize,
    ) -> (Self, ClusterAssignment) {
        Self::fit_internal(vectors, k, metric, max_iterations, true)
    }

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
            // Compute distance from each vector to nearest centroid (parallel)
            let distances: Vec<f32> = vectors
                .par_iter()
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
        // Use KMEANS_DEBUG environment variable to enable debug output
        let verbose = std::env::var("KMEANS_DEBUG").is_ok();
        Self::fit_internal(vectors, k, metric, max_iterations, verbose)
    }

    /// Assign vectors to nearest centroids (parallel with optimized chunking)
    pub fn assign(&self, vectors: &[Vec<f32>]) -> ClusterAssignment {
        // Use larger chunks for better cache locality and reduced overhead
        // Chunk size of 256 vectors balances parallelism with cache efficiency
        let (assignments, distances): (Vec<_>, Vec<_>) = vectors
            .par_chunks(256)
            .flat_map(|chunk| {
                chunk.iter().map(|v| {
                    // Find nearest centroid
                    let (best_idx, best_dist) = self
                        .centroids
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, distance(v, c, self.metric)))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap();
                    (best_idx, best_dist)
                }).collect::<Vec<_>>()
            })
            .unzip();

        ClusterAssignment {
            assignments,
            distances,
        }
    }

    /// Update centroids based on current assignment (parallel with reduce)
    fn update_centroids(
        &self,
        vectors: &[Vec<f32>],
        assignment: &ClusterAssignment,
    ) -> Vec<Vec<f32>> {
        // Parallel reduction: each thread accumulates locally, then combine
        let (sum_centroids, sum_counts) = vectors
            .par_chunks(2048)  // Larger chunks for better cache locality
            .zip(assignment.assignments.par_chunks(2048))
            .map(|(vector_chunk, assignment_chunk)| {
                // Thread-local accumulators (no locks!)
                let mut local_centroids = vec![vec![0.0; self.dimension]; self.k];
                let mut local_counts = vec![0usize; self.k];
                
                // Accumulate in thread-local storage
                for (vector, &cluster_id) in vector_chunk.iter().zip(assignment_chunk.iter()) {
                    for (i, &val) in vector.iter().enumerate() {
                        local_centroids[cluster_id][i] += val;
                    }
                    local_counts[cluster_id] += 1;
                }
                
                (local_centroids, local_counts)
            })
            .reduce(
                || (vec![vec![0.0; self.dimension]; self.k], vec![0usize; self.k]),
                |mut acc, curr| {
                    // Merge two thread-local accumulators
                    for cluster_id in 0..self.k {
                        for i in 0..self.dimension {
                            acc.0[cluster_id][i] += curr.0[cluster_id][i];
                        }
                        acc.1[cluster_id] += curr.1[cluster_id];
                    }
                    acc
                },
            );

        let mut final_centroids = sum_centroids;
        let final_counts = sum_counts;

        // Compute means (parallel)
        final_centroids
            .par_iter_mut()
            .zip(final_counts.par_iter())
            .for_each(|(centroid, &count)| {
                if count > 0 {
                    let count_f32 = count as f32;
                    for val in centroid.iter_mut() {
                        *val /= count_f32;
                    }
                }
            });

        // Handle empty clusters by reinitializing with random vectors
        for (i, count) in final_counts.iter().enumerate() {
            if *count == 0 {
                let random_idx = rand::random::<usize>() % vectors.len();
                final_centroids[i] = vectors[random_idx].clone();
            }
        }

        final_centroids
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
