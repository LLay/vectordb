//! Distance calculation implementations
//! 
//! This module provides NEON SIMD-optimized vector distance functions
//! for Apple Silicon (M1/M2/M3).

pub mod scalar;
pub mod simd_neon;

/// Distance metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance
    L2,
    /// Cosine distance (1 - cosine_similarity)
    Cosine,
    /// Dot product (for normalized vectors, equivalent to cosine)
    DotProduct,
}

/// Compute distance between two vectors using the specified metric
/// 
/// Uses NEON SIMD optimizations.
#[inline]
pub fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::L2 => simd_neon::l2_squared(a, b),
        DistanceMetric::DotProduct => -simd_neon::dot_product(a, b), // Negate so smaller = more similar
        DistanceMetric::Cosine => {
            let dot = simd_neon::dot_product(a, b);
            let norm_a = simd_neon::dot_product(a, a).sqrt();
            let norm_b = simd_neon::dot_product(b, b).sqrt();
            1.0 - (dot / (norm_a * norm_b))
        }
    }
}

/// Batch distance computation - compute distances from query to many vectors
/// 
/// Returns vector of (index, distance) pairs, unsorted.
pub fn batch_distances(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, distance(query, v, metric)))
        .collect()
}

/// Parallel batch distance computation using rayon
pub fn batch_distances_parallel(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;
    
    vectors
        .par_iter()
        .enumerate()
        .map(|(i, v)| (i, distance(query, v, metric)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        // Just ensure they run without panicking
        let _ = distance(&a, &b, DistanceMetric::L2);
        let _ = distance(&a, &b, DistanceMetric::DotProduct);
        let _ = distance(&a, &b, DistanceMetric::Cosine);
    }
    
    #[test]
    fn test_batch_distances() {
        let query = vec![1.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ];
        
        let results = batch_distances(&query, &vectors, DistanceMetric::L2);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1, 0.0); // Identical to query
    }
}
