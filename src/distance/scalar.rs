//! Scalar (non-SIMD) distance functions - baseline for correctness testing

/// Euclidean (L2) distance squared between two vectors
/// 
/// We return squared distance to avoid the sqrt, which is expensive
/// and unnecessary for comparisons (sqrt is monotonic).
#[inline]
pub fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Dot product of two vectors
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Cosine similarity between two vectors
/// Returns value in [-1, 1] where 1 = identical direction
#[inline]
pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_scalar(a, b);
    let norm_a = dot_product_scalar(a, a).sqrt();
    let norm_b = dot_product_scalar(b, b).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

/// Cosine distance = 1 - cosine_similarity
/// Returns value in [0, 2] where 0 = identical direction
#[inline]
pub fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_squared_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(l2_squared_scalar(&v, &v), 0.0);
    }
    
    #[test]
    fn test_l2_squared_known() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 2.0];
        assert_eq!(l2_squared_scalar(&a, &b), 9.0); // 1 + 4 + 4
    }
    
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product_scalar(&a, &b), 32.0); // 4 + 10 + 18
    }
    
    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity_scalar(&v, &v) - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity_scalar(&a, &b).abs() < 1e-6);
    }
}
