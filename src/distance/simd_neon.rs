//! ARM NEON SIMD-optimized distance functions.
//! 
//! NEON provides 128-bit registers, allowing us to process 4 f32s at once.
//! Available on all Apple Silicon (M1/M2/M3).

use std::arch::aarch64::*;

/// Dot product using NEON with 4x loop unrolling
/// 
/// Uses 4 independent accumulators to hide FMA latency (4 cycles on M1).
/// Processes 16 floats per iteration.
#[target_feature(enable = "neon")]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let n = a.len();
    let mut i = 0;
    
    // 4 independent accumulators to utilize instruction-level parallelism
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    
    // Process 16 elements at a time (4 x 4)
    while i + 16 <= n {
        let va0 = vld1q_f32(a.as_ptr().add(i));
        let vb0 = vld1q_f32(b.as_ptr().add(i));
        sum0 = vfmaq_f32(sum0, va0, vb0);
        
        let va1 = vld1q_f32(a.as_ptr().add(i + 4));
        let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
        sum1 = vfmaq_f32(sum1, va1, vb1);
        
        let va2 = vld1q_f32(a.as_ptr().add(i + 8));
        let vb2 = vld1q_f32(b.as_ptr().add(i + 8));
        sum2 = vfmaq_f32(sum2, va2, vb2);
        
        let va3 = vld1q_f32(a.as_ptr().add(i + 12));
        let vb3 = vld1q_f32(b.as_ptr().add(i + 12));
        sum3 = vfmaq_f32(sum3, va3, vb3);
        
        i += 16;
    }
    
    // Combine accumulators
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    
    // Horizontal sum
    let mut result = vaddvq_f32(sum0);
    
    // Handle 4-element chunks
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let prod = vmulq_f32(va, vb);
        result += vaddvq_f32(prod);
        i += 4;
    }
    
    // Remainder
    while i < n {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

/// L2 squared distance using NEON with 4x loop unrolling
/// 
/// Uses 4 independent accumulators to hide FMA latency.
/// Processes 16 floats per iteration.
#[target_feature(enable = "neon")]
pub unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let n = a.len();
    let mut i = 0;
    
    // 4 independent accumulators
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    
    // Process 16 elements at a time (4 x 4)
    while i + 16 <= n {
        let va0 = vld1q_f32(a.as_ptr().add(i));
        let vb0 = vld1q_f32(b.as_ptr().add(i));
        let diff0 = vsubq_f32(va0, vb0);
        sum0 = vfmaq_f32(sum0, diff0, diff0);
        
        let va1 = vld1q_f32(a.as_ptr().add(i + 4));
        let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
        let diff1 = vsubq_f32(va1, vb1);
        sum1 = vfmaq_f32(sum1, diff1, diff1);
        
        let va2 = vld1q_f32(a.as_ptr().add(i + 8));
        let vb2 = vld1q_f32(b.as_ptr().add(i + 8));
        let diff2 = vsubq_f32(va2, vb2);
        sum2 = vfmaq_f32(sum2, diff2, diff2);
        
        let va3 = vld1q_f32(a.as_ptr().add(i + 12));
        let vb3 = vld1q_f32(b.as_ptr().add(i + 12));
        let diff3 = vsubq_f32(va3, vb3);
        sum3 = vfmaq_f32(sum3, diff3, diff3);
        
        i += 16;
    }
    
    // Combine accumulators
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    
    // Horizontal sum
    let mut result = vaddvq_f32(sum0);
    
    // Handle 4-element chunks
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        let sq = vmulq_f32(diff, diff);
        result += vaddvq_f32(sq);
        i += 4;
    }
    
    // Remainder
    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }
    
    result
}

/// Safe wrapper for L2 squared distance
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    unsafe { l2_squared_neon(a, b) }
}

/// Safe wrapper for dot product
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    unsafe { dot_product_neon(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_squared() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1) + 0.5).collect();
        
        let result = l2_squared(&a, &b);
        
        // Manually compute expected result for verification
        let expected: f32 = a.iter().zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        
        let relative_error = (expected - result).abs() / expected;
        assert!(
            relative_error < 1e-4,
            "Result incorrect: expected={}, got={}, error={}",
            expected,
            result,
            relative_error
        );
    }
    
    #[test]
    fn test_dot_product() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001) + 0.1).collect();
        
        let result = dot_product(&a, &b);
        
        // Manually compute expected result
        let expected: f32 = a.iter().zip(b.iter())
            .map(|(x, y)| x * y)
            .sum();
        
        let relative_error = (expected - result).abs() / expected.abs();
        assert!(
            relative_error < 1e-5,
            "Result incorrect: expected={}, got={}, relative_error={}",
            expected,
            result,
            relative_error
        );
    }
    
    #[test]
    fn test_various_lengths() {
        // Test non-aligned lengths
        for len in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1023, 1024, 1025] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 2.0).collect();
            
            let result = dot_product(&a, &b);
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            
            let relative_error = if expected.abs() > 1e-10 {
                (expected - result).abs() / expected.abs()
            } else {
                (expected - result).abs()
            };
            
            assert!(
                relative_error < 1e-4,
                "Length {}: expected={}, got={}, error={}",
                len,
                expected,
                result,
                relative_error
            );
        }
    }
    
    #[test]
    fn test_edge_cases() {
        // Empty vectors
        let empty: Vec<f32> = vec![];
        assert_eq!(l2_squared(&empty, &empty), 0.0);
        assert_eq!(dot_product(&empty, &empty), 0.0);
        
        // Single element
        let single_a = vec![2.0];
        let single_b = vec![3.0];
        assert_eq!(l2_squared(&single_a, &single_b), 1.0);
        assert_eq!(dot_product(&single_a, &single_b), 6.0);
        
        // Identical vectors
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(l2_squared(&v, &v), 0.0);
    }
}
