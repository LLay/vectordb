//! ARM NEON SIMD-optimized distance functions.
//! 
//! NEON provides 128-bit registers, allowing us to process 4 f32s at once.
//! Available on all Apple Silicon (M1/M2/M3) and modern ARM processors.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// L2 squared distance using NEON
/// 
/// Processes 4 floats per iteration using 128-bit registers.
/// Falls back to scalar for the remainder.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let n = a.len();
    let mut i = 0;
    
    // Accumulator for 4 partial sums
    let mut sum = vdupq_n_f32(0.0);
    
    // Process 4 elements at a time
    while i + 4 <= n {
        // Load 4 floats from each vector
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        
        // Compute difference
        let diff = vsubq_f32(va, vb);
        
        // Square and accumulate using FMA: sum = sum + diff * diff
        sum = vfmaq_f32(sum, diff, diff);
        
        i += 4;
    }
    
    // Horizontal sum of the 4 partial sums
    let mut result = vaddvq_f32(sum);
    
    // Handle remainder with scalar code
    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }
    
    result
}

/// Dot product using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let n = a.len();
    let mut i = 0;
    
    let mut sum = vdupq_n_f32(0.0);
    
    // Process 4 elements at a time
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        
        // FMA: sum = sum + va * vb
        sum = vfmaq_f32(sum, va, vb);
        
        i += 4;
    }
    
    // Horizontal sum
    let mut result = vaddvq_f32(sum);
    
    // Remainder
    while i < n {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

/// Optimized version that processes 16 elements per iteration
/// Uses 4 accumulators to hide FMA latency
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_product_neon_unrolled(a: &[f32], b: &[f32]) -> f32 {
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

/// Optimized L2 with loop unrolling
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l2_squared_neon_unrolled(a: &[f32], b: &[f32]) -> f32 {
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

// Safe wrappers that check for CPU support at runtime
// (On Apple Silicon, NEON is always available, but we keep the check for portability)

pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64, but we use is_aarch64_feature_detected
        // for consistency. On M1, this will always be true.
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { l2_squared_neon_unrolled(a, b) };
        }
    }
    
    // Fallback to scalar (shouldn't happen on M1)
    crate::distance::scalar::l2_squared_scalar(a, b)
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { dot_product_neon_unrolled(a, b) };
        }
    }
    
    crate::distance::scalar::dot_product_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::scalar;
    
    #[test]
    fn test_l2_matches_scalar() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1) + 0.5).collect();
        
        let scalar_result = scalar::l2_squared_scalar(&a, &b);
        let simd_result = l2_squared(&a, &b);
        
        let relative_error = (scalar_result - simd_result).abs() / scalar_result;
        assert!(
            relative_error < 1e-4,
            "Results differ: scalar={}, simd={}, error={}",
            scalar_result,
            simd_result,
            relative_error
        );
    }
    
    #[test]
    fn test_dot_matches_scalar() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001) + 0.1).collect();
        
        let scalar_result = scalar::dot_product_scalar(&a, &b);
        let simd_result = dot_product(&a, &b);
        
        let relative_error = (scalar_result - simd_result).abs() / scalar_result.abs();
        assert!(
            relative_error < 1e-5,
            "Results differ: scalar={}, simd={}, relative_error={}",
            scalar_result,
            simd_result,
            relative_error
        );
    }
    
    #[test]
    fn test_various_lengths() {
        // Test non-aligned lengths
        for len in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1023, 1024, 1025] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 2.0).collect();
            
            let scalar_result = scalar::dot_product_scalar(&a, &b);
            let simd_result = dot_product(&a, &b);
            
            let relative_error = if scalar_result.abs() > 1e-10 {
                (scalar_result - simd_result).abs() / scalar_result.abs()
            } else {
                (scalar_result - simd_result).abs()
            };
            
            assert!(
                relative_error < 1e-4,
                "Length {}: scalar={}, simd={}, error={}",
                len,
                scalar_result,
                simd_result,
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
