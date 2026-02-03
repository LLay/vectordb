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

/// Compute L2 squared distances from one query to 4 targets simultaneously
/// 
/// This processes 4 distance calculations in parallel, which is much faster than
/// calling l2_squared() 4 times. Returns distances in the same order as targets.
/// 
/// # Safety
/// Requires all slices to have the same length.
#[target_feature(enable = "neon")]
unsafe fn l2_squared_batch4_neon(
    query: &[f32],
    target0: &[f32],
    target1: &[f32],
    target2: &[f32],
    target3: &[f32],
) -> [f32; 4] {
    debug_assert_eq!(query.len(), target0.len());
    debug_assert_eq!(query.len(), target1.len());
    debug_assert_eq!(query.len(), target2.len());
    debug_assert_eq!(query.len(), target3.len());
    
    let dim = query.len();
    let mut i = 0;
    
    // 4 accumulators, one for each distance calculation
    let mut dist0 = vdupq_n_f32(0.0);
    let mut dist1 = vdupq_n_f32(0.0);
    let mut dist2 = vdupq_n_f32(0.0);
    let mut dist3 = vdupq_n_f32(0.0);
    
    // Process 4 elements at a time from each vector
    while i + 4 <= dim {
        // Load query vector (shared across all 4 distance calculations)
        let vq = vld1q_f32(query.as_ptr().add(i));
        
        // Load each target and compute differences
        let vt0 = vld1q_f32(target0.as_ptr().add(i));
        let diff0 = vsubq_f32(vq, vt0);
        dist0 = vfmaq_f32(dist0, diff0, diff0);
        
        let vt1 = vld1q_f32(target1.as_ptr().add(i));
        let diff1 = vsubq_f32(vq, vt1);
        dist1 = vfmaq_f32(dist1, diff1, diff1);
        
        let vt2 = vld1q_f32(target2.as_ptr().add(i));
        let diff2 = vsubq_f32(vq, vt2);
        dist2 = vfmaq_f32(dist2, diff2, diff2);
        
        let vt3 = vld1q_f32(target3.as_ptr().add(i));
        let diff3 = vsubq_f32(vq, vt3);
        dist3 = vfmaq_f32(dist3, diff3, diff3);
        
        i += 4;
    }
    
    // Horizontal sums
    let mut result0 = vaddvq_f32(dist0);
    let mut result1 = vaddvq_f32(dist1);
    let mut result2 = vaddvq_f32(dist2);
    let mut result3 = vaddvq_f32(dist3);
    
    // Handle remainder
    while i < dim {
        let q = query[i];
        let d0 = q - target0[i];
        result0 += d0 * d0;
        let d1 = q - target1[i];
        result1 += d1 * d1;
        let d2 = q - target2[i];
        result2 += d2 * d2;
        let d3 = q - target3[i];
        result3 += d3 * d3;
        i += 1;
    }
    
    [result0, result1, result2, result3]
}

/// Batch compute L2 squared distances from query to multiple targets
/// 
/// Processes targets in batches of 4 for optimal SIMD utilization.
/// This is significantly faster than calling l2_squared() in a loop.
pub fn l2_squared_batch(query: &[f32], targets: &[&[f32]]) -> Vec<f32> {
    if targets.is_empty() {
        return Vec::new();
    }
    
    let mut results = Vec::with_capacity(targets.len());
    let mut i = 0;
    
    // Process 4 targets at a time with SIMD
    while i + 4 <= targets.len() {
        let batch = unsafe {
            l2_squared_batch4_neon(query, targets[i], targets[i+1], targets[i+2], targets[i+3])
        };
        results.extend_from_slice(&batch);
        i += 4;
    }
    
    // Handle remaining targets (< 4) with regular SIMD
    while i < targets.len() {
        results.push(l2_squared(query, targets[i]));
        i += 1;
    }
    
    results
}

/// Compute dot products from one query to 4 targets simultaneously
#[target_feature(enable = "neon")]
unsafe fn dot_product_batch4_neon(
    query: &[f32],
    target0: &[f32],
    target1: &[f32],
    target2: &[f32],
    target3: &[f32],
) -> [f32; 4] {
    debug_assert_eq!(query.len(), target0.len());
    debug_assert_eq!(query.len(), target1.len());
    debug_assert_eq!(query.len(), target2.len());
    debug_assert_eq!(query.len(), target3.len());
    
    let dim = query.len();
    let mut i = 0;
    
    // 4 accumulators, one for each dot product
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    
    // Process 4 elements at a time
    while i + 4 <= dim {
        let vq = vld1q_f32(query.as_ptr().add(i));
        
        let vt0 = vld1q_f32(target0.as_ptr().add(i));
        sum0 = vfmaq_f32(sum0, vq, vt0);
        
        let vt1 = vld1q_f32(target1.as_ptr().add(i));
        sum1 = vfmaq_f32(sum1, vq, vt1);
        
        let vt2 = vld1q_f32(target2.as_ptr().add(i));
        sum2 = vfmaq_f32(sum2, vq, vt2);
        
        let vt3 = vld1q_f32(target3.as_ptr().add(i));
        sum3 = vfmaq_f32(sum3, vq, vt3);
        
        i += 4;
    }
    
    // Horizontal sums
    let mut result0 = vaddvq_f32(sum0);
    let mut result1 = vaddvq_f32(sum1);
    let mut result2 = vaddvq_f32(sum2);
    let mut result3 = vaddvq_f32(sum3);
    
    // Handle remainder
    while i < dim {
        let q = query[i];
        result0 += q * target0[i];
        result1 += q * target1[i];
        result2 += q * target2[i];
        result3 += q * target3[i];
        i += 1;
    }
    
    [result0, result1, result2, result3]
}

/// Batch compute dot products from query to multiple targets
pub fn dot_product_batch(query: &[f32], targets: &[&[f32]]) -> Vec<f32> {
    if targets.is_empty() {
        return Vec::new();
    }
    
    let mut results = Vec::with_capacity(targets.len());
    let mut i = 0;
    
    // Process 4 targets at a time with SIMD
    while i + 4 <= targets.len() {
        let batch = unsafe {
            dot_product_batch4_neon(query, targets[i], targets[i+1], targets[i+2], targets[i+3])
        };
        results.extend_from_slice(&batch);
        i += 4;
    }
    
    // Handle remaining targets
    while i < targets.len() {
        results.push(dot_product(query, targets[i]));
        i += 1;
    }
    
    results
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
    
    #[test]
    fn test_l2_batch() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let t0 = vec![1.0, 2.0, 3.0, 4.0];
        let t1 = vec![2.0, 3.0, 4.0, 5.0];
        let t2 = vec![0.0, 0.0, 0.0, 0.0];
        let t3 = vec![5.0, 5.0, 5.0, 5.0];
        
        let targets = vec![t0.as_slice(), t1.as_slice(), t2.as_slice(), t3.as_slice()];
        let batch_results = l2_squared_batch(&query, &targets);
        
        // Verify against individual calculations
        assert_eq!(batch_results.len(), 4);
        for i in 0..4 {
            let expected = l2_squared(&query, targets[i]);
            let relative_error = (batch_results[i] - expected).abs() / (expected.abs().max(1e-6));
            assert!(
                relative_error < 1e-5,
                "Batch result {} differs: expected={}, got={}, error={}",
                i,
                expected,
                batch_results[i],
                relative_error
            );
        }
        
        // Test with fewer than 4 targets
        let targets_3 = vec![t0.as_slice(), t1.as_slice(), t2.as_slice()];
        let batch_results_3 = l2_squared_batch(&query, &targets_3);
        assert_eq!(batch_results_3.len(), 3);
        
        // Test with more than 4 targets (to test batching logic)
        let t4 = vec![1.0, 1.0, 1.0, 1.0];
        let t5 = vec![2.0, 2.0, 2.0, 2.0];
        let targets_6 = vec![
            t0.as_slice(), t1.as_slice(), t2.as_slice(), 
            t3.as_slice(), t4.as_slice(), t5.as_slice()
        ];
        let batch_results_6 = l2_squared_batch(&query, &targets_6);
        assert_eq!(batch_results_6.len(), 6);
        for i in 0..6 {
            let expected = l2_squared(&query, targets_6[i]);
            let relative_error = (batch_results_6[i] - expected).abs() / (expected.abs().max(1e-6));
            assert!(
                relative_error < 1e-5,
                "Batch result {} (n=6) differs: expected={}, got={}",
                i,
                expected,
                batch_results_6[i]
            );
        }
    }
    
    #[test]
    fn test_dot_product_batch() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let t0 = vec![1.0, 0.0, 0.0, 0.0];
        let t1 = vec![0.0, 1.0, 0.0, 0.0];
        let t2 = vec![1.0, 1.0, 1.0, 1.0];
        let t3 = vec![2.0, 2.0, 2.0, 2.0];
        
        let targets = vec![t0.as_slice(), t1.as_slice(), t2.as_slice(), t3.as_slice()];
        let batch_results = dot_product_batch(&query, &targets);
        
        // Verify against individual calculations
        assert_eq!(batch_results.len(), 4);
        for i in 0..4 {
            let expected = dot_product(&query, targets[i]);
            let relative_error = (batch_results[i] - expected).abs() / (expected.abs().max(1e-6));
            assert!(
                relative_error < 1e-5,
                "Batch result {} differs: expected={}, got={}",
                i,
                expected,
                batch_results[i]
            );
        }
    }
}
