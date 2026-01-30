//! Binary quantization (1 bit per dimension)
//! 
//! Compresses f32 vectors to 1 bit per dimension:
//! - bit = 1 if value > threshold
//! - bit = 0 if value <= threshold
//! 
//! Achieves 32x compression (f32 = 32 bits → 1 bit)
//! Distance computation uses fast Hamming distance (popcount)

use std::arch::aarch64::*;

/// Binary quantized vector (1 bit per dimension)
/// 
/// Stores bits packed into u64s for efficient computation
#[derive(Debug, Clone)]
pub struct BinaryVector {
    /// Packed bits (64 dimensions per u64)
    pub bits: Vec<u64>,
    /// Original dimension count
    pub dimension: usize,
}

impl BinaryVector {
    /// Create a new binary vector from packed bits
    pub fn new(bits: Vec<u64>, dimension: usize) -> Self {
        assert!(bits.len() * 64 >= dimension, "Not enough bits for dimension");
        Self { bits, dimension }
    }

    /// Get the bit at a specific dimension
    pub fn get_bit(&self, idx: usize) -> bool {
        assert!(idx < self.dimension);
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        (self.bits[word_idx] >> bit_idx) & 1 == 1
    }

    /// Compute Hamming distance to another binary vector
    /// 
    /// Hamming distance = count of differing bits
    /// Uses NEON for fast popcount
    pub fn hamming_distance(&self, other: &BinaryVector) -> u32 {
        assert_eq!(self.dimension, other.dimension);
        
        let mut distance = 0u32;
        
        // XOR to find differing bits, then popcount
        for (a, b) in self.bits.iter().zip(other.bits.iter()) {
            let xor = a ^ b;
            distance += xor.count_ones();
        }
        
        distance
    }

    /// Fast Hamming distance using NEON vcnt (population count)
    #[target_feature(enable = "neon")]
    pub unsafe fn hamming_distance_neon(&self, other: &BinaryVector) -> u32 {
        assert_eq!(self.dimension, other.dimension);
        
        let mut total = 0u32;
        let len = self.bits.len();
        let mut i = 0;

        // Process 2 u64s at a time (128 bits = 1 NEON register)
        while i + 2 <= len {
            // Load two u64s as a 128-bit vector into a dedicated 128-bit NEON register.
            // Do this once for each of a and b
            let a = vld1q_u64(self.bits.as_ptr().add(i));
            let b = vld1q_u64(other.bits.as_ptr().add(i));
            
            // XOR to find differing bits
            let xor = veorq_u64(a, b);
            
            // Reinterpret as u8x16 for vcnt. No data movement - just tells the CPU to treat the bits differently
            let xor_u8 = vreinterpretq_u8_u64(xor);
            
            // Counts the number of 1 bits in each of the 16 bytes in parallel
            let popcnt = vcntq_u8(xor_u8);
            
            // Sum all bytes (horizontal sum)
            let sum = vaddlvq_u8(popcnt);
            
            total += sum as u32;
            i += 2;
        }

        // Handle remainder
        while i < len {
            total += (self.bits[i] ^ other.bits[i]).count_ones();
            i += 1;
        }

        total
    }
}

/// Binary quantizer
/// 
/// Quantizes f32 vectors to binary vectors
pub struct BinaryQuantizer {
    /// Dimension of vectors
    dimension: usize,
    /// Threshold for quantization (typically 0.0 or mean)
    threshold: f32,
}

impl BinaryQuantizer {
    /// Create a new binary quantizer
    /// 
    /// # Arguments
    /// * `dimension` - Vector dimension
    /// * `threshold` - Value threshold (bit=1 if value > threshold)
    pub fn new(dimension: usize, threshold: f32) -> Self {
        Self {
            dimension,
            threshold,
        }
    }

    /// Create quantizer with automatic threshold (mean of first batch)
    pub fn from_vectors(vectors: &[Vec<f32>]) -> Self {
        assert!(!vectors.is_empty());
        let dimension = vectors[0].len();
        
        // Compute mean across all values
        let total: f32 = vectors
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .sum();
        let count = vectors.len() * dimension;
        let threshold = total / count as f32;
        
        println!("Binary quantizer: threshold = {:.4}", threshold);
        
        Self {
            dimension,
            threshold,
        }
    }

    /// Quantize a vector to binary
    pub fn quantize(&self, vector: &[f32]) -> BinaryVector {
        assert_eq!(vector.len(), self.dimension);
        
        let num_words = (self.dimension + 63) / 64; // Ceiling division
        let mut bits = vec![0u64; num_words];
        
        for (i, &value) in vector.iter().enumerate() {
            if value > self.threshold {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                bits[word_idx] |= 1u64 << bit_idx;
            }
        }
        
        BinaryVector::new(bits, self.dimension)
    }

    /// Quantize multiple vectors
    pub fn quantize_batch(&self, vectors: &[Vec<f32>]) -> Vec<BinaryVector> {
        vectors.iter().map(|v| self.quantize(v)).collect()
    }

    /// Quantize in parallel using rayon
    pub fn quantize_batch_parallel(&self, vectors: &[Vec<f32>]) -> Vec<BinaryVector> {
        use rayon::prelude::*;
        vectors.par_iter().map(|v| self.quantize(v)).collect()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

/// Compute Hamming distance between two binary vectors (safe wrapper)
pub fn hamming_distance(a: &BinaryVector, b: &BinaryVector) -> u32 {
    unsafe { a.hamming_distance_neon(b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_quantization() {
        let quantizer = BinaryQuantizer::new(8, 0.0);
        
        let vector = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1];
        let binary = quantizer.quantize(&vector);
        
        assert_eq!(binary.dimension, 8);
        
        // Check bits: 1, 0, 1, 0, 1, 0, 1, 0
        assert!(binary.get_bit(0)); // 1.0 > 0
        assert!(!binary.get_bit(1)); // -1.0 <= 0
        assert!(binary.get_bit(2)); // 0.5 > 0
        assert!(!binary.get_bit(3)); // -0.5 <= 0
    }

    #[test]
    fn test_hamming_distance() {
        let quantizer = BinaryQuantizer::new(4, 0.0);
        
        let v1 = vec![1.0, 1.0, 1.0, 1.0]; // bits: 1111
        let v2 = vec![1.0, 1.0, -1.0, -1.0]; // bits: 1100
        
        let b1 = quantizer.quantize(&v1);
        let b2 = quantizer.quantize(&v2);
        
        let distance = hamming_distance(&b1, &b2);
        assert_eq!(distance, 2); // 2 bits differ
    }

    #[test]
    fn test_identical_vectors() {
        let quantizer = BinaryQuantizer::new(128, 0.0);
        
        let vector: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b1 = quantizer.quantize(&vector);
        let b2 = quantizer.quantize(&vector);
        
        let distance = hamming_distance(&b1, &b2);
        assert_eq!(distance, 0); // Identical
    }

    #[test]
    fn test_opposite_vectors() {
        let quantizer = BinaryQuantizer::new(64, 0.0);
        
        let v1: Vec<f32> = vec![1.0; 64]; // All positive
        let v2: Vec<f32> = vec![-1.0; 64]; // All negative
        
        let b1 = quantizer.quantize(&v1);
        let b2 = quantizer.quantize(&v2);
        
        let distance = hamming_distance(&b1, &b2);
        assert_eq!(distance, 64); // All bits differ
    }

    #[test]
    fn test_quantizer_from_vectors() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![-1.0, -2.0, -3.0],
            vec![0.0, 0.0, 0.0],
        ];
        
        let quantizer = BinaryQuantizer::from_vectors(&vectors);
        assert_eq!(quantizer.dimension, 3);
        // Mean is 0.0, so threshold should be 0.0
        assert!((quantizer.threshold - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_quantization() {
        let quantizer = BinaryQuantizer::new(4, 0.0);
        
        let vectors = vec![
            vec![1.0, -1.0, 1.0, -1.0],
            vec![-1.0, 1.0, -1.0, 1.0],
        ];
        
        let binaries = quantizer.quantize_batch(&vectors);
        assert_eq!(binaries.len(), 2);
        
        let distance = hamming_distance(&binaries[0], &binaries[1]);
        assert_eq!(distance, 4); // All bits differ
    }

    #[test]
    fn test_large_dimension() {
        let quantizer = BinaryQuantizer::new(1024, 0.0);
        
        let v1: Vec<f32> = (0..1024).map(|i| i as f32 - 512.0).collect();
        let v2: Vec<f32> = (0..1024).map(|i| i as f32 - 500.0).collect();
        
        let b1 = quantizer.quantize(&v1);
        let b2 = quantizer.quantize(&v2);
        
        let distance = hamming_distance(&b1, &b2);
        assert!(distance > 0); // Should differ
        assert!(distance < 1024); // But not all bits
    }
}
