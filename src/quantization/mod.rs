//! Vector quantization for compression and fast distance computation
//! 
//! Implements binary quantization (1 bit per dimension) for 32x compression.

pub mod binary;

pub use binary::{BinaryQuantizer, BinaryVector, hamming_distance};