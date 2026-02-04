//! Vector quantization for compression and fast distance computation
//! 
//! Implements:
//! - Binary quantization (1 bit per dimension) for 32x compression
//! - RaBitQ (1 bit per dimension with unbiased estimator and theoretical guarantees)

pub mod binary;
pub mod rabitq;

pub use binary::{BinaryQuantizer, BinaryVector, hamming_distance};
pub use rabitq::{RaBitQQuantizer, RaBitQVector};