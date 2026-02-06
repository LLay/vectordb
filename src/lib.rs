//! VectorDB - A high-performance vector database implementation
//! 
//! This library provides vector similarity search capabilities with
//! various distance metrics and indexing strategies.

pub mod distance;
pub mod index;
pub mod clustering;
pub mod quantization;
pub mod storage;
pub mod visualization;

// Python bindings (only compiled with python feature)
#[cfg(feature = "python")]
pub mod python_bindings;

// Re-export commonly used types
pub use distance::{DistanceMetric, distance, batch_distances, batch_distances_parallel};
pub use index::{ClusteredIndex, ClusteredIndexWithRaBitQ};
pub use clustering::{KMeans, ClusterAssignment};
pub use quantization::{BinaryQuantizer, BinaryVector, RaBitQQuantizer, RaBitQVector};