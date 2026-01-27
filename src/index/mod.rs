//! Index implementations for vector search
//! 
//! Provides hierarchical clustered index with binary quantization.

pub mod hierarchical;

pub use hierarchical::ClusteredIndex;