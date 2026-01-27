//! Index implementations for vector search
//! 
//! Provides k-means clustered index for fast similarity search.

pub mod clustered;

pub use clustered::ClusteredIndex;
