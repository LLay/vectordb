//! Clustering algorithms for building vector indices
//! 
//! Provides implementations of:
//! - K-means clustering with k-means++ initialization

pub mod kmeans;

pub use kmeans::{KMeans, ClusterAssignment};
