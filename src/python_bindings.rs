//! Python bindings for VectorDB using PyO3
//! 
//! This module exposes the Rust VectorDB implementation to Python for use with VectorDBBench.
//! 
//! Build with: maturin develop --release --features python
//! Import in Python: from vectordb import PyVectorDB

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::PyList;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::index::ClusteredIndexWithRaBitQ;
use crate::index::hierarchical_with_rabitq::SearchStatsWithRaBitQ;
use crate::distance::DistanceMetric;

/// Python wrapper for the Rust VectorDB implementation
/// 
/// This provides a high-performance vector database with hierarchical clustering
/// and RaBitQ quantization, accessible from Python.
#[pyclass(name = "PyVectorDB")]
pub struct PyVectorDB {
    /// The underlying Rust index (wrapped in Arc<Mutex<>> for thread safety)
    index: Arc<Mutex<Option<ClusteredIndexWithRaBitQ>>>,
    /// Vector dimensionality
    dimension: usize,
    /// Branching factor for hierarchical tree
    branching_factor: usize,
    /// Target leaf size
    target_leaf_size: usize,
    /// Distance metric
    metric: DistanceMetric,
    /// Accumulated vectors (for batch building)
    vectors: Arc<Mutex<Vec<Vec<f32>>>>,
    /// Temporary file path for mmap storage
    temp_file_path: String,
}

#[pymethods]
impl PyVectorDB {
    /// Create a new VectorDB instance
    /// 
    /// Args:
    ///     dimension: Vector dimensionality
    ///     branching_factor: Branching factor for hierarchical tree (default: 100)
    ///     target_leaf_size: Target vectors per leaf node (default: 100)
    ///     metric: Distance metric - "L2", "Cosine", or "InnerProduct" (default: "L2")
    ///     temp_file_path: Path for temporary mmap storage (default: "/tmp/vectordb_py.bin")
    #[new]
    #[pyo3(signature = (dimension, branching_factor=100, target_leaf_size=100, metric="L2", temp_file_path="/tmp/vectordb_py.bin"))]
    fn new(
        dimension: usize,
        branching_factor: usize,
        target_leaf_size: usize,
        metric: &str,
        temp_file_path: &str,
    ) -> PyResult<Self> {
        let metric = match metric {
            "L2" => DistanceMetric::L2,
            "Cosine" => DistanceMetric::Cosine,
            "InnerProduct" | "DotProduct" => DistanceMetric::DotProduct,
            _ => return Err(PyValueError::new_err(
                format!("Unknown metric '{}'. Use 'L2', 'Cosine', or 'InnerProduct'", metric)
            )),
        };

        Ok(PyVectorDB {
            index: Arc::new(Mutex::new(None)),
            dimension,
            branching_factor,
            target_leaf_size,
            metric,
            vectors: Arc::new(Mutex::new(Vec::new())),
            temp_file_path: temp_file_path.to_string(),
        })
    }

    /// Add vectors to the database (batched - doesn't build index yet)
    /// 
    /// Args:
    ///     vectors: List of vectors (each vector is a list of floats)
    /// 
    /// Returns:
    ///     Number of vectors added
    fn insert_embeddings(&mut self, _py: Python, vectors: &PyList) -> PyResult<usize> {
        // Convert Python list to Rust vectors (must hold GIL for this)
        let rust_vectors: Vec<Vec<f32>> = vectors.iter()
            .map(|item| item.extract::<Vec<f32>>())
            .collect::<Result<Vec<_>, _>>()?;

        // Validate dimensions
        for (i, vec) in rust_vectors.iter().enumerate() {
            if vec.len() != self.dimension {
                return Err(PyValueError::new_err(
                    format!("Vector {} has dimension {}, expected {}", i, vec.len(), self.dimension)
                ));
            }
        }

        let count = rust_vectors.len();
        {
            let mut vecs = self.vectors.lock().unwrap();
            vecs.extend(rust_vectors);
        }
        
        // Persist to staging file for multiprocessing support
        self.save_vectors_to_staging()?;

        Ok(count)
    }

    /// Build the index from accumulated vectors
    /// 
    /// This should be called after all vectors have been inserted.
    /// It constructs the hierarchical tree and quantizes vectors.
    /// 
    /// Returns:
    ///     Build time in seconds
    fn optimize(&mut self, py: Python) -> PyResult<f64> {
        let vectors = {
            let mut vecs = self.vectors.lock().unwrap();
            
            // If vectors are empty (e.g., after unpickling in a different process),
            // try loading from staging file
            if vecs.is_empty() {
                drop(vecs); // Release lock before loading
                let loaded = self.load_vectors_from_staging()?;
                if loaded.is_empty() {
                    return Err(PyValueError::new_err("No vectors to build index from"));
                }
                println!("Loaded {} vectors from staging file", loaded.len());
                loaded
            } else {
                std::mem::take(&mut *vecs)
            }
        };

        println!("Building index with {} vectors...", vectors.len());
        let start = Instant::now();

        // Build index (release GIL during heavy computation)
        let index = py.allow_threads(|| {
            ClusteredIndexWithRaBitQ::build(
                vectors,
                &self.temp_file_path,
                self.branching_factor,
                self.target_leaf_size,
                self.metric,
                20,  // max_iterations for k-means
            )
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to build index: {}", e)))?;

        let build_time = start.elapsed().as_secs_f64();
        println!("Index built in {:.2}s", build_time);

        *self.index.lock().unwrap() = Some(index);
        
        // Note: We don't clean up the staging file here because it may be needed
        // by other subprocesses in multiprocessing scenarios (e.g., VectorDBBench)
        // The staging file will be cleaned up when the temp directory is cleaned

        Ok(build_time)
    }

    /// Search for k nearest neighbors
    /// 
    /// Args:
    ///     query: Query vector (list of floats)
    ///     k: Number of neighbors to return (default: 10)
    ///     probes: Number of clusters to probe per level (default: 100)
    ///     rerank_factor: Rerank factor for two-phase search (default: 10)
    /// 
    /// Returns:
    ///     List of (vector_id, distance) tuples, sorted by distance
    #[pyo3(signature = (query, k=10, probes=100, rerank_factor=10))]
    fn search(
        &self,
        py: Python,
        query: Vec<f32>,
        k: usize,
        probes: usize,
        rerank_factor: usize,
    ) -> PyResult<Vec<(usize, f32)>> {
        if query.len() != self.dimension {
            return Err(PyValueError::new_err(
                format!("Query dimension {} doesn't match index dimension {}", 
                        query.len(), self.dimension)
            ));
        }

        let index = self.index.lock().unwrap();
        let index = index.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Index not built. Call optimize() first"))?;

        // Release GIL during search
        let results = py.allow_threads(|| {
            index.search(&query, k, probes, rerank_factor)
        });

        Ok(results)
    }

    /// Search with statistics tracking
    /// 
    /// Args:
    ///     query: Query vector (list of floats)
    ///     k: Number of neighbors to return
    ///     probes: Number of clusters to probe per level
    ///     rerank_factor: Rerank factor for two-phase search
    /// 
    /// Returns:
    ///     Tuple of (results, stats_dict) where:
    ///     - results: List of (vector_id, distance) tuples
    ///     - stats_dict: Dictionary with search statistics
    #[pyo3(signature = (query, k=10, probes=100, rerank_factor=10))]
    fn search_with_stats(
        &self,
        py: Python,
        query: Vec<f32>,
        k: usize,
        probes: usize,
        rerank_factor: usize,
    ) -> PyResult<(Vec<(usize, f32)>, PyObject)> {
        if query.len() != self.dimension {
            return Err(PyValueError::new_err(
                format!("Query dimension {} doesn't match index dimension {}", 
                        query.len(), self.dimension)
            ));
        }

        let index = self.index.lock().unwrap();
        let index = index.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Index not built. Call optimize() first"))?;

        // Release GIL during search
        let (results, stats) = py.allow_threads(|| {
            index.search_with_stats(&query, k, probes, rerank_factor)
        });

        // Convert stats to Python dict
        let stats_dict = pyo3::types::PyDict::new(py);
        stats_dict.set_item("total_leaves", stats.total_leaves)?;
        stats_dict.set_item("leaves_searched", stats.leaves_searched)?;
        stats_dict.set_item("total_vectors", stats.total_vectors)?;
        stats_dict.set_item("vectors_scanned_rabitq", stats.vectors_scanned_rabitq)?;
        stats_dict.set_item("vectors_reranked_full", stats.vectors_reranked_full)?;
        stats_dict.set_item("tree_depth", stats.tree_depth)?;

        Ok((results, stats_dict.into()))
    }

    /// Get index statistics
    /// 
    /// Returns:
    ///     Dictionary with index information
    fn get_stats(&self, py: Python) -> PyResult<PyObject> {
        let index = self.index.lock().unwrap();
        let index = index.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Index not built"))?;

        let stats = pyo3::types::PyDict::new(py);
        stats.set_item("num_vectors", index.len())?;
        stats.set_item("dimension", index.dimension())?;
        stats.set_item("max_depth", index.max_depth())?;
        stats.set_item("num_nodes", index.num_nodes())?;
        stats.set_item("num_leaves", index.count_leaves())?;
        stats.set_item("max_leaf_size", index.max_leaf_size())?;
        stats.set_item("memory_usage_bytes", index.memory_usage_bytes())?;
        stats.set_item("disk_usage_bytes", index.disk_usage_bytes())?;

        Ok(stats.into())
    }

    /// Get number of vectors in the index
    fn __len__(&self) -> PyResult<usize> {
        let index = self.index.lock().unwrap();
        match index.as_ref() {
            Some(idx) => Ok(idx.len()),
            None => {
                // Return accumulated vectors count if not built yet
                Ok(self.vectors.lock().unwrap().len())
            }
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        let index = self.index.lock().unwrap();
        match index.as_ref() {
            Some(idx) => format!(
                "PyVectorDB(vectors={}, dimension={}, depth={}, leaves={})",
                idx.len(),
                self.dimension,
                idx.max_depth(),
                idx.count_leaves()
            ),
            None => format!(
                "PyVectorDB(dimension={}, not_built, accumulated_vectors={})",
                self.dimension,
                self.vectors.lock().unwrap().len()
            ),
        }
    }
}

// Helper methods for persistence (multiprocessing support)
impl PyVectorDB {
    /// Save vectors to staging file for multiprocessing support
    /// This appends new vectors instead of rewriting the entire file
    fn save_vectors_to_staging(&self) -> PyResult<()> {
        use std::fs::{File, OpenOptions};
        use std::io::{Read, Write, Seek, SeekFrom};
        use std::path::Path;
        
        let staging_path = format!("{}.staging", self.temp_file_path);
        let vectors = self.vectors.lock().unwrap();
        
        if vectors.is_empty() {
            return Ok(());
        }
        
        let path = Path::new(&staging_path);
        let (mut file, vectors_written_before) = if path.exists() {
            // Open existing file for appending
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&staging_path)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to open staging file: {}", e)))?;
            
            // Read existing vector count
            let mut num_vectors_bytes = [0u8; 8];
            f.seek(SeekFrom::Start(0))
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to seek: {}", e)))?;
            f.read_exact(&mut num_vectors_bytes)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read header: {}", e)))?;
            let existing_count = u64::from_le_bytes(num_vectors_bytes) as usize;
            
            // Seek to end for appending
            f.seek(SeekFrom::End(0))
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to seek to end: {}", e)))?;
            
            (f, existing_count)
        } else {
            // Create new file and write header
            let mut f = File::create(&staging_path)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create staging file: {}", e)))?;
            
            // Write header: num_vectors, dimension (num_vectors will be updated later)
            let num_vectors = 0u64;
            let dimension = self.dimension as u64;
            f.write_all(&num_vectors.to_le_bytes())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to write header: {}", e)))?;
            f.write_all(&dimension.to_le_bytes())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to write header: {}", e)))?;
            
            (f, 0)
        };
        
        // Write only new vectors (those added since last save)
        for vec in vectors.iter().skip(vectors_written_before) {
            for &val in vec.iter() {
                file.write_all(&val.to_le_bytes())
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to write vector: {}", e)))?;
            }
        }
        
        // Update the vector count in the header
        let total_count = vectors.len() as u64;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to seek to header: {}", e)))?;
        file.write_all(&total_count.to_le_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update count: {}", e)))?;
        
        file.flush()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to flush staging file: {}", e)))?;
        
        Ok(())
    }
    
    /// Load vectors from staging file for multiprocessing support
    fn load_vectors_from_staging(&self) -> PyResult<Vec<Vec<f32>>> {
        use std::fs::File;
        use std::io::Read;
        use std::path::Path;
        
        let staging_path = format!("{}.staging", self.temp_file_path);
        
        if !Path::new(&staging_path).exists() {
            return Ok(Vec::new());
        }
        
        let mut file = File::open(&staging_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open staging file: {}", e)))?;
        
        // Read header
        let mut num_vectors_bytes = [0u8; 8];
        let mut dimension_bytes = [0u8; 8];
        file.read_exact(&mut num_vectors_bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read header: {}", e)))?;
        file.read_exact(&mut dimension_bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read header: {}", e)))?;
        
        let num_vectors = u64::from_le_bytes(num_vectors_bytes) as usize;
        let dimension = u64::from_le_bytes(dimension_bytes) as usize;
        
        if dimension != self.dimension {
            return Err(PyValueError::new_err(
                format!("Dimension mismatch: expected {}, got {}", self.dimension, dimension)
            ));
        }
        
        // Read vectors
        let mut vectors = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let mut vec = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                let mut val_bytes = [0u8; 4];
                file.read_exact(&mut val_bytes)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to read vector data: {}", e)))?;
                vec.push(f32::from_le_bytes(val_bytes));
            }
            vectors.push(vec);
        }
        
        Ok(vectors)
    }
}

/// Python module definition
#[pymodule]
fn vectordb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVectorDB>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
