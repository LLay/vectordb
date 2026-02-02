/// Utilities for loading SIFT dataset in native .fvecs/.ivecs format
/// 
/// No conversion needed - we read the standard format directly.

use std::fs::File;
use std::io::{BufReader, Read, Write};

/// Read .fvecs format (SIFT vectors)
/// Format: [dim: i32][vector: f32*dim] repeated
pub fn read_fvecs(path: &str) -> std::io::Result<(Vec<Vec<f32>>, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    let mut dimension = 0;

    loop {
        // Read dimension (4 bytes)
        let mut dim_bytes = [0u8; 4];
        match reader.read_exact(&mut dim_bytes) {
            Ok(_) => {},
            Err(_) => break,  // EOF
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        
        if dimension == 0 {
            dimension = dim;
        }

        // Read vector (dim * 4 bytes)
        let mut vec_bytes = vec![0u8; dim * 4];
        reader.read_exact(&mut vec_bytes)?;
        
        let vector: Vec<f32> = vec_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        vectors.push(vector);

        // Progress indicator for large files
        if vectors.len() % 100_000 == 0 {
            print!("\r  Loading: {} vectors", vectors.len());
            std::io::stdout().flush()?;
        }
    }

    if vectors.len() > 100_000 {
        println!("\r  Loaded: {} vectors   ", vectors.len());
    }

    Ok((vectors, dimension))
}

/// Read .ivecs format (ground truth neighbor indices)
/// Format: [k: i32][neighbors: i32*k] repeated
pub fn read_ivecs(path: &str) -> std::io::Result<Vec<Vec<usize>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut ground_truth = Vec::new();

    loop {
        // Read k (number of neighbors)
        let mut k_bytes = [0u8; 4];
        match reader.read_exact(&mut k_bytes) {
            Ok(_) => {},
            Err(_) => break,  // EOF
        }
        let k = i32::from_le_bytes(k_bytes) as usize;

        // Read neighbor indices
        let mut indices_bytes = vec![0u8; k * 4];
        reader.read_exact(&mut indices_bytes)?;
        
        let indices: Vec<usize> = indices_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let idx = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                idx as usize
            })
            .collect();
        
        ground_truth.push(indices);
    }

    Ok(ground_truth)
}

#[cfg(test)]
mod tests {

    #[test]
    #[ignore] // Only run if SIFT dataset is downloaded
    fn test_load_sift_base() {
        let (vectors, dims) = super::read_fvecs("datasets/sift/data/sift_base.fvecs").unwrap();
        assert_eq!(vectors.len(), 1_000_000);
        assert_eq!(dims, 128);
        assert_eq!(vectors[0].len(), 128);
    }

    #[test]
    #[ignore] // Only run if SIFT dataset is downloaded
    fn test_load_sift_queries() {
        let (queries, dims) = super::read_fvecs("datasets/sift/data/sift_query.fvecs").unwrap();
        assert_eq!(queries.len(), 10_000);
        assert_eq!(dims, 128);
    }

    #[test]
    #[ignore] // Only run if SIFT dataset is downloaded
    fn test_load_ground_truth() {
        let ground_truth = super::read_ivecs("datasets/sift/data/sift_groundtruth.ivecs").unwrap();
        assert_eq!(ground_truth.len(), 10_000);
        assert_eq!(ground_truth[0].len(), 100); // 100-NN for each query
    }
}
