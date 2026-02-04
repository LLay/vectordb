/// GIST dataset loader
/// 
/// GIST-960M is a standard benchmark dataset for high-dimensional vector search.
/// 960 dimensions, 1M base vectors, 1K queries
/// Format: .fvecs (float vectors) and .ivecs (int vectors)

use std::fs::File;
use std::io::{self, Read, BufReader};

/// Read vectors from .fvecs file (float vectors)
/// 
/// Format: [dim:i32][values:f32 x dim]... repeated
pub fn read_fvecs(path: &str) -> io::Result<(Vec<Vec<f32>>, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    let mut dimension = 0;
    
    // Progress reporting for large files
    let file_size = std::fs::metadata(path)?.len();
    let mut bytes_read = 0u64;
    let mut last_reported = 0u64;
    
    loop {
        // Read dimension
        let mut dim_bytes = [0u8; 4];
        match reader.read_exact(&mut dim_bytes) {
            Ok(_) => {},
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        bytes_read += 4;
        
        if dimension == 0 {
            dimension = dim;
        } else if dim != dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Inconsistent dimensions: expected {}, got {}", dimension, dim)
            ));
        }
        
        // Read vector values
        let mut vec = vec![0.0f32; dim];
        let mut bytes = vec![0u8; dim * 4];
        reader.read_exact(&mut bytes)?;
        bytes_read += (dim * 4) as u64;
        
        for i in 0..dim {
            let float_bytes = [
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ];
            vec[i] = f32::from_le_bytes(float_bytes);
        }
        
        vectors.push(vec);
        
        // Progress reporting every 10%
        let progress = (bytes_read * 100) / file_size;
        if progress >= last_reported + 10 {
            eprint!("  Loading: {} vectors  ", vectors.len());
            last_reported = progress;
        }
    }
    
    eprintln!("Loaded: {} vectors   ", vectors.len());
    
    Ok((vectors, dimension))
}

/// Read ground truth from .ivecs file (int vectors)
/// 
/// Format: [dim:i32][values:i32 x dim]... repeated
pub fn read_ivecs(path: &str) -> io::Result<Vec<Vec<usize>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    
    loop {
        // Read dimension
        let mut dim_bytes = [0u8; 4];
        match reader.read_exact(&mut dim_bytes) {
            Ok(_) => {},
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        
        // Read vector values
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut val_bytes = [0u8; 4];
            reader.read_exact(&mut val_bytes)?;
            let val = i32::from_le_bytes(val_bytes) as usize;
            vec.push(val);
        }
        
        vectors.push(vec);
    }
    
    Ok(vectors)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_load_gist_query() {
        let (vectors, dim) = read_fvecs("datasets/gist/data/gist/gist_query.fvecs")
            .expect("Failed to load GIST queries");
        
        assert_eq!(dim, 960, "GIST dimension should be 960");
        assert_eq!(vectors.len(), 1000, "GIST should have 1000 queries");
        assert_eq!(vectors[0].len(), 960);
    }
    
    #[test]
    fn test_load_gist_groundtruth() {
        let gt = read_ivecs("datasets/gist/data/gist/gist_groundtruth.ivecs")
            .expect("Failed to load GIST ground truth");
        
        assert_eq!(gt.len(), 1000, "Should have 1000 ground truth sets");
        assert_eq!(gt[0].len(), 100, "Each ground truth should have 100 neighbors");
    }
}
