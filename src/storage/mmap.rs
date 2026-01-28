//! Memory-mapped vector storage
//! 
//! Provides zero-copy access to vectors stored on disk using memory mapping.
//! The OS handles paging, so hot vectors stay in RAM while cold vectors
//! are loaded on demand.

use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter};
use std::path::Path;

/// Memory-mapped vector store with zero-copy access
pub struct MmapVectorStore {
    mmap: Mmap,
    dimension: usize,
    count: usize,
}

impl MmapVectorStore {
    /// Create a new memory-mapped vector store from a file
    /// 
    /// # Arguments
    /// * `path` - Path to the vector file
    /// * `dimension` - Vector dimensionality
    /// * `count` - Number of vectors in the file
    pub fn open<P: AsRef<Path>>(path: P, dimension: usize, count: usize) -> std::io::Result<Self> {
        let file = File::open(path)?;
        
        // Verify file size matches expected
        let expected_size = count * dimension * std::mem::size_of::<f32>();
        let metadata = file.metadata()?;
        
        if metadata.len() as usize != expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "File size mismatch: expected {} bytes, got {}",
                    expected_size,
                    metadata.len()
                ),
            ));
        }
        
        // Memory map the file
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        Ok(Self {
            mmap,
            dimension,
            count,
        })
    }
    
    /// Write vectors to a file and create a memory-mapped store
    /// 
    /// # Arguments
    /// * `path` - Path where vectors will be written
    /// * `vectors` - Vectors to write
    pub fn create<P: AsRef<Path>>(path: P, vectors: &[Vec<f32>]) -> std::io::Result<Self> {
        if vectors.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot create store from empty vectors",
            ));
        }
        
        let dimension = vectors[0].len();
        let count = vectors.len();
        
        // Write vectors to file
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        
        let mut writer = BufWriter::new(file);
        
        for vector in vectors {
            if vector.len() != dimension {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "All vectors must have the same dimension",
                ));
            }
            
            // Write as raw bytes (f32)
            for &value in vector {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        
        writer.flush()?;
        drop(writer);
        
        // Now open it as mmap
        Self::open(path, dimension, count)
    }
    
    /// Get a vector by index (zero-copy)
    /// 
    /// Returns a slice directly into the memory-mapped region.
    /// This is fast because no copying occurs - the slice points
    /// directly to memory (which may be backed by disk).
    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        assert!(idx < self.count, "Index out of bounds: {} >= {}", idx, self.count);
        
        let offset = idx * self.dimension;
        let ptr = self.mmap.as_ptr() as *const f32;
        
        unsafe {
            std::slice::from_raw_parts(ptr.add(offset), self.dimension)
        }
    }
    
    /// Get the number of vectors
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Check if store is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Get vector dimension
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    
    /// Get size in bytes
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.mmap.len()
    }
}

// MmapVectorStore is safe to send between threads
unsafe impl Send for MmapVectorStore {}
// MmapVectorStore is safe to share between threads (reads are safe)
unsafe impl Sync for MmapVectorStore {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_create_and_open() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        
        let path = "test_vectors.bin";
        
        // Create
        let store = MmapVectorStore::create(path, &vectors).unwrap();
        
        assert_eq!(store.len(), 3);
        assert_eq!(store.dimension(), 4);
        
        // Verify data
        let v0 = store.get(0);
        assert_eq!(v0, &[1.0, 2.0, 3.0, 4.0]);
        
        let v1 = store.get(1);
        assert_eq!(v1, &[5.0, 6.0, 7.0, 8.0]);
        
        let v2 = store.get(2);
        assert_eq!(v2, &[9.0, 10.0, 11.0, 12.0]);
        
        drop(store);
        
        // Re-open and verify
        let store2 = MmapVectorStore::open(path, 4, 3).unwrap();
        let v0_2 = store2.get(0);
        assert_eq!(v0_2, &[1.0, 2.0, 3.0, 4.0]);
        
        // Cleanup
        fs::remove_file(path).unwrap();
    }
    
    #[test]
    fn test_large_vectors() {
        let dim = 1024;
        let count = 100;
        
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f32).collect())
            .collect();
        
        let path = "test_large_vectors.bin";
        let store = MmapVectorStore::create(path, &vectors).unwrap();
        
        // Verify a few vectors
        for i in 0..count {
            let v = store.get(i);
            assert_eq!(v.len(), dim);
            assert_eq!(v[0], (i * dim) as f32);
            assert_eq!(v[dim - 1], (i * dim + dim - 1) as f32);
        }
        
        // Cleanup
        fs::remove_file(path).unwrap();
    }
    
    #[test]
    fn test_zero_copy() {
        let vectors = vec![vec![1.0, 2.0, 3.0]];
        let path = "test_zero_copy.bin";
        
        let store = MmapVectorStore::create(path, &vectors).unwrap();
        
        // Get same vector twice
        let v1 = store.get(0);
        let v2 = store.get(0);
        
        // They should point to the same memory (zero-copy)
        assert_eq!(v1.as_ptr(), v2.as_ptr());
        
        // Cleanup
        fs::remove_file(path).unwrap();
    }
    
    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_out_of_bounds() {
        let vectors = vec![vec![1.0, 2.0]];
        let path = "test_oob.bin";
        
        let store = MmapVectorStore::create(path, &vectors).unwrap();
        let _ = store.get(1); // Should panic
        
        fs::remove_file(path).unwrap();
    }
}
