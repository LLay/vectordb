/// Utility module for loading vector datasets
/// 
/// This can be included in other examples to load pre-generated datasets

use std::fs::File;
use std::io::{BufReader, Read};

pub fn load_vectors(filename: &str) -> (Vec<Vec<f32>>, usize) {
    println!("Loading dataset from {}...", filename);
    
    let file = File::open(filename).expect(&format!("Failed to open {}", filename));
    let mut reader = BufReader::new(file);
    
    // Read header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).unwrap();
    
    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    println!("  → {} vectors, {} dimensions", num_vectors, dims);
    
    // Read vectors
    let mut vectors = Vec::with_capacity(num_vectors);
    let mut buffer = vec![0u8; dims * 4];
    
    for _ in 0..num_vectors {
        reader.read_exact(&mut buffer).unwrap();
        
        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        vectors.push(vector);
    }
    
    println!("  ✓ Loaded successfully\n");
    
    (vectors, dims)
}
