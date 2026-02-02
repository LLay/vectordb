/// Quick tool to inspect generated datasets for issues

use std::fs::File;
use std::io::{BufReader, Read};

fn main() {
    let filename = "datasets/generated_by_me/data_10k_1024d_100clusters.bin";
    
    let file = File::open(filename).expect("Failed to open file");
    let mut reader = BufReader::new(file);
    
    // Read header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).unwrap();
    
    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    println!("Dataset: {} vectors, {} dims", num_vectors, dims);
    
    // Read first few vectors
    let mut buffer = vec![0u8; dims * 4];
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut sum = 0.0f32;
    let mut count = 0;
    
    for i in 0..num_vectors.min(10) {
        reader.read_exact(&mut buffer).unwrap();
        
        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        for &val in &vector {
            if val.is_nan() {
                nan_count += 1;
            }
            if val.is_infinite() {
                inf_count += 1;
            }
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            sum += val;
            count += 1;
        }
        
        if i < 3 {
            println!("\nVector {}: {:?}...", i, &vector[0..5.min(vector.len())]);
        }
    }
    
    println!("\nStatistics (first 10 vectors):");
    println!("  Min: {}", min_val);
    println!("  Max: {}", max_val);
    println!("  Mean: {}", sum / count as f32);
    println!("  NaN count: {}", nan_count);
    println!("  Inf count: {}", inf_count);
}
