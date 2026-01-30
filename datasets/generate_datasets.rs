/// Generate and save various vector datasets with different distributions
/// 
/// This creates datasets that are more realistic than uniform random data,
/// which should help test recall performance with clustered/structured data.

use rand::Rng;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    println!("Generating vector datasets...\n");
    
    // Create datasets directory if it doesn't exist
    std::fs::create_dir_all("datasets").expect("Failed to create datasets directory");
    
    // Generate different dataset types
    generate_gaussian_clusters("datasets/data_10k_1024d_10clusters.bin", 10_000, 1024, 10);
    generate_gaussian_clusters("datasets/data_10k_1024d_50clusters.bin", 10_000, 1024, 50);
    generate_gaussian_clusters("datasets/data_10k_1024d_100clusters.bin", 10_000, 1024, 100);
    generate_gaussian_clusters("datasets/data_100k_1024d_100clusters.bin", 100_000, 1024, 100);
    
    // Also generate a uniform random dataset for comparison
    generate_uniform_random("datasets/data_10k_1024d_random.bin", 10_000, 1024);
    
    println!("\n✓ All datasets generated successfully!");
    println!("\nDatasets available in datasets/ folder:");
    println!("  - data_10k_1024d_10clusters.bin   (10K vectors, 10 natural clusters)");
    println!("  - data_10k_1024d_50clusters.bin   (10K vectors, 50 natural clusters)");
    println!("  - data_10k_1024d_100clusters.bin  (10K vectors, 100 natural clusters)");
    println!("  - data_100k_1024d_100clusters.bin (100K vectors, 100 natural clusters)");
    println!("  - data_10k_1024d_random.bin       (10K vectors, uniform random - baseline)");
}

/// Generate vectors with Gaussian clusters
/// Each cluster has a random center, and vectors are sampled from a Gaussian around that center
fn generate_gaussian_clusters(filename: &str, num_vectors: usize, dims: usize, num_clusters: usize) {
    println!("Generating {} with {} clusters...", filename, num_clusters);
    
    let mut rng = rand::thread_rng();
    
    // Generate cluster centers
    let mut cluster_centers: Vec<Vec<f32>> = Vec::new();
    for _ in 0..num_clusters {
        let center: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
        cluster_centers.push(center);
    }
    
    // Generate vectors around cluster centers
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let cluster_std_dev = 0.2; // Standard deviation within each cluster
    
    for i in 0..num_vectors {
        // Assign vector to a cluster
        let cluster_id = i % num_clusters;
        let center = &cluster_centers[cluster_id];
        
        // Sample from Gaussian around the cluster center
        let vector: Vec<f32> = center
            .iter()
            .map(|&c| {
                // Box-Muller transform for Gaussian sampling
                // Clamp u1 away from 0 to avoid ln(0) = -inf
                let u1: f32 = rng.gen::<f32>().max(1e-10);
                let u2: f32 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                c + z * cluster_std_dev
            })
            .collect();
        
        vectors.push(vector);
    }
    
    // Save to binary file
    save_vectors(filename, &vectors);
    
    println!("  ✓ Generated {} vectors in {} clusters", num_vectors, num_clusters);
}

/// Generate uniform random vectors (baseline for comparison)
fn generate_uniform_random(filename: &str, num_vectors: usize, dims: usize) {
    println!("Generating {} (uniform random)...", filename);
    
    let mut rng = rand::thread_rng();
    
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    
    save_vectors(filename, &vectors);
    
    println!("  ✓ Generated {} random vectors", num_vectors);
}

/// Save vectors to binary file
/// Format: [num_vectors: u32][dims: u32][vector1_data][vector2_data]...
fn save_vectors(filename: &str, vectors: &[Vec<f32>]) {
    let file = File::create(filename).expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    
    let num_vectors = vectors.len() as u32;
    let dims = vectors[0].len() as u32;
    
    // Write header
    writer.write_all(&num_vectors.to_le_bytes()).unwrap();
    writer.write_all(&dims.to_le_bytes()).unwrap();
    
    // Write vectors
    for vector in vectors {
        for &val in vector {
            writer.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    
    writer.flush().unwrap();
    
    let file_size_mb = (num_vectors as usize * dims as usize * 4 + 8) as f64 / (1024.0 * 1024.0);
    println!("  → Saved to {} ({:.2} MB)", filename, file_size_mb);
}
