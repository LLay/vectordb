/// Generate and save various vector datasets with different distributions
/// 
/// This creates datasets that are more realistic than uniform random data,
/// which should help test recall performance with clustered/structured data.
///
/// Usage:
///   rustc generate_datasets.rs && ./generate_datasets          # Standard datasets
///   rustc generate_datasets.rs && ./generate_datasets --large  # Include 1M dataset

use rand::Rng;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          Vector Dataset Generator                             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let generate_large = args.iter().any(|arg| arg == "--large" || arg == "-l");
    
    if generate_large {
        println!("Generating ALL datasets including 1M vectors...\n");
    } else {
        println!("Generating standard datasets (10K-100K)");
        println!("Use --large flag to also generate 1M dataset\n");
    }
    
    // Generate different dataset types
    generate_gaussian_clusters("generated_by_me/data_10k_1024d_10clusters.bin", 10_000, 1024, 10);
    generate_gaussian_clusters("generated_by_me/data_10k_1024d_50clusters.bin", 10_000, 1024, 50);
    generate_gaussian_clusters("generated_by_me/data_10k_1024d_100clusters.bin", 10_000, 1024, 100);
    generate_gaussian_clusters("generated_by_me/data_100k_1024d_100clusters.bin", 100_000, 1024, 100);
    
    if generate_large {
        println!("\nGenerating large dataset (this will take a few minutes)...");
        generate_gaussian_clusters("generated_by_me/data_1m_1024d_1000clusters.bin", 1_000_000, 1024, 1000);
    }
    
    // Also generate a uniform random dataset for comparison
    generate_uniform_random("generated_by_me/data_10k_1024d_random.bin", 10_000, 1024);
    
    println!("\n✓ All datasets generated successfully!");
    println!("\nDatasets available:");
    println!("  - generated_by_me/data_10k_1024d_10clusters.bin   (10K vectors, 10 clusters, ~39 MB)");
    println!("  - generated_by_me/data_10k_1024d_50clusters.bin   (10K vectors, 50 clusters, ~39 MB)");
    println!("  - generated_by_me/data_10k_1024d_100clusters.bin  (10K vectors, 100 clusters, ~39 MB)");
    println!("  - generated_by_me/data_100k_1024d_100clusters.bin (100K vectors, 100 clusters, ~391 MB)");
    if generate_large {
        println!("  - generated_by_me/data_1m_1024d_1000clusters.bin  (1M vectors, 1000 clusters, ~3.9 GB)");
    }
    println!("  - generated_by_me/data_10k_1024d_random.bin       (10K vectors, uniform random, ~39 MB)");
}

/// Generate vectors with Gaussian clusters
/// Each cluster has a random center, and vectors are sampled from a Gaussian around that center
fn generate_gaussian_clusters(filename: &str, num_vectors: usize, dims: usize, num_clusters: usize) {
    println!("Generating {} ({} vectors, {} clusters)...", filename, num_vectors, num_clusters);
    
    let mut rng = rand::thread_rng();
    
    // Generate cluster centers
    let mut cluster_centers: Vec<Vec<f32>> = Vec::new();
    for _ in 0..num_clusters {
        let center: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
        cluster_centers.push(center);
    }
    
    let cluster_std_dev = 0.2; // Standard deviation within each cluster
    
    // For large datasets, write directly to file to save memory
    if num_vectors > 100_000 {
        generate_large_dataset_streaming(filename, num_vectors, dims, &cluster_centers, cluster_std_dev, &mut rng);
    } else {
        // For small datasets, generate in memory then write (faster)
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        
        for i in 0..num_vectors {
            let cluster_id = i % num_clusters;
            let center = &cluster_centers[cluster_id];
            
            let vector: Vec<f32> = center
                .iter()
                .map(|&c| {
                    let u1: f32 = rng.gen::<f32>().max(1e-10);
                    let u2: f32 = rng.gen();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    c + z * cluster_std_dev
                })
                .collect();
            
            vectors.push(vector);
        }
        
        save_vectors(filename, &vectors);
    }
    
    println!("  ✓ Generated {} vectors in {} clusters", num_vectors, num_clusters);
}

/// Generate large dataset by streaming directly to file (memory efficient)
fn generate_large_dataset_streaming(
    filename: &str,
    num_vectors: usize,
    dims: usize,
    cluster_centers: &[Vec<f32>],
    cluster_std_dev: f32,
    rng: &mut impl Rng,
) {
    let num_clusters = cluster_centers.len();
    let file = File::create(filename).expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    
    // Write header
    writer.write_all(&(num_vectors as u32).to_le_bytes()).unwrap();
    writer.write_all(&(dims as u32).to_le_bytes()).unwrap();
    
    // Write vectors in batches with progress updates
    let batch_size = 10_000;
    for batch_start in (0..num_vectors).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_vectors);
        
        for i in batch_start..batch_end {
            let cluster_id = i % num_clusters;
            let center = &cluster_centers[cluster_id];
            
            for &c in center {
                let u1: f32 = rng.gen::<f32>().max(1e-10);
                let u2: f32 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                let val = c + z * cluster_std_dev;
                writer.write_all(&val.to_le_bytes()).unwrap();
            }
        }
        
        // Progress update every 100K vectors
        if (batch_start / batch_size) % 10 == 0 && batch_start > 0 {
            println!("    Progress: {}/{} ({:.1}%)", batch_end, num_vectors, 
                     batch_end as f64 / num_vectors as f64 * 100.0);
        }
    }
    
    writer.flush().unwrap();
    
    let file_size_mb = (num_vectors * dims * 4 + 8) as f64 / (1024.0 * 1024.0);
    println!("    → Saved ({:.2} MB)", file_size_mb);
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
