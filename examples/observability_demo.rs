/// Demonstration of index observability features
/// 
/// Shows tree structure, leaf distribution, and search statistics

use vectordb::{ClusteredIndex, DistanceMetric};
use vectordb::visualization::{visualize_vector_space, visualize_tree_structure, print_coverage_report};
use rand::Rng;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn load_vectors_from_file(filename: &str) -> Option<(Vec<Vec<f32>>, usize)> {
    if !Path::new(filename).exists() {
        return None;
    }
    
    println!("Loading dataset from {}...", filename);
    
    let file = File::open(filename).ok()?;
    let mut reader = BufReader::new(file);
    
    // Read header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).ok()?;
    
    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    println!("  → {} vectors, {} dimensions", num_vectors, dims);
    
    // Read vectors
    let mut vectors = Vec::with_capacity(num_vectors);
    let mut buffer = vec![0u8; dims * 4];
    
    for _ in 0..num_vectors {
        reader.read_exact(&mut buffer).ok()?;
        
        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        vectors.push(vector);
    }
    
    println!("  ✓ Loaded successfully\n");
    
    Some((vectors, dims))
}

fn generate_random_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn compute_ground_truth(
    vectors: &[Vec<f32>],
    query: &[f32],
    k: usize,
) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, vec)| {
            let dist: f32 = vec
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (idx, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.iter().take(k).map(|(idx, _)| *idx).collect()
}

fn calculate_recall(ground_truth: &[usize], results: &[(usize, f32)]) -> f64 {
    let gt_set: HashSet<usize> = ground_truth.iter().copied().collect();
    let found: usize = results.iter().filter(|(idx, _)| gt_set.contains(idx)).count();
    found as f64 / ground_truth.len() as f64
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════╗");
    println!("║            Index Observability Demonstration                    ║");
    println!("╚═════════════════════════════════════════════════════════════════╝\n");
    
    // Try to load pre-generated dataset, fall back to random generation
    let dataset_file = "datasets/data_100k_1024d_100clusters.bin";
    let (vectors, dim) = if let Some((vecs, d)) = load_vectors_from_file(dataset_file) {
        (vecs, d)
    } else {
        println!("Dataset file '{}' not found. Generating random vectors...", dataset_file);
        println!("(Run 'cargo run --release --example generate_datasets' to create datasets)\n");
        let num_vectors = 10_000;
        let dim = 1024;
        (generate_random_vectors(num_vectors, dim), dim)
    };
    
    let num_vectors = vectors.len();
    let k = 1000;
    
    println!("Dataset: {} vectors, {} dimensions\n", num_vectors, dim);
    
    // Test different configurations
    let configs = vec![
        // (10, 500, "Small branching"),
        // (20, 1000, "Medium branching"),
        // (30, 1500, "Large branching"),
        (100, 100, "turbopuffer leaves"),
    ];
    
    for (branching, target_leaf_size, name) in configs {
        println!("\n{}", "═".repeat(70));
        println!("Configuration: {} (branching={}, target_leaf_size={})", name, branching, target_leaf_size);
        println!("{}", "═".repeat(70));
        
        // Build index with observability
        let index = ClusteredIndex::build(
            vectors.clone(),
            format!("obs_demo_{}.bin", name.replace(" ", "_")),
            branching,
            target_leaf_size,
            DistanceMetric::L2,
            10,
        ).expect("Failed to build");
        
        // Print tree structure
        index.print_tree_structure(100);
        
        // Test search with different probe counts
        let query = &vectors[42]; // Use an in-dataset query
        let gt = compute_ground_truth(&vectors, query, k);
        
        println!("\n--- Search Performance ---\n");
        
        // Store the last stats for visualization
        let mut last_stats = None;
        
        for probes in [10] {
            println!("Probes = {}:", probes);
            println!("{}", "-".repeat(50));
            
            // Test with different rerank factors to show the impact
            for rerank_factor in [3] {
                let (results, stats) = index.search_with_stats(query, k, probes, rerank_factor);
                let recall = calculate_recall(&gt, &results);
                
                println!("  rerank_factor={}: Recall@{}={:.1}%, reranked={}, returned={}", 
                         rerank_factor, k, recall * 100.0, stats.vectors_reranked_full, results.len());
                
                index.print_search_stats(&stats, probes);
                
                // Save stats for visualization
                last_stats = Some(stats);
            }
            println!();
        }
        
        // Generate visualizations for this configuration
        if let Some(stats) = last_stats {
            println!("\n--- Generating Visualizations ---\n");
            
            // Generate coverage report
            print_coverage_report(&index, &gt, &stats);
            
            // Generate vector space visualization
            println!();
            visualize_vector_space(
                &index,
                &vectors,
                query,
                &gt,
                &stats,
                &format!("examples/visualization/vector_space_{}.csv", name.replace(" ", "_")),
            ).expect("Failed to generate vector space visualization");
            
            // Generate tree structure visualization
            println!();
            visualize_tree_structure(
                &index,
                &stats,
                &gt,
                &format!("examples/visualization/tree_structure_{}.dot", name.replace(" ", "_")),
            ).expect("Failed to generate tree visualization");
        }
        
        // Clean up
        std::fs::remove_file(format!("obs_demo_{}.bin", name.replace(" ", "_"))).ok();
    }
    
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║              Visualization Files Generated                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    // println!("\nRun these commands to generate images:");
    // println!("  python3 visualize.py  # Generates vector_space.png");
    // println!("  dot -Tpng tree_structure_turbopuffer_leaves.dot -o tree_structure.png");

    // Run visualization generation script
    std::process::Command::new("./examples/visualization/generate_visualizations.sh")
        .output()
        .expect("Failed to run generate_visualizations.sh");
    println!("Generated visualizations");
    println!("  examples/visualization/vector_space.png");
    println!("  examples/visualization/tree_structure.png");
}
