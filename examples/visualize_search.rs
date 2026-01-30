/// Visualize vector space and search behavior
/// 
/// Generates:
/// 1. vector_space.csv - 2D projection of vectors showing what was searched vs ground truth
/// 2. tree_structure.dot - Tree visualization showing search path vs ground truth location

use vectordb::{ClusteredIndex, DistanceMetric};
use vectordb::visualization::{visualize_vector_space, visualize_tree_structure, print_coverage_report};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};

fn load_vectors(filename: &str) -> (Vec<Vec<f32>>, usize) {
    let file = File::open(filename).expect("Failed to open file");
    let mut reader = BufReader::new(file);
    
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).unwrap();
    
    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
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
    
    (vectors, dims)
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
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          Vector Space & Search Visualization                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    // Load dataset
    let dataset_file = "datasets/data_10k_1024d_100clusters.bin";
    println!("Loading {}...", dataset_file);
    let (vectors, _dims) = load_vectors(dataset_file);
    println!("  ✓ Loaded {} vectors\n", vectors.len());
    
    // Build index
    println!("Building index...");
    let index = ClusteredIndex::build(
        vectors.clone(),
        "viz_index.bin",
        100,  // branching_factor
        100,  // target_leaf_size
        DistanceMetric::L2,
        10,   // max_iterations
    ).expect("Failed to build index");
    println!("  ✓ Built index with {} nodes\n", index.nodes.len());
    
    // Choose a query vector
    let query_idx = 42;
    let query = &vectors[query_idx];
    let k = 100;
    
    // Compute ground truth
    println!("Computing ground truth (k={})...", k);
    let ground_truth = compute_ground_truth(&vectors, query, k);
    println!("  ✓ Found {} nearest neighbors\n", ground_truth.len());
    
    // Perform search with stats
    println!("Performing search (probes=10)...");
    let probes = 10;
    let rerank_factor = 10;
    let (results, stats) = index.search_with_stats(query, k, probes, rerank_factor);
    let recall = calculate_recall(&ground_truth, &results);
    println!("  ✓ Search complete");
    println!("    Recall: {:.1}%", recall * 100.0);
    println!("    Leaves searched: {}", stats.leaves_searched);
    println!("    Vectors reranked: {}\n", stats.vectors_reranked_full);
    
    // Generate visualizations
    println!("Generating visualizations...\n");
    
    // 1. Coverage report (terminal output)
    print_coverage_report(&index, &ground_truth, &stats);
    
    // 2. Vector space visualization
    println!();
    visualize_vector_space(
        &index,
        &vectors,
        query,
        &ground_truth,
        &stats,
        "examples/visualization/vector_space.csv",
    ).expect("Failed to generate vector space visualization");
    
    // 3. Tree structure visualization
    println!();
    visualize_tree_structure(
        &index,
        &stats,
        &ground_truth,
        "examples/visualization/tree_structure.dot",
    ).expect("Failed to generate tree visualization");
    
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    Visualization Complete!                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!("\nFiles generated:");
    println!("  • examples/visualization/vector_space.csv");
    println!("  • examples/visualization/tree_structure.dot");
    println!("\nGenerate images:");
    println!("  cd examples/visualization && ./generate_visualizations.sh");
    println!("\nOr manually:");
    println!("  cd examples/visualization");
    println!("  python3 visualize.py");
    println!("  dot -Tpng tree_structure.dot -o tree_structure.png");
    
    // Clean up
    std::fs::remove_file("viz_index.bin").ok();
}
