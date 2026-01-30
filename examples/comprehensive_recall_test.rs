/// Comprehensive recall test with observability
/// 
/// Tests different configurations on 100K vectors to understand recall bottlenecks

use vectordb::{ClusteredIndex, DistanceMetric};
use rand::Rng;
use std::collections::HashSet;

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
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║        Comprehensive Recall Test with Observability              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    let num_vectors = 100_000;
    let dim = 128;
    let k = 10;
    let num_queries = 10;
    
    println!("Dataset: {} vectors, {} dimensions", num_vectors, dim);
    println!("Testing with {} queries, k={}\n", num_queries, k);
    
    let vectors = generate_random_vectors(num_vectors, dim);
    let query_indices: Vec<usize> = (0..num_queries)
        .map(|i| i * vectors.len() / num_queries)
        .collect();
    
    // Test configurations based on cluster balance analysis
    let configs = vec![
        (10, 100, "Balanced b10_l100"),
        (20, 150, "Balanced b20_l150"),
        (30, 200, "Balanced b30_l200"),
    ];
    
    for (branching, max_leaf, name) in configs {
        println!("\n{}", "═".repeat(70));
        println!("{} (branching={}, max_leaf={})", name, branching, max_leaf);
        println!("{}", "═".repeat(70));
        
        let index = ClusteredIndex::build(
            vectors.clone(),
            format!("recall_test_{}.bin", name.replace(" ", "_")),
            branching,
            max_leaf,
            DistanceMetric::L2,
            10,
        ).expect("Failed to build");
        
        // Print tree structure
        index.print_tree_structure(10);
        
        // Test with increasing probe counts
        println!("\n{:<10} {:<12} {:<12} {:<15} {:<15}", 
                 "Probes", "Recall@10", "Coverage", "Binary Scan", "Rerank");
        println!("{}", "-".repeat(70));
        
        for probes in [1, 2, 5, 10, 20, 50] {
            let mut total_recall = 0.0;
            let mut total_coverage = 0.0;
            let mut total_binary_pct = 0.0;
            let mut total_rerank_pct = 0.0;
            
            for &query_idx in &query_indices {
                let query = &vectors[query_idx];
                let gt = compute_ground_truth(&vectors, query, k);
                // Use high rerank_factor to avoid binary quantization filtering out true neighbors
                // With rerank=3, binary quantization is too lossy and we get low recall
                // With rerank=50-100, we get more accurate results
                let (results, stats) = index.search_with_stats(query, k, probes, 50);
                
                total_recall += calculate_recall(&gt, &results);
                total_coverage += stats.leaves_searched as f64 / stats.total_leaves as f64;
                total_binary_pct += stats.vectors_scanned_binary as f64 / stats.total_vectors as f64;
                total_rerank_pct += stats.vectors_reranked_full as f64 / stats.total_vectors as f64;
            }
            
            let avg_recall = total_recall / num_queries as f64;
            let avg_coverage = total_coverage / num_queries as f64;
            let avg_binary = total_binary_pct / num_queries as f64;
            let avg_rerank = total_rerank_pct / num_queries as f64;
            
            let marker = if avg_recall >= 0.8 {
                " ✓✓"
            } else if avg_recall >= 0.6 {
                " ✓"
            } else {
                ""
            };
            
            println!("{:<10} {:<12.1}% {:<12.2}% {:<15.2}% {:<15.3}%{}", 
                     probes,
                     avg_recall * 100.0,
                     avg_coverage * 100.0,
                     avg_binary * 100.0,
                     avg_rerank * 100.0,
                     marker);
            
            // Print detailed stats for first query at this probe level
            if probes == 10 {
                println!("\n  Example search statistics (probes={}, rerank=50):", probes);
                let query = &vectors[query_indices[0]];
                let (_results, stats) = index.search_with_stats(query, k, probes, 50);
                index.print_search_stats(&stats, probes);
            }
        }
        
        std::fs::remove_file(format!("recall_test_{}.bin", name.replace(" ", "_"))).ok();
    }
    
    println!("\n\n{}", "═".repeat(70));
}
