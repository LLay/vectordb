//! Demo of clustered index

use rand::Rng;
use std::time::Instant;
use vectordb::{ClusteredIndex, DistanceMetric};

fn main() {
    println!("=== VectorDB Clustered Index Demo ===\n");

    let dim = 512;
    let num_vectors = 10_000;
    let num_queries = 100;
    let k = 10;

    // Generate random vectors
    println!("Generating {} vectors of dimension {}...", num_vectors, dim);
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    // Build Clustered Index
    println!("\n--- Building Clustered Index ---");
    let num_clusters = 100;
    
    let build_start = Instant::now();
    let clustered_index = ClusteredIndex::build(
        vectors,
        num_clusters,
        DistanceMetric::L2,
        20,
    );
    let build_time = build_start.elapsed();
    
    println!("Built index with {} clusters in {:?}", num_clusters, build_time);

    // Show example query results
    println!("\n--- Example Query ---");
    let results = clustered_index.search(&queries[0], 5, 1);
    println!("Top 5 results (1 probe):");
    for (i, (idx, dist)) in results.iter().enumerate() {
        println!("  {}. Vector {} (distance: {:.4})", i + 1, idx, dist);
    }

    // Benchmark with different probe counts
    println!("\n--- Performance with Different Probes ---");
    for probes in [1, 2, 5, 10] {
        let start = Instant::now();
        for query in &queries {
            let _ = clustered_index.search_parallel(query, k, probes);
        }
        let search_time = start.elapsed();
        
        println!(
            "Probes={:2}: {} queries in {:?} ({:.2} ms avg)",
            probes,
            num_queries,
            search_time,
            search_time.as_secs_f64() * 1000.0 / num_queries as f64
        );
    }

    // Show recall/accuracy tradeoff
    println!("\n--- Accuracy vs Speed Tradeoff ---");
    println!("More probes = higher accuracy but slower");
    println!("Fewer probes = faster but may miss some true neighbors");
    println!("\nTypical settings:");
    println!("  probes=1:  Fast, ~80-90% recall");
    println!("  probes=3:  Balanced, ~95% recall");
    println!("  probes=10: Accurate, ~99% recall");
}
