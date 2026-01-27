//! Demo of hierarchical clustered index with binary quantization

use rand::Rng;
use std::time::Instant;
use vectordb::{ClusteredIndex, DistanceMetric};

fn main() {
    println!("=== Hierarchical Vector Database Demo ===\n");

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

    // Build hierarchical index with different branching factors
    println!("\n--- Building Hierarchical Indices ---");
    
    for branching in [5, 10, 20] {
        let build_start = Instant::now();
        let index = ClusteredIndex::build(
            vectors.clone(),
            branching,
            DistanceMetric::L2,
            20,
        );
        let build_time = build_start.elapsed();
        
        println!("\nBranching factor {}:", branching);
        println!("  Build time: {:?}", build_time);
        println!("  Levels: {}", index.num_levels());
        println!("  Total nodes: {}", index.num_nodes());
        
        // Show tree structure
        let avg_nodes_per_level = index.num_nodes() as f64 / index.num_levels() as f64;
        println!("  Avg nodes/level: {:.1}", avg_nodes_per_level);
    }

    // Build final index for benchmarking
    println!("\n--- Performance Benchmark ---");
    let branching = 10;
    let build_start = Instant::now();
    let index = ClusteredIndex::build(
        vectors,
        branching,
        DistanceMetric::L2,
        20,
    );
    let build_time = build_start.elapsed();
    
    println!("Built index in {:?}", build_time);
    println!("Tree: {} levels, {} nodes", index.num_levels(), index.num_nodes());

    // Show example query
    println!("\n--- Example Query ---");
    let query = &queries[0];
    let results = index.search(query, 5, 2, 3);
    println!("Top 5 results:");
    for (i, (idx, dist)) in results.iter().enumerate() {
        println!("  {}. Vector {} (distance: {:.4})", i + 1, idx, dist);
    }

    // Benchmark with different probe counts
    println!("\n--- Probes Per Level vs Performance ---");
    for probes in [1, 2, 5] {
        let start = Instant::now();
        for q in &queries {
            let _ = index.search(q, k, probes, 3);
        }
        let duration = start.elapsed();
        
        println!(
            "Probes={}: {} queries in {:?} ({:.2} ms avg)",
            probes,
            num_queries,
            duration,
            duration.as_secs_f64() * 1000.0 / num_queries as f64
        );
    }

    // Benchmark with different rerank factors
    println!("\n--- Rerank Factor vs Performance ---");
    for rerank in [2, 3, 5, 10] {
        let start = Instant::now();
        for q in &queries {
            let _ = index.search(q, k, 2, rerank);
        }
        let duration = start.elapsed();
        
        println!(
            "Rerank={}x: {} queries in {:?} ({:.2} ms avg)",
            rerank,
            num_queries,
            duration,
            duration.as_secs_f64() * 1000.0 / num_queries as f64
        );
    }

    // Show key insights
    println!("\n--- How Hierarchical Clustering Works ---");
    println!("1. Tree Structure:");
    println!("   - Root level: ~{} clusters", branching);
    println!("   - Each level splits into {} sub-clusters", branching);
    println!("   - Total levels: {}", index.num_levels());
    println!("   - Leaf level contains actual vectors");
    println!("\n2. Search Process:");
    println!("   - Start at root, find {} nearest clusters", 2);
    println!("   - Traverse down tree level-by-level");
    println!("   - At each level, pick {} nearest children", 2);
    println!("   - At leaf level, collect binary candidates");
    println!("   - Rerank top candidates with full precision");
    println!("\n3. Binary Quantization:");
    println!("   - Each vector: 32x compression (f32 → 1 bit)");
    println!("   - Fast Hamming distance for filtering");
    println!("   - Full precision reranking for accuracy");
    println!("\n4. Performance:");
    println!("   - Search cost: O(branching × levels × probes)");
    println!("   - With 10,000 vectors:");
    println!("     * Flat search: 10,000 comparisons");
    println!("     * Hierarchical: ~{}  comparisons", branching * index.num_levels() * 2);
    println!("     * Speedup: ~{}x", 10000 / (branching * index.num_levels() * 2));
}
