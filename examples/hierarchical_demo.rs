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
    println!("\n--- Building Adaptive Hierarchical Indices ---");
    let max_leaf_size = 150;
    
    for branching in [5, 10, 20] {
        let build_start = Instant::now();
        let vector_file = format!("hier_demo_branch_{}.bin", branching);
        let index = ClusteredIndex::build(
            vectors.clone(),
            &vector_file,
            branching,
            max_leaf_size,
            DistanceMetric::L2,
            10,
        ).expect("Failed to build index");
        let build_time = build_start.elapsed();
        
        println!("\nBranching factor {}:", branching);
        println!("  Build time: {:?}", build_time);
        println!("  Max depth: {}", index.max_depth());
        println!("  Total nodes: {}", index.num_nodes());
        
        // Quick search test
        let query = &queries[0];
        let results = index.search(query, k, 2, 3);
        let avg_dist = results.iter().map(|(_, d)| d).sum::<f32>() / results.len() as f32;
        println!("  Avg distance: {:.2}", avg_dist);
        println!();
        
        std::fs::remove_file(&vector_file).ok();
    }

    // Build final index for benchmarking
    println!("\n--- Performance Benchmark ---");
    let branching = 10;
    let build_start = Instant::now();
    let vector_file = "hier_demo_main.bin";
    let index = ClusteredIndex::build(
        vectors,
        vector_file,
        branching,
        max_leaf_size,
        DistanceMetric::L2,
        20,
    ).expect("Failed to build index");
    let build_time = build_start.elapsed();
    
    println!("Built index in {:?}", build_time);
    println!("Tree: max depth {}, {} nodes", index.max_depth(), index.num_nodes());

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
    println!("\n--- How Adaptive Hierarchical Clustering Works ---");
    println!("1. Adaptive Tree Structure:");
    println!("   - Root level: ~{} clusters", branching);
    println!("   - Each cluster splits into {} sub-clusters", branching);
    println!("   - Max depth: {}", index.max_depth());
    println!("   - Splits continue until leaf size ≤ {}", max_leaf_size);
    println!("   - Handles non-uniform data distributions");
    println!("\n2. Search Process:");
    println!("   - Start at root, find {} nearest clusters", 2);
    println!("   - Traverse down tree level-by-level");
    println!("   - At each level, pick {} nearest children", 2);
    println!("   - Continue until reaching leaf nodes");
    println!("   - At leaf level, collect binary candidates");
    println!("   - Rerank top candidates with full precision");
    println!("\n3. Binary Quantization:");
    println!("   - Each vector: 32x compression (f32 → 1 bit)");
    println!("   - Fast Hamming distance for filtering");
    println!("   - Full precision reranking for accuracy");
    println!("\n4. Performance:");
    println!("   - Search cost: O(branching × depth × probes)");
    println!("   - With 10,000 vectors:");
    println!("     * Flat search: 10,000 comparisons");
    println!("     * Hierarchical: ~{}  comparisons", branching * index.max_depth() * 2);
    println!("     * Speedup: ~{}x", 10000 / (branching * index.max_depth() * 2).max(1));
    
    println!("\n=== Summary ===");
    println!("The hierarchical index provides:");
    println!("  - Fast search through tree-based pruning");
    println!("  - Tunable accuracy/speed tradeoff (probes parameter)");
    println!("  - Scalability to large datasets");
    
    std::fs::remove_file(vector_file).ok();
}
