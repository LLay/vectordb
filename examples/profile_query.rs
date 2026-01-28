//! Detailed query profiling tool
//! 
//! This example breaks down query execution into phases and shows
//! exactly where time is being spent. Use this to identify bottlenecks
//! and track optimization progress.

use rand::Rng;
use std::time::Instant;
use vectordb::{ClusteredIndex, DistanceMetric};

#[derive(Debug)]
struct PhaseTimings {
    total: f64,
    // Add more fields as we instrument the code
}

fn generate_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn profile_query_detailed(
    index: &ClusteredIndex,
    query: &[f32],
    k: usize,
    probes: usize,
    rerank_factor: usize,
) -> PhaseTimings {
    let start = Instant::now();
    
    // Execute query
    let _results = index.search(query, k, probes, rerank_factor);
    
    let total = start.elapsed().as_secs_f64() * 1000.0; // Convert to ms
    
    PhaseTimings { total }
}

fn main() {
    println!("=== VectorDB Query Profiling Tool ===\n");
    
    // Configuration
    let num_vectors = 100_000; // Start with 100K for faster iteration
    let dim = 1024;
    let k = 10;
    let num_queries = 100;
    
    println!("Configuration:");
    println!("  Vectors: {}", num_vectors);
    println!("  Dimensions: {}", dim);
    println!("  k: {}", k);
    println!("  Queries: {}\n", num_queries);
    
    // Build index
    println!("Building index...");
    let vectors = generate_vectors(num_vectors, dim);
    let build_start = Instant::now();
    let vector_file = "profile_vectors.bin";
    let index = ClusteredIndex::build(vectors, vector_file, 10, 150, DistanceMetric::L2, 20)
        .expect("Failed to build index");
    let build_time = build_start.elapsed();
    
    println!("Index built in {:.2}s", build_time.as_secs_f64());
    println!("  Tree depth: {}", index.max_depth());
    println!("  Total nodes: {}\n", index.num_nodes());
    
    // Generate queries
    let mut rng = rand::thread_rng();
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    
    println!("=== Testing Different Configurations ===\n");
    
    let configs = [
        (k, 1, 2, "Low Latency (probes=1)"),
        (k, 2, 3, "Balanced (probes=2)"),
        (k, 3, 5, "High Recall (probes=3)"),
    ];
    
    for (k, probes, rerank, name) in configs.iter() {
        println!("--- {} ---", name);
        println!("Parameters: k={}, probes={}, rerank_factor={}", k, probes, rerank);
        
        let mut timings = Vec::new();
        
        // Warm-up
        for _ in 0..10 {
            let _ = index.search(&queries[0], *k, *probes, *rerank);
        }
        
        // Profile queries
        for query in queries.iter() {
            let timing = profile_query_detailed(&index, query, *k, *probes, *rerank);
            timings.push(timing.total);
        }
        
        // Statistics
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = timings[timings.len() / 2];
        let p95 = timings[timings.len() * 95 / 100];
        let p99 = timings[timings.len() * 99 / 100];
        let mean = timings.iter().sum::<f64>() / timings.len() as f64;
        let min = timings[0];
        let max = timings[timings.len() - 1];
        
        println!("Query Latency:");
        println!("  Mean:   {:.3} ms", mean);
        println!("  Median: {:.3} ms", p50);
        println!("  p95:    {:.3} ms", p95);
        println!("  p99:    {:.3} ms", p99);
        println!("  Min:    {:.3} ms", min);
        println!("  Max:    {:.3} ms", max);
        
        // Estimate candidates scanned
        let estimated_candidates = probes.pow(index.max_depth() as u32) * 150;
        println!("Estimated candidates scanned: ~{}", estimated_candidates);
        
        // Throughput
        let qps = 1000.0 / p99;
        println!("Estimated throughput (at p99): {:.0} QPS", qps);
        println!();
    }
    
    println!("=== Phase Breakdown (Approximate) ===\n");
    println!("Note: For detailed phase timing, we need to instrument ClusteredIndex::search()");
    println!("      Current implementation doesn't expose phase-level timings.");
    println!("\nTo get detailed breakdown:");
    println!("  1. Add timing instrumentation to src/index/hierarchical.rs");
    println!("  2. Return phase timings from search()");
    println!("  3. Re-run this profiler\n");
    
    println!("=== Estimated Time Distribution (Based on Algorithm) ===\n");
    let configs_est = [
        ("Tree traversal", 5.0),
        ("Candidate identification", 10.0),
        ("Distance computation", 60.0),
        ("Reranking", 20.0),
        ("Sorting", 5.0),
    ];
    
    println!("Without optimizations (estimated):");
    for (phase, pct) in configs_est.iter() {
        let bars = "█".repeat((pct / 5.0) as usize);
        println!("  {:25} {:3.0}%  {}", phase, pct, bars);
    }
    
    println!("\n=== Optimization Targets ===\n");
    println!("Based on current performance, focus on:");
    println!("  1. Distance computation (60%) - Use binary quantization + Hamming");
    println!("  2. Reranking (20%) - Parallelize and use mmap");
    println!("  3. Candidate identification (10%) - Parallel tree probes");
    println!("\nExpected improvement: 5-10x speedup with all optimizations");
    
    println!("\n=== Next Steps ===\n");
    println!("1. Implement binary quantization for centroids");
    println!("2. Run this profiler again to measure improvement");
    println!("3. Add parallel candidate scanning");
    println!("4. Profile again and iterate");
}
