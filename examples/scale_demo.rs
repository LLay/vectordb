//! Scale demonstration: Testing performance at various vector counts
//! 
//! This example shows how latency scales from 1M to 50M vectors
//! and helps you understand the practical limits of your hardware.

use rand::Rng;
use std::time::Instant;
use vectordb::{ClusteredIndex, BinaryQuantizer, DistanceMetric};

fn generate_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    println!("Generating {} vectors of dimension {}...", num, dim);
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn measure_latency(
    index: &ClusteredIndex,
    queries: &[Vec<f32>],
    k: usize,
    probes: usize,
    rerank: usize,
) -> (f64, f64, f64) {
    let mut latencies = Vec::with_capacity(queries.len());
    
    for query in queries {
        let start = Instant::now();
        let _ = index.search(query, k, probes, rerank);
        latencies.push(start.elapsed().as_secs_f64() * 1000.0); // Convert to ms
    }
    
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let median = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() * 99) / 100];
    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    
    (mean, median, p99)
}

fn test_scale(num_vectors: usize, dim: usize) {
    println!("\n{}", "=".repeat(80));
    println!("TESTING: {} vectors, {} dimensions", num_vectors, dim);
    println!("{}", "=".repeat(80));
    
    // Generate data
    let vectors = generate_vectors(num_vectors, dim);
    
    // Memory estimate
    let raw_size_gb = (num_vectors * dim * 4) as f64 / 1_073_741_824.0;
    let quantized_size_gb = (num_vectors * dim / 8) as f64 / 1_073_741_824.0;
    println!("\nStorage:");
    println!("  Raw (f32): {:.2} GB", raw_size_gb);
    println!("  Quantized (binary): {:.2} GB", quantized_size_gb);
    println!("  Compression: {:.1}x", raw_size_gb / quantized_size_gb);
    
    // Build index
    println!("\nBuilding index...");
    let branching = 10;
    let max_leaf = 150;
    let max_iters = 20;
    
    let build_start = Instant::now();
    let index = ClusteredIndex::build(
        vectors.clone(),
        branching,
        max_leaf,
        DistanceMetric::L2,
        max_iters,
    );
    let build_time = build_start.elapsed();
    
    println!("  Build time: {:.2}s", build_time.as_secs_f64());
    
    // Estimate index memory
    let centroid_memory_mb = (index.num_nodes() * dim * 4) as f64 / 1_048_576.0;
    println!("  Index RAM: {:.2} MB", centroid_memory_mb);
    
    println!("\nQuerying...");
    // Generate test queries
    let num_queries = 100;
    let queries = generate_vectors(num_queries, dim);
    
    // Test different configurations
    let configs = [
        (10, 1, 2, "Low latency"),
        (10, 2, 3, "Balanced"),
        (10, 3, 5, "High recall"),
    ];
    
    println!("\n{:<20} {:>8} {:>8} {:>8} {:>8}", "Config", "k", "probes", "rerank", "");
    println!("{}", "-".repeat(60));
    
    for (k, probes, rerank, name) in configs.iter() {
        // Warm-up
        for _ in 0..5 {
            let _ = index.search(&queries[0], *k, *probes, *rerank);
        }
        
        // Measure
        let (mean, median, p99) = measure_latency(&index, &queries, *k, *probes, *rerank);
        
        println!(
            "{:<20} {:>8} {:>8} {:>8}",
            name, k, probes, rerank
        );
        println!("  Mean: {:.3}ms, Median: {:.3}ms, p99: {:.3}ms", mean, median, p99);
        
        // Throughput estimate
        let qps = 1000.0 / p99;
        println!("  Est. throughput: {:.0} QPS (at p99 latency)", qps);
    }
    
    // Binary quantization test
    println!("\n--- With Binary Quantization ---");
    
    let quantizer = BinaryQuantizer::from_vectors(&vectors);
    let binary_vectors = quantizer.quantize_batch_parallel(&vectors);
    
    println!("Quantized {} vectors", binary_vectors.len());
    
    // Test Hamming distance speed
    let query_binary = quantizer.quantize(&queries[0]);
    let hamming_start = Instant::now();
    let mut distances: Vec<(usize, u32)> = binary_vectors
        .iter()
        .enumerate()
        .map(|(i, b)| (i, vectordb::quantization::hamming_distance(&query_binary, b)))
        .collect();
    distances.sort_by_key(|&(_, d)| d);
    distances.truncate(100); // Get top 100
    let hamming_time = hamming_start.elapsed();
    
    let hamming_ms = hamming_time.as_secs_f64() * 1000.0;
    println!("Linear scan with Hamming: {:.3}ms", hamming_ms);
    println!("Vectors scanned: {}", num_vectors);
    println!("Throughput: {:.0} M vectors/sec", num_vectors as f64 / 1_000_000.0 / hamming_time.as_secs_f64());
}

fn main() {
    println!("VectorDB Scale Analysis");
    println!("Testing how performance scales with dataset size\n");
    
    // Test progressively larger scales
    let test_configs = [
        (100_000, 1024, "100K vectors (quick test)"),
        // (1_000_000, 1024, "1M vectors (laptop-friendly)"),
        // (5_000_000, 512, "5M vectors (moderate)"),
        // Uncomment to test larger scales (warning: takes time!)
        // (10_000_000, 512, "10M vectors (large)"),
        // (50_000_000, 128, "50M vectors (very large)"),
    ];
    
    for (num_vectors, dim, description) in test_configs.iter() {
        println!("\n\n{}", "=".repeat(80));
        println!("TEST: {}", description);
        test_scale(*num_vectors, *dim);
    }
}
