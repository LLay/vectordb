//! Demo of memory-mapped storage for vector indexing
//! 
//! This shows how memory-mapped storage can save RAM while still
//! providing fast query performance. The OS automatically caches
//! hot vectors in memory.

use rand::Rng;
use std::time::Instant;
use vectordb::{ClusteredIndex, DistanceMetric};

fn main() {
    println!("=== Memory-Mapped Storage Demo ===\n");
    
    // Generate test data
    let dimension = 768;
    let num_vectors = 100_000;
    let num_queries = 100;
    
    println!("Generating {} {}-dimensional vectors...", num_vectors, dimension);
    let vectors = generate_vectors(num_vectors, dimension);
    let queries = generate_vectors(num_queries, dimension);
    
    // Build index with mmap storage
    println!("\nBuilding hierarchical index with mmap storage...");
    let start = Instant::now();
    let vector_file = "demo_vectors.bin";
    let index = ClusteredIndex::build(
        vectors.clone(),
        vector_file,
        10,    // branching_factor
        1000,  // max_leaf_size
        DistanceMetric::L2,
        20,    // max_iterations
    ).expect("Failed to build index");
    let build_time = start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());
    
    // Measure memory and disk usage
    let mem_usage = index.memory_usage_bytes();
    let disk_usage = index.disk_usage_bytes();
    println!("\nMemory usage (RAM): {:.2} MB", mem_usage as f64 / 1_048_576.0);
    println!("Disk usage (mmap file): {:.2} MB", disk_usage as f64 / 1_048_576.0);
    
    // Run queries (cold - first access)
    println!("\nRunning {} queries (cold)...", num_queries);
    let start = Instant::now();
    let k = 10;
    let probes = 4;
    let rerank_factor = 5;
    
    for query in &queries {
        let _ = index.search(query, k, probes, rerank_factor);
    }
    
    let query_time_cold = start.elapsed();
    let avg_latency_cold = query_time_cold.as_micros() as f64 / num_queries as f64;
    println!("  Avg latency: {:.2} ms", avg_latency_cold / 1000.0);
    
    // Run queries again (warm - vectors now cached by OS)
    println!("\nRunning {} queries (warm - OS cached)...", num_queries);
    let start = Instant::now();
    
    for query in &queries {
        let _ = index.search(query, k, probes, rerank_factor);
    }
    
    let query_time_warm = start.elapsed();
    let avg_latency_warm = query_time_warm.as_micros() as f64 / num_queries as f64;
    println!("  Avg latency: {:.2} ms", avg_latency_warm / 1000.0);
    
    // Summary
    println!("\n=== Summary ===");
    println!("Dataset: {} vectors @ {} dimensions", num_vectors, dimension);
    println!("RAM usage: {:.2} MB (index structure + quantized vectors)", mem_usage as f64 / 1_048_576.0);
    println!("Disk usage: {:.2} MB (full precision vectors)", disk_usage as f64 / 1_048_576.0);
    println!("\nQuery Performance:");
    println!("  Cold latency:  {:.2} ms (first access, disk I/O)", avg_latency_cold / 1000.0);
    println!("  Warm latency:  {:.2} ms (OS cached)", avg_latency_warm / 1000.0);
    println!("  Speedup:       {:.1}x (warm vs cold)", avg_latency_cold / avg_latency_warm);
    
    let total_size_mb = (num_vectors * dimension * 4) as f64 / 1_048_576.0;
    let ram_saved_mb = total_size_mb - mem_usage as f64 / 1_048_576.0;
    println!("\nMemory Efficiency:");
    println!("  Full vectors would use: {:.2} MB in RAM", total_size_mb);
    println!("  Actual RAM usage:       {:.2} MB", mem_usage as f64 / 1_048_576.0);
    println!("  RAM saved:              {:.2} MB ({:.1}%)", 
        ram_saved_mb,
        ram_saved_mb / total_size_mb * 100.0
    );
    
    // Cleanup
    std::fs::remove_file(vector_file).ok();
}

fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}
