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
    
    // Build index (in-memory first)
    println!("\nBuilding hierarchical index (in-memory)...");
    let start = Instant::now();
    let mut index = ClusteredIndex::build(
        vectors,
        10,    // branching_factor
        1000,  // max_leaf_size
        DistanceMetric::L2,
        20,    // max_iterations
    );
    let build_time = start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());
    
    // Measure memory usage before mmap
    let mem_before = index.memory_usage_bytes();
    println!("\nMemory usage (in-memory): {:.2} MB", mem_before as f64 / 1_048_576.0);
    
    // Run queries (in-memory)
    println!("\nRunning {} queries (in-memory)...", num_queries);
    let start = Instant::now();
    let k = 10;
    let probes = 4;
    let rerank_factor = 5;
    
    for query in &queries {
        let _ = index.search(query, k, probes, rerank_factor);
    }
    
    let query_time = start.elapsed();
    let avg_latency = query_time.as_micros() as f64 / num_queries as f64;
    println!("  Avg latency: {:.2} ms", avg_latency / 1000.0);
    
    // Convert to memory-mapped storage
    println!("\nConverting to memory-mapped storage...");
    let vector_file = "demo_vectors.bin";
    match index.use_mmap_storage(vector_file) {
        Ok(_) => {
            let mem_after = index.memory_usage_bytes();
            println!("\nMemory usage (mmap): {:.2} MB", mem_after as f64 / 1_048_576.0);
            println!("Memory saved: {:.2} MB ({:.1}%)", 
                (mem_before - mem_after) as f64 / 1_048_576.0,
                (mem_before - mem_after) as f64 / mem_before as f64 * 100.0
            );
            
            // Run queries again with mmap (first pass - cold)
            println!("\nRunning {} queries (mmap, cold)...", num_queries);
            let start = Instant::now();
            
            for query in &queries {
                let _ = index.search(query, k, probes, rerank_factor);
            }
            
            let query_time_cold = start.elapsed();
            let avg_latency_cold = query_time_cold.as_micros() as f64 / num_queries as f64;
            println!("  Avg latency: {:.2} ms", avg_latency_cold / 1000.0);
            
            // Run queries again (warm - vectors now cached by OS)
            println!("\nRunning {} queries (mmap, warm)...", num_queries);
            let start = Instant::now();
            
            for query in &queries {
                let _ = index.search(query, k, probes, rerank_factor);
            }
            
            let query_time_warm = start.elapsed();
            let avg_latency_warm = query_time_warm.as_micros() as f64 / num_queries as f64;
            println!("  Avg latency: {:.2} ms", avg_latency_warm / 1000.0);
            
            // Summary
            println!("\n=== Summary ===");
            println!("In-Memory latency:  {:.2} ms", avg_latency / 1000.0);
            println!("Mmap cold latency:  {:.2} ms ({:.1}x slower)", 
                avg_latency_cold / 1000.0,
                avg_latency_cold / avg_latency
            );
            println!("Mmap warm latency:  {:.2} ms ({:.1}x slower)",
                avg_latency_warm / 1000.0,
                avg_latency_warm / avg_latency
            );
            println!("\nConclusion: Mmap adds ~{:.1}x overhead when warm, but saves {:.2} MB RAM",
                avg_latency_warm / avg_latency,
                (mem_before - mem_after) as f64 / 1_048_576.0
            );
            
            // Cleanup
            std::fs::remove_file(vector_file).ok();
        }
        Err(e) => {
            eprintln!("Error converting to mmap: {}", e);
        }
    }
}

fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}
