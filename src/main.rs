//! VectorDB CLI - Command-line interface for testing and benchmarking

use clap::{Parser, Subcommand};
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "vectordb")]
#[command(about = "A high-performance vector database", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Test distance computation
    Test {
        /// Dimension of vectors
        #[arg(short, long, default_value_t = 128)]
        dim: usize,
        
        /// Number of vectors
        #[arg(short, long, default_value_t = 1000)]
        num: usize,
    },
    
    /// Benchmark hierarchical clustered index
    Bench {
        /// Dimension of vectors
        #[arg(short, long, default_value_t = 1024)]
        dim: usize,
        
        /// Number of vectors
        #[arg(short, long, default_value_t = 10000)]
        num: usize,
        
        /// Branching factor (clusters per level)
        #[arg(short, long, default_value_t = 10)]
        branching: usize,
        
        /// Number of queries
        #[arg(short = 'q', long, default_value_t = 100)]
        queries: usize,
        
        /// K nearest neighbors
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        
        /// Number of probes per level
        #[arg(short, long, default_value_t = 2)]
        probes: usize,
        
        /// Rerank factor (rerank k*factor candidates)
        #[arg(short, long, default_value_t = 3)]
        rerank: usize,
    },
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Test { dim, num } => {
            run_test(dim, num);
        }
        Commands::Bench { dim, num, branching, queries, k, probes, rerank } => {
            bench_hierarchical_index(dim, num, branching, queries, k, probes, rerank);
        }
    }
}

fn run_test(dim: usize, num: usize) {
    use rand::Rng;
    use vectordb::{DistanceMetric, batch_distances_parallel};
    use std::time::Instant;
    
    println!("Generating {} random vectors of dimension {}...", num, dim);
    
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    println!("Computing distances...");
    let start = Instant::now();
    let mut results = batch_distances_parallel(&query, &vectors, DistanceMetric::L2);
    let duration = start.elapsed();
    
    println!("Computed {} distances in {:?}", results.len(), duration);
    println!("Throughput: {:.2} distances/sec", num as f64 / duration.as_secs_f64());
    
    // Sort by distance and show top 5
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("\nTop 5 nearest neighbors:");
    for (i, (idx, dist)) in results.iter().take(5).enumerate() {
        println!("  {}. Vector {} (distance: {:.4})", i + 1, idx, dist);
    }
}

fn bench_hierarchical_index(
    dim: usize,
    num: usize,
    branching_factor: usize,
    num_queries: usize,
    k: usize,
    probes_per_level: usize,
    rerank_factor: usize,
) {
    use rand::Rng;
    use vectordb::ClusteredIndex;
    use vectordb::DistanceMetric;
    use std::time::Instant;

    println!("=== Hierarchical Clustered Index Benchmark ===");
    println!(
        "Vectors: {}, Dimension: {}, Branching: {}, Queries: {}, K: {}, Probes/Level: {}, Rerank: {}x\n",
        num, dim, branching_factor, num_queries, k, probes_per_level, rerank_factor
    );

    // Generate random vectors
    println!("Generating vectors...");
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    // Build index
    println!("Building hierarchical index...");
    let build_start = Instant::now();
    let index = ClusteredIndex::build(vectors, branching_factor, DistanceMetric::L2, 20);
    let build_time = build_start.elapsed();
    println!("Index built in {:?}", build_time);
    println!("  Levels: {}", index.num_levels());
    println!("  Nodes: {}\n", index.num_nodes());

    // Generate queries
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    // Warm-up
    let _ = index.search(&queries[0], k, probes_per_level, rerank_factor);

    // Benchmark
    println!("Running queries...");
    let start = Instant::now();
    
    for query in &queries {
        let _ = index.search(query, k, probes_per_level, rerank_factor);
    }
    
    let duration = start.elapsed();
    let avg_latency = duration.as_secs_f64() / num_queries as f64;
    let qps = num_queries as f64 / duration.as_secs_f64();

    println!("\nResults:");
    println!("  Build time: {:?}", build_time);
    println!("  Query time: {:?}", duration);
    println!("  Avg latency: {:.2} ms", avg_latency * 1000.0);
    println!("  Throughput: {:.2} QPS", qps);
    println!("\nCompression: {}x (f32 → binary)", 32);
}
