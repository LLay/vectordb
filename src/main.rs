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
    /// Run a simple test
    Test {
        /// Dimension of vectors
        #[arg(short, long, default_value_t = 128)]
        dim: usize,
        
        /// Number of vectors
        #[arg(short, long, default_value_t = 1000)]
        num: usize,
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
