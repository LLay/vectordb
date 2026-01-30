/// Benchmark for 1M vector dataset
/// 
/// Measures search latency and recall on large-scale clustered data.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vectordb::{ClusteredIndex, DistanceMetric};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};

fn load_vectors(filename: &str) -> (Vec<Vec<f32>>, usize) {
    println!("Loading dataset from {}...", filename);
    let file = File::open(filename).expect(&format!("Failed to open {}", filename));
    let mut reader = BufReader::new(file);
    
    // Read header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).unwrap();
    
    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    println!("  → {} vectors, {} dimensions", num_vectors, dims);
    
    // Read vectors
    let mut vectors = Vec::with_capacity(num_vectors);
    let mut buffer = vec![0u8; dims * 4];
    
    for i in 0..num_vectors {
        reader.read_exact(&mut buffer).unwrap();
        
        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        vectors.push(vector);
        
        if (i + 1) % 100_000 == 0 {
            println!("    Loaded {}/{} vectors", i + 1, num_vectors);
        }
    }
    
    println!("  ✓ Loaded successfully\n");
    
    (vectors, dims)
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

fn bench_1m_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_1m");
    group.sample_size(10);
    
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║           1M Vector Scale Benchmark                          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    // Load 1M vector dataset
    let (vectors, dims) = load_vectors("datasets/data_1m_1024d_1000clusters.bin");
    let num_vectors = vectors.len();
    let num_queries = 100;
    let k = 10;
    
    println!("Dataset: {} vectors, {} dims", num_vectors, dims);
    println!("Queries: {}, k={}\n", num_queries, k);
    
    // Select query vectors from different parts of the dataset
    println!("Selecting query vectors...");
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|i| vectors[i * (num_vectors / num_queries)].clone())
        .collect();
    
    // Compute ground truth for a subset of queries (expensive!)
    println!("Computing ground truth for {} queries (this will take a moment)...", 10);
    let ground_truths: Vec<Vec<usize>> = queries
        .iter()
        .take(10)
        .map(|q| {
            let gt = compute_ground_truth(&vectors, q, k);
            gt
        })
        .collect();
    println!("  ✓ Ground truth computed\n");
    
    // Test configurations
    let configs = [
        (100, 100, 3, 10, "conservative"),
        (100, 100, 5, 10, "balanced"),
        (100, 100, 10, 10, "aggressive"),
    ];
    
    println!("{:<15} {:<8} {:<10} {:<12} {:<15} {:<10} {:<10}", 
             "Config", "Depth", "Recall@10", "Latency(μs)", "Build(s)", "Probes", "Rerank");
    println!("{}", "=".repeat(90));
    
    for (branching, target_leaf, probes, rerank, name) in configs {
        println!("\nBuilding index: {}...", name);
        let build_start = std::time::Instant::now();
        let index = ClusteredIndex::build(
            vectors.clone(),
            format!("scale_1m_{}.bin", name),
            branching,
            target_leaf,
            DistanceMetric::L2,
            20,  // K-means iterations
        ).expect("Failed to build index");
        let build_time = build_start.elapsed().as_secs_f64();
        
        // Measure recall on subset
        let mut total_recall = 0.0;
        for (query, gt) in queries.iter().take(10).zip(ground_truths.iter()) {
            let results = index.search(query, k, probes, rerank);
            total_recall += calculate_recall(gt, &results);
        }
        let avg_recall = total_recall / 10.0;
        
        // Measure latency on all queries
        let start = std::time::Instant::now();
        for query in queries.iter() {
            black_box(index.search(black_box(query), k, probes, rerank));
        }
        let latency_us = start.elapsed().as_micros() as f64 / num_queries as f64;
        
        // Measure p99 latency
        let mut latencies: Vec<u128> = Vec::new();
        for query in queries.iter() {
            let start = std::time::Instant::now();
            black_box(index.search(black_box(query), k, probes, rerank));
            latencies.push(start.elapsed().as_micros());
        }
        latencies.sort();
        let p99_latency = latencies[(latencies.len() * 99) / 100];
        
        println!("{:<15} {:<8} {:<10.1}% {:<12.1} {:<15.2} {:<10} {:<10}", 
                 name, index.max_depth(), avg_recall * 100.0, latency_us, build_time, probes, rerank);
        println!("                         p99: {:.1} μs", p99_latency);
        
        std::fs::remove_file(format!("scale_1m_{}.bin", name)).ok();
    }
    
    println!("\n{}", "=".repeat(90));
    println!("✓ Benchmark complete!");
    
    group.finish();
}

criterion_group!(benches, bench_1m_scale);
criterion_main!(benches);
