/// Fast recall benchmark for development
/// 
/// Uses pre-generated clustered datasets for quick iteration.
/// Should complete in < 30 seconds.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vectordb::{ClusteredIndex, DistanceMetric};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};

fn load_vectors(filename: &str) -> (Vec<Vec<f32>>, usize) {
    let file = File::open(filename).expect(&format!("Failed to open {}", filename));
    let mut reader = BufReader::new(file);
    
    // Read header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).unwrap();
    
    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    // Read vectors
    let mut vectors = Vec::with_capacity(num_vectors);
    let mut buffer = vec![0u8; dims * 4];
    
    for _ in 0..num_vectors {
        reader.read_exact(&mut buffer).unwrap();
        
        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        vectors.push(vector);
    }
    
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

fn bench_recall_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_fast");
    group.sample_size(10);  // Reduce samples for speed
    
    println!("\n=== Fast Recall Benchmark (for development) ===\n");
    
    // Load pre-generated clustered dataset
    println!("Loading dataset from datasets/data_10k_1024d_100clusters.bin...");
    let (vectors, dim) = load_vectors("datasets/data_10k_1024d_100clusters.bin");
    let num_vectors = vectors.len();
    let num_queries = 50;  // More queries now that we have clustered data
    let k = 10;
    
    println!("Dataset: {} vectors, {} dims, {} queries", num_vectors, dim, num_queries);
    println!("Building index...\n");
    
    // Test different tree configurations quickly
    let configs = [
        (100, 100, "turbopuffer"),
        (10, 100, "balanced"),
        (20, 100, "wide"),
    ];
    
    println!("{:<15} {:<8} {:<10} {:<12} {:<15} {:<10}", 
             "Config", "Depth", "Recall@10", "Latency(μs)", "Build(s)", "Probes");
    println!("{}", "-".repeat(75));
    
    for (branching, target_leaf, name) in configs {
        let build_start = std::time::Instant::now();
        let index = ClusteredIndex::build(
            vectors.clone(),
            format!("recall_fast_{}.bin", name),
            branching,
            target_leaf,
            DistanceMetric::L2,
            15,  // K-means iterations
        ).expect("Failed to build index");
        let build_time = build_start.elapsed().as_secs_f64();
        
        // Use in-dataset queries from different clusters (should have good recall)
        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|i| vectors[i * (num_vectors / num_queries)].clone())
            .collect();
        
        // Compute ground truth
        let ground_truths: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| compute_ground_truth(&vectors, q, k))
            .collect();
        
        // Test with balanced config
        let probes = 5;
        let rerank = 10;
        
        // Measure recall
        let mut total_recall = 0.0;
        for (query, gt) in queries.iter().zip(ground_truths.iter()) {
            let results = index.search(query, k, probes, rerank);
            total_recall += calculate_recall(gt, &results);
        }
        let avg_recall = total_recall / num_queries as f64;
        
        // Measure latency
        let start = std::time::Instant::now();
        for query in queries.iter() {
            black_box(index.search(black_box(query), k, probes, rerank));
        }
        let latency_us = start.elapsed().as_micros() as f64 / num_queries as f64;
        
        println!("{:<15} {:<8} {:<10.1}% {:<12.1} {:<15.2} {:<10}", 
                 name, index.max_depth(), avg_recall * 100.0, latency_us, build_time, probes);
        
        std::fs::remove_file(format!("recall_fast_{}.bin", name)).ok();
    }
    
    println!("\n💡 Fast benchmark using clustered data (100 clusters)");
    println!("   Use for quick development iteration!");
    
    group.finish();
}

criterion_group!(benches, bench_recall_fast);
criterion_main!(benches);
