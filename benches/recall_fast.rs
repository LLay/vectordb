/// Fast recall benchmark for development
/// 
/// Uses small datasets and fewer queries for quick iteration.
/// Should complete in < 30 seconds.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use vectordb::{ClusteredIndex, DistanceMetric};
use std::collections::HashSet;

fn generate_random_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
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
    
    // Small dataset for fast iteration
    let dim = 128;
    let num_vectors = 2_000;
    let num_queries = 20;  // Fewer queries
    let k = 10;
    
    println!("Dataset: {} vectors, {} dims, {} queries", num_vectors, dim, num_queries);
    println!("Building index...");
    
    let vectors = generate_random_vectors(num_vectors, dim);
    
    // Test different tree configurations quickly
    let configs = [
        (10, 50, "large_leaves"),
        (10, 30, "medium_leaves"),
        (10, 20, "small_leaves"),
    ];
    
    println!("\n{:<15} {:<8} {:<10} {:<12} {:<15}", 
             "Config", "Depth", "Recall@10", "Latency(μs)", "Build(s)");
    println!("{}", "-".repeat(65));
    
    for (branching, max_leaf, name) in configs {
        let build_start = std::time::Instant::now();
        let index = ClusteredIndex::build(
            vectors.clone(),
            format!("recall_fast_{}.bin", name),
            branching,
            max_leaf,
            DistanceMetric::L2,
            10,  // Fewer k-means iterations
        ).expect("Failed to build index");
        let build_time = build_start.elapsed().as_secs_f64();
        
        // Use in-dataset queries (should have good recall)
        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|i| vectors[i * 100].clone())
            .collect();
        
        // Compute ground truth
        let ground_truths: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| compute_ground_truth(&vectors, q, k))
            .collect();
        
        // Test with balanced config
        let probes = 3;
        let rerank = 3;
        
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
        
        println!("{:<15} {:<8} {:<10.1}% {:<12.1} {:<15.2}", 
                 name, index.max_depth(), avg_recall * 100.0, latency_us, build_time);
        
        std::fs::remove_file(format!("recall_fast_{}.bin", name)).ok();
    }
    
    println!("\n💡 Quick iteration benchmark - use for development");
    println!("   Run full benchmarks (recall_proper) before production!");
    
    group.finish();
}

criterion_group!(benches, bench_recall_fast);
criterion_main!(benches);
