/// Tune tree parameters for optimal recall/speed tradeoff
/// 
/// Tests different branching factors and max_leaf_size values to find
/// the configuration that gives high recall with acceptable latency.

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

fn bench_tree_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_param_tuning");
    group.sample_size(10);
    
    println!("\n=== Tuning Tree Parameters for Optimal Recall ===\n");
    
    let dim = 256;
    let num_vectors = 10_000;
    let num_queries = 50;
    let k = 10;
    
    println!("Dataset: {} vectors, {} dims\n", num_vectors, dim);
    
    let vectors = generate_random_vectors(num_vectors, dim);
    let queries = generate_random_vectors(num_queries, dim);
    
    // Compute ground truth once
    println!("Computing ground truth...");
    let ground_truths: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| compute_ground_truth(&vectors, q, k))
        .collect();
    println!("Done\n");
    
    // Test different configurations
    let configs = [
        // (branching, max_leaf_size, probes, rerank, description)
        (10, 150, 2, 3, "current_default"),
        (10, 50, 2, 3, "smaller_leaves"),
        (10, 30, 2, 3, "tiny_leaves"),
        (10, 20, 2, 3, "micro_leaves"),
        (15, 30, 2, 3, "wider_tree"),
        (8, 30, 2, 3, "narrower_tree"),
        (10, 30, 3, 5, "more_probes"),
        (10, 30, 4, 5, "many_probes"),
    ];
    
    println!("{:<20} {:<8} {:<12} {:<8} {:<10} {:<12} {:<15}", 
             "Config", "Depth", "Nodes", "Leaves", "Recall", "Latency(μs)", "Build(s)");
    println!("{}", "-".repeat(95));
    
    for (branching, max_leaf, probes, rerank, name) in configs {
        let build_start = std::time::Instant::now();
        let index = ClusteredIndex::build(
            vectors.clone(),
            format!("tune_{}_{}.bin", branching, max_leaf),
            branching,
            max_leaf,
            DistanceMetric::L2,
            20,
        ).expect("Failed to build index");
        let build_time = build_start.elapsed().as_secs_f64();
        
        let num_leaves = index.num_nodes() - index.num_nodes() / (branching + 1);
        
        // Measure recall
        let mut total_recall = 0.0;
        for (query, gt) in queries.iter().zip(ground_truths.iter()) {
            let results = index.search(query, k, probes, rerank);
            total_recall += calculate_recall(gt, &results);
        }
        let avg_recall = total_recall / num_queries as f64;
        
        // Measure latency
        let latency_start = std::time::Instant::now();
        for query in queries.iter() {
            black_box(index.search(black_box(query), k, probes, rerank));
        }
        let latency_us = latency_start.elapsed().as_micros() as f64 / num_queries as f64;
        
        println!("{:<20} {:<8} {:<12} {:<8} {:<10.1}% {:<12.1} {:<15.2}", 
                 name, index.max_depth(), index.num_nodes(), num_leaves,
                 avg_recall * 100.0, latency_us, build_time);
        
        // Cleanup
        std::fs::remove_file(format!("tune_{}_{}.bin", branching, max_leaf)).ok();
    }
    
    println!("\n💡 Recommendations:");
    println!("  - Target: >80% recall with <200μs latency");
    println!("  - Smaller leaves = deeper tree = better recall (but slower)");
    println!("  - More probes = more leaves searched = better recall (but slower)");
    println!("  - Find the sweet spot for your use case!");
    
    group.finish();
}

criterion_group!(benches, bench_tree_params);
criterion_main!(benches);
