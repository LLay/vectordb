/// Proper recall benchmark with realistic test scenarios
/// 
/// Tests recall using:
/// 1. In-dataset queries (vectors from the dataset itself)
/// 2. Perturbed queries (dataset vectors + small noise)
/// 3. Random queries (completely random, hardest case)

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

fn perturb_vector(vec: &[f32], noise_level: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    vec.iter()
        .map(|&v| v + rng.gen_range(-noise_level..noise_level))
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

fn bench_recall_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_scenarios");
    group.sample_size(10);
    
    println!("\n=== Realistic Recall Benchmark ===\n");
    
    let dim = 256;
    let num_vectors = 10_000;
    let num_queries = 100;
    let k = 10;
    
    println!("Dataset: {} vectors, {} dims", num_vectors, dim);
    println!("Building index...");
    
    let vectors = generate_random_vectors(num_vectors, dim);
    
    let index = ClusteredIndex::build(
        vectors.clone(),
        "recall_proper.bin",
        10,
        30,  // Smaller leaves for better recall
        DistanceMetric::L2,
        20,
    ).expect("Failed to build index");
    
    println!("Index: depth={}, nodes={}\n", index.max_depth(), index.num_nodes());
    
    // Test 3 scenarios
    let scenarios = [
        ("In-Dataset", "exact"),
        ("Perturbed (1% noise)", "perturbed_small"),
        ("Perturbed (5% noise)", "perturbed_medium"),
        ("Random", "random"),
    ];
    
    for (scenario_name, scenario_type) in scenarios {
        println!("--- Scenario: {} ---", scenario_name);
        
        // Generate queries for this scenario
        let queries: Vec<Vec<f32>> = match scenario_type {
            "exact" => {
                // Use vectors from dataset
                (0..num_queries).map(|i| vectors[i * 100].clone()).collect()
            }
            "perturbed_small" => {
                // Add 1% noise to dataset vectors
                (0..num_queries).map(|i| perturb_vector(&vectors[i * 100], 0.01)).collect()
            }
            "perturbed_medium" => {
                // Add 5% noise to dataset vectors
                (0..num_queries).map(|i| perturb_vector(&vectors[i * 100], 0.05)).collect()
            }
            "random" => {
                // Completely random queries
                generate_random_vectors(num_queries, dim)
            }
            _ => unreachable!(),
        };
        
        // Compute ground truth
        let ground_truths: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| compute_ground_truth(&vectors, q, k))
            .collect();
        
        // Test different probe configurations
        let configs = [(1, 2), (2, 3), (3, 5), (5, 5)];
        
        println!("{:<15} {:<12} {:<15}", "Config", "Recall@10", "Latency(μs)");
        println!("{}", "-".repeat(45));
        
        for (probes, rerank) in configs {
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
            
            println!("p{}_r{}          {:<12.1}% {:<15.1}",
                     probes, rerank, avg_recall * 100.0, latency_us);
        }
        
        println!();
    }
    
    println!("\n💡 Interpretation:");
    println!("  - In-Dataset: Should be ~100% (sanity check)");
    println!("  - Perturbed: Realistic queries similar to dataset");
    println!("  - Random: Hardest case, queries unlike any data");
    println!("\n🎯 For production, optimize for your actual query distribution!");
    
    std::fs::remove_file("recall_proper.bin").ok();
    group.finish();
}

criterion_group!(benches, bench_recall_scenarios);
criterion_main!(benches);
