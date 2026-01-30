/// Recall and accuracy benchmarking
/// 
/// Measures the quality of search results by comparing against ground truth (brute force).
/// Tests different rerank_factor and probe configurations to find optimal speed/accuracy tradeoff.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::Rng;
use vectordb::{ClusteredIndex, DistanceMetric};
use std::collections::HashSet;

fn generate_random_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Compute ground truth using brute force search
/// 
/// NOTE: Uses L2-SQUARED (not L2) to match the index's distance function.
/// This is intentional - sqrt is expensive and doesn't change ranking.
fn compute_ground_truth(
    vectors: &[Vec<f32>],
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, vec)| {
            let dist = match metric {
                DistanceMetric::L2 => {
                    // L2-SQUARED (not sqrt) - matches index behavior
                    vec.iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                }
                DistanceMetric::Cosine => {
                    let dot: f32 = vec.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                    let mag_a: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let mag_b: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                    1.0 - (dot / (mag_a * mag_b))
                }
                DistanceMetric::DotProduct => {
                    -vec.iter().zip(query.iter()).map(|(a, b)| a * b).sum::<f32>()
                }
            };
            (idx, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.iter().take(k).map(|(idx, _)| *idx).collect()
}

/// Calculate recall@k: what fraction of true top-k results did we find?
fn calculate_recall(ground_truth: &[usize], results: &[(usize, f32)]) -> f64 {
    let gt_set: HashSet<usize> = ground_truth.iter().copied().collect();
    let found: usize = results.iter().filter(|(idx, _)| gt_set.contains(idx)).count();
    found as f64 / ground_truth.len() as f64
}

/// Calculate average rank error: how far off are the rankings?
fn calculate_rank_error(ground_truth: &[usize], results: &[(usize, f32)]) -> f64 {
    let gt_positions: std::collections::HashMap<usize, usize> = ground_truth
        .iter()
        .enumerate()
        .map(|(pos, &idx)| (idx, pos))
        .collect();

    let mut total_error = 0.0;
    let mut found_count = 0;

    for (result_pos, (idx, _)) in results.iter().enumerate() {
        if let Some(&gt_pos) = gt_positions.get(idx) {
            total_error += (result_pos as i32 - gt_pos as i32).abs() as f64;
            found_count += 1;
        }
    }

    if found_count > 0 {
        total_error / found_count as f64
    } else {
        ground_truth.len() as f64
    }
}

struct RecallMetrics {
    recall: f64,
    rank_error: f64,
    latency_us: f64,
}

fn bench_recall_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_tuning");
    group.sample_size(20);
    
    println!("\n=== Recall Benchmark: Finding Optimal Rerank Factor ===");
    
    let dim = 512; // Smaller dim for faster ground truth computation
    let num_vectors = 50_000;
    let num_queries = 100;
    let k = 10;
    
    println!("Dataset: {} vectors, {} dims", num_vectors, dim);
    println!("Building index...");
    
    let vectors = generate_random_vectors(num_vectors, dim);
    
    let index = ClusteredIndex::build(
        vectors.clone(),
        "bench_recall_tuning.bin",
        10,
        150,
        DistanceMetric::L2,
        20,
    ).expect("Failed to build index");
    
    println!("Index built: depth={}, nodes={}", index.max_depth(), index.num_nodes());
    
    // Generate test queries
    let queries = generate_random_vectors(num_queries, dim);
    
    println!("\nComputing ground truth (this may take a moment)...");
    let ground_truths: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| compute_ground_truth(&vectors, q, k, DistanceMetric::L2))
        .collect();
    println!("Ground truth computed for {} queries", num_queries);
    
    // Test different configurations
    let probe_configs = [1, 2, 3, 4];
    let rerank_configs = [1, 2, 3, 4, 5, 10];
    
    println!("\n{:<10} {:<10} {:<12} {:<15} {:<15}", 
             "Probes", "Rerank", "Recall@10", "Rank Error", "Latency (μs)");
    println!("{}", "-".repeat(70));
    
    for &probes in &probe_configs {
        for &rerank in &rerank_configs {
            let config_name = format!("p{}_r{}", probes, rerank);
            
            // Measure recall
            let mut total_recall = 0.0;
            let mut total_rank_error = 0.0;
            
            for (query, gt) in queries.iter().zip(ground_truths.iter()) {
                let results = index.search(query, k, probes, rerank);
                total_recall += calculate_recall(gt, &results);
                total_rank_error += calculate_rank_error(gt, &results);
            }
            
            let avg_recall = total_recall / num_queries as f64;
            let avg_rank_error = total_rank_error / num_queries as f64;
            
            // Benchmark latency
            let mut latency_us = 0.0;
            group.bench_function(BenchmarkId::from_parameter(&config_name), |b| {
                b.iter(|| {
                    for query in queries.iter() {
                        black_box(index.search(black_box(query), k, probes, rerank));
                    }
                });
                
                // Measure average latency
                let start = std::time::Instant::now();
                for query in queries.iter() {
                    black_box(index.search(black_box(query), k, probes, rerank));
                }
                let elapsed = start.elapsed();
                latency_us = elapsed.as_micros() as f64 / num_queries as f64;
            });
            
            println!("{:<10} {:<10} {:<12.1}% {:<15.2} {:<15.1}", 
                     probes, rerank, avg_recall * 100.0, avg_rank_error, latency_us);
        }
        println!(); // Blank line between probe groups
    }
    
    group.finish();
    
    println!("\n💡 Interpretation:");
    println!("  - Recall@10: % of true top-10 results found (higher is better)");
    println!("  - Rank Error: Average position error for found results (lower is better)");
    println!("  - Latency: Query time in microseconds (lower is better)");
    println!("\n🎯 Look for the sweet spot: high recall + low latency");
}

fn bench_recall_vs_dataset_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_vs_size");
    group.sample_size(10);
    
    println!("\n=== Recall vs Dataset Size ===");
    
    let dim = 256;
    let k = 10;
    let num_queries = 50;
    let probes = 2;
    let rerank = 3;
    
    let sizes = [1_000, 5_000, 10_000, 25_000];
    
    println!("\n{:<12} {:<12} {:<12} {:<15}", "Size", "Recall@10", "Latency(μs)", "Build Time(s)");
    println!("{}", "-".repeat(55));
    
    for &size in &sizes {
        let vectors = generate_random_vectors(size, dim);
        let queries = generate_random_vectors(num_queries, dim);
        
        let build_start = std::time::Instant::now();
        let index = ClusteredIndex::build(
            vectors.clone(),
            format!("bench_recall_size_{}.bin", size),
            10,
            150,
            DistanceMetric::L2,
            20,
        ).expect("Failed to build index");
        let build_time = build_start.elapsed().as_secs_f64();
        
        // Compute ground truth
        let ground_truths: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| compute_ground_truth(&vectors, q, k, DistanceMetric::L2))
            .collect();
        
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
        
        println!("{:<12} {:<12.1}% {:<12.1} {:<15.2}", 
                 size, avg_recall * 100.0, latency_us, build_time);
        
        // Cleanup
        std::fs::remove_file(format!("bench_recall_size_{}.bin", size)).ok();
    }
    
    group.finish();
}

fn bench_recall_k_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_varying_k");
    group.sample_size(20);
    
    println!("\n=== Recall for Different k Values ===");
    
    let dim = 256;
    let num_vectors = 10_000;
    let num_queries = 50;
    let probes = 2;
    let rerank = 3;
    
    println!("Building index for {} vectors...", num_vectors);
    let vectors = generate_random_vectors(num_vectors, dim);
    let queries = generate_random_vectors(num_queries, dim);
    
    let index = ClusteredIndex::build(
        vectors.clone(),
        "bench_recall_k.bin",
        10,
        150,
        DistanceMetric::L2,
        20,
    ).expect("Failed to build index");
    
    let k_values = [5, 10, 20, 50, 100];
    
    println!("\n{:<8} {:<12} {:<15} {:<15}", "k", "Recall@k", "Rank Error", "Latency (μs)");
    println!("{}", "-".repeat(55));
    
    for &k in &k_values {
        // Compute ground truth for this k
        let ground_truths: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| compute_ground_truth(&vectors, q, k, DistanceMetric::L2))
            .collect();
        
        // Measure recall
        let mut total_recall = 0.0;
        let mut total_rank_error = 0.0;
        
        for (query, gt) in queries.iter().zip(ground_truths.iter()) {
            let results = index.search(query, k, probes, rerank);
            total_recall += calculate_recall(gt, &results);
            total_rank_error += calculate_rank_error(gt, &results);
        }
        
        let avg_recall = total_recall / num_queries as f64;
        let avg_rank_error = total_rank_error / num_queries as f64;
        
        // Measure latency
        let start = std::time::Instant::now();
        for query in queries.iter() {
            black_box(index.search(black_box(query), k, probes, rerank));
        }
        let latency_us = start.elapsed().as_micros() as f64 / num_queries as f64;
        
        println!("{:<8} {:<12.1}% {:<15.2} {:<15.1}", 
                 k, avg_recall * 100.0, avg_rank_error, latency_us);
    }
    
    std::fs::remove_file("bench_recall_k.bin").ok();
    group.finish();
}

criterion_group!(
    benches,
    bench_recall_tuning,
    bench_recall_vs_dataset_size,
    bench_recall_k_values,
);
criterion_main!(benches);
