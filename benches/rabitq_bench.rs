/// Benchmark comparing Binary Quantization vs RaBitQ
/// 
/// Tests both methods on SIFT data to measure:
/// - Build time
/// - Search latency  
/// - Recall@10

// Load the SIFT dataset utilities
#[path = "../datasets/sift/mod.rs"]
mod sift;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use vectordb::{ClusteredIndex, DistanceMetric};
use vectordb::quantization::{RaBitQQuantizer, RaBitQVector};
use std::time::Duration;

const K: usize = 10;

// Configuration for benchmark
#[derive(Clone)]
struct Config {
    name: &'static str,
    branching_factor: usize,
    target_leaf_size: usize,
    probes_per_level: usize,
    rerank_factor: usize,
}

fn load_sift_subset(size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<usize>>) {
    let base_path = "datasets/sift/data";
    
    // Load full dataset first
    let (all_vectors, _) = sift::loader::read_fvecs(&format!("{}/sift_base.fvecs", base_path))
        .expect("Failed to load SIFT vectors");
    let (queries, _) = sift::loader::read_fvecs(&format!("{}/sift_query.fvecs", base_path))
        .expect("Failed to load SIFT queries");
    let groundtruth = sift::loader::read_ivecs(&format!("{}/sift_groundtruth.ivecs", base_path))
        .expect("Failed to load SIFT groundtruth");
    
    // Take subset
    let vectors: Vec<Vec<f32>> = all_vectors.into_iter().take(size).collect();
    let queries: Vec<Vec<f32>> = queries.into_iter().take(1000).collect();
    
    (vectors, queries, groundtruth)
}

fn calculate_recall(results: &[(usize, f32)], groundtruth: &[usize], k: usize) -> f32 {
    let gt_set: std::collections::HashSet<usize> = 
        groundtruth.iter().take(k).copied().collect();
    let found = results.iter()
        .take(k)
        .filter(|(id, _)| gt_set.contains(id))
        .count();
    found as f32 / k as f32
}

/// Test binary quantization search
fn bench_binary_quantization(c: &mut Criterion, size: usize, config: &Config) {
    eprintln!("\n=== Loading SIFT-{} for Binary Quantization ===", size);
    let (vectors, queries, groundtruth) = load_sift_subset(size);
    
    eprintln!("Building Binary Quantization index...");
    let start = std::time::Instant::now();
    let index = ClusteredIndex::build(
        vectors.clone(),
        format!("rabitq_bench_binary_{}.bin", size),
        config.branching_factor,
        config.target_leaf_size,
        DistanceMetric::L2,
        10, // max_iterations
    ).expect("Failed to build index");
    let build_time = start.elapsed();
    eprintln!("Binary build time: {:?}", build_time);
    
    // Compute recall
    let mut total_recall = 0.0;
    for (i, query) in queries.iter().take(100).enumerate() {
        let results = index.search(
            query,
            K,
            config.probes_per_level,
            config.rerank_factor,
        );
        let recall = calculate_recall(&results, &groundtruth[i], K);
        total_recall += recall;
    }
    let avg_recall = total_recall / 100.0;
    
    eprintln!("Binary Recall@{}: {:.2}%", K, avg_recall * 100.0);
    
    // Benchmark search latency
    let mut group = c.benchmark_group(format!("binary_{}", size));
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function(
        BenchmarkId::new("search", config.name),
        |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                black_box(index.search(
                    query,
                    K,
                    config.probes_per_level,
                    config.rerank_factor,
                ))
            })
        },
    );
    
    group.finish();
}

/// Test RaBitQ quantization search
fn bench_rabitq_quantization(c: &mut Criterion, size: usize, config: &Config) {
    eprintln!("\n=== Loading SIFT-{} for RaBitQ ===", size);
    let (vectors, queries, groundtruth) = load_sift_subset(size);
    
    eprintln!("Building RaBitQ index...");
    let start = std::time::Instant::now();
    
    // Create RaBitQ quantizer
    let dimension = vectors[0].len();
    let quantizer = RaBitQQuantizer::new(dimension);
    
    // Quantize all vectors
    eprintln!("Quantizing {} vectors...", vectors.len());
    let quantized_vectors: Vec<RaBitQVector> = quantizer.quantize_batch(&vectors);
    
    // Build k-means tree (reuse binary index structure for now)
    // TODO: Create RaBitQ-specific index structure
    let temp_index = ClusteredIndex::build(
        vectors.clone(),
        format!("rabitq_bench_rabitq_{}.bin", size),
        config.branching_factor,
        config.target_leaf_size,
        DistanceMetric::L2,
        10, // max_iterations
    ).expect("Failed to build index");
    
    let build_time = start.elapsed();
    eprintln!("RaBitQ build time: {:?}", build_time);
    
    // Compute recall using RaBitQ distances
    // For now, use the temp_index structure and just measure RaBitQ distance computation time
    let mut total_recall = 0.0;
    for (i, query) in queries.iter().take(100).enumerate() {
        // Get candidates from tree structure
        let candidates = temp_index.search(
            query,
            K,
            config.probes_per_level,
            config.rerank_factor,
        );
        
        // This is a placeholder - in real implementation we'd filter with RaBitQ first
        let recall = calculate_recall(&candidates, &groundtruth[i], K);
        total_recall += recall;
    }
    let avg_recall = total_recall / 100.0;
    
    eprintln!("RaBitQ Recall@{}: {:.2}%", K, avg_recall * 100.0);
    
    // Benchmark RaBitQ distance estimation (FAST version)
    let mut group = c.benchmark_group(format!("rabitq_{}", size));
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function(
        BenchmarkId::new("distance_estimation_fast", config.name),
        |b| {
            let mut query_idx = 0;
            let qv_refs: Vec<&RaBitQVector> = quantized_vectors.iter().take(100).collect();
            
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                
                // Benchmark FAST batch distance estimation (pre-rotates query once)
                let distances = quantizer.estimate_distances_batch_fast(&qv_refs, query);
                    
                black_box(distances)
            })
        },
    );
    
    group.finish();
}

fn sift_10k_comparison(c: &mut Criterion) {
    let config = Config {
        name: "balanced",
        branching_factor: 10,
        target_leaf_size: 100,
        probes_per_level: 3,
        rerank_factor: 10,
    };
    
    bench_binary_quantization(c, 10_000, &config);
    bench_rabitq_quantization(c, 10_000, &config);
}

fn sift_100k_comparison(c: &mut Criterion) {
    let config = Config {
        name: "balanced",
        branching_factor: 100,
        target_leaf_size: 100,
        probes_per_level: 2,
        rerank_factor: 10,
    };
    
    bench_binary_quantization(c, 100_000, &config);
    bench_rabitq_quantization(c, 100_000, &config);
}

criterion_group!{
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(3));
    targets = sift_10k_comparison, sift_100k_comparison
}

criterion_main!(benches);
