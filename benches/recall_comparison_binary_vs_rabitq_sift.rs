/// SIFT Binary vs RaBitQ Comparison Benchmark
/// 
/// Directly compares Binary Quantization and RaBitQ on SIFT data:
/// - Same index configuration
/// - Same search parameters
/// - Measures recall, latency, and throughput
/// 
/// Usage:
///   SIFT_SIZE=10000 cargo bench --bench sift_comparison   # 10K subset
///   SIFT_SIZE=100000 cargo bench --bench sift_comparison  # 100K subset

#[path = "../datasets/sift/mod.rs"]
mod sift;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use vectordb::{ClusteredIndex, ClusteredIndexWithRaBitQ, DistanceMetric};
use std::collections::HashSet;
use std::time::Duration;
use std::env;

fn calculate_recall(ground_truth: &[usize], results: &[(usize, f32)], k: usize) -> f64 {
    let gt_set: HashSet<_> = ground_truth.iter().take(k).copied().collect();
    let result_set: HashSet<_> = results.iter().take(k).map(|(idx, _)| *idx).collect();
    
    let intersection = gt_set.intersection(&result_set).count();
    intersection as f64 / k as f64
}

fn load_sift_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<usize>>) {
    let dataset_size = env::var("SIFT_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    
    let base_file = if let Some(size) = dataset_size {
        format!("datasets/sift/data/sift_base_{}.fvecs", size)
    } else {
        "datasets/sift/data/sift_base.fvecs".to_string()
    };
    
    eprintln!("\n╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║       SIFT Binary vs RaBitQ Comparison Benchmark             ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    eprintln!("Loading dataset: {}", base_file);
    let (vectors, _) = sift::loader::read_fvecs(&base_file)
        .unwrap_or_else(|_| panic!("Failed to load {}", base_file));
    
    eprintln!("Loading queries...");
    let (queries, _) = sift::loader::read_fvecs("datasets/sift/data/sift_query.fvecs")
        .expect("Failed to load queries");
    
    eprintln!("Loading ground truth...");
    let ground_truth = sift::loader::read_ivecs("datasets/sift/data/sift_groundtruth.ivecs")
        .expect("Failed to load ground truth");
    
    eprintln!("Loaded: {} vectors, {} queries\n", vectors.len(), queries.len());
    
    (vectors, queries, ground_truth)
}

fn compare_indexes(c: &mut Criterion) {
    let (vectors, queries, ground_truth) = load_sift_dataset();
    let num_vectors = vectors.len();
    
    // Configuration
    // Note: RaBitQ needs higher rerank_factor on low-dimensional data (SIFT-128D)
    // due to ~19% error rate. Binary uses standard values.
    let (branching_factor, target_leaf_size, probes, binary_rerank, rabitq_rerank) = if num_vectors <= 10_000 {
        (10, 100, 3, 10, 30)  // 10K config
    } else if num_vectors <= 100_000 {
        (100, 100, 2, 10, 30)  // 100K config
    } else {
        (100, 100, 2, 10, 30)  // 1M config
    };
    
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("Configuration:");
    eprintln!("  Vectors: {}", num_vectors);
    eprintln!("  Branching factor: {}", branching_factor);
    eprintln!("  Target leaf size: {}", target_leaf_size);
    eprintln!("  Probes per level: {}", probes);
    eprintln!("  Binary rerank factor: {}", binary_rerank);
    eprintln!("  RaBitQ rerank factor: {} (higher due to ~19% error on SIFT-128D)", rabitq_rerank);
    eprintln!("═══════════════════════════════════════════════════════════════\n");
    
    // Build Binary Quantization index
    eprintln!("[1/2] Building Binary Quantization index...");
    let binary_start = std::time::Instant::now();
    let binary_index = ClusteredIndex::build(
        vectors.clone(),
        format!("sift_comparison_binary_{}.bin", num_vectors),
        branching_factor,
        target_leaf_size,
        DistanceMetric::L2,
        10,
    ).expect("Failed to build binary index");
    let binary_build_time = binary_start.elapsed();
    eprintln!("Binary build time: {:?}\n", binary_build_time);
    
    // Build RaBitQ index
    eprintln!("[2/2] Building RaBitQ index...");
    let rabitq_start = std::time::Instant::now();
    let rabitq_index = ClusteredIndexWithRaBitQ::build(
        vectors.clone(),
        format!("sift_comparison_rabitq_{}.bin", num_vectors),
        branching_factor,
        target_leaf_size,
        DistanceMetric::L2,
        10,
    ).expect("Failed to build RaBitQ index");
    let rabitq_build_time = rabitq_start.elapsed();
    eprintln!("RaBitQ build time: {:?}\n", rabitq_build_time);
    
    // Calculate recall for both
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("Calculating recall (100 queries)...\n");
    
    let mut binary_recall10 = 0.0;
    let mut rabitq_recall10 = 0.0;
    
    for i in 0..100.min(queries.len()) {
        let binary_results = binary_index.search(&queries[i], 10, probes, binary_rerank);
        let rabitq_results = rabitq_index.search(&queries[i], 10, probes, rabitq_rerank);
        
        binary_recall10 += calculate_recall(&ground_truth[i], &binary_results, 10);
        rabitq_recall10 += calculate_recall(&ground_truth[i], &rabitq_results, 10);
    }
    
    binary_recall10 /= 100.0;
    rabitq_recall10 /= 100.0;
    
    eprintln!("Binary Quantization Recall@10: {:.2}%", binary_recall10 * 100.0);
    eprintln!("RaBitQ Recall@10:               {:.2}%", rabitq_recall10 * 100.0);
    eprintln!("Improvement:                    {:+.2}%\n", (rabitq_recall10 - binary_recall10) * 100.0);
    
    // Benchmark search latency
    let mut group = c.benchmark_group(format!("sift_{}", num_vectors));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    
    group.bench_function(
        BenchmarkId::new("binary_search", "k10"),
        |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                black_box(binary_index.search(query, 10, probes, binary_rerank))
            })
        },
    );
    
    group.bench_function(
        BenchmarkId::new("rabitq_search", "k10"),
        |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                black_box(rabitq_index.search(query, 10, probes, rabitq_rerank))
            })
        },
    );
    
    group.finish();
    
    eprintln!("\n╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║               Comparison Complete                            ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝\n");
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(5));
    targets = compare_indexes
}

criterion_main!(benches);
