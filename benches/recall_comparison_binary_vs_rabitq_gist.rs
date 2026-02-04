/// GIST Binary vs RaBitQ Comparison Benchmark
/// 
/// Compares Binary Quantization and RaBitQ on GIST-960D data:
/// - 960 dimensions (vs SIFT's 128D)
/// - Same index configuration
/// - Same search parameters
/// - Measures recall, latency, and throughput
/// 
/// GIST-960D is what the RaBitQ paper tested on, so we expect:
/// - Much lower distance error (<5% vs 20% on SIFT)
/// - Better recall improvement over binary
/// - RaBitQ to demonstrate its advantages
/// 
/// Usage:
///   cargo bench --bench gist_comparison

#[path = "../datasets/gist/mod.rs"]
mod gist;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use vectordb::{ClusteredIndex, ClusteredIndexWithRaBitQ, DistanceMetric};
use std::collections::HashSet;
use std::time::Duration;
use std::env;
use std::io::Write;

fn calculate_recall(ground_truth: &[usize], results: &[(usize, f32)], k: usize) -> f64 {
    let gt_set: HashSet<_> = ground_truth.iter().take(k).copied().collect();
    let result_set: HashSet<_> = results.iter().take(k).map(|(idx, _)| *idx).collect();
    
    let intersection = gt_set.intersection(&result_set).count();
    intersection as f64 / k as f64
}

fn compute_ground_truth_brute_force(
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

fn load_gist_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<usize>>) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║       GIST-960D Binary vs RaBitQ Comparison                  ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    // Check for subset size from environment variable
    let dataset_size = env::var("GIST_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    
    let base_file = if let Some(size) = dataset_size {
        format!("datasets/gist/data/gist/gist_base_{}.fvecs", size)
    } else {
        "datasets/gist/data/gist/gist_base.fvecs".to_string()
    };
    
    eprintln!("Loading GIST-960D dataset...");
    eprintln!("  This is the dataset the RaBitQ paper tested on!");
    eprintln!("  Expected: <5% error (vs 20% on SIFT-128D)\n");
    
    eprintln!("Loading base vectors from {}...", base_file);
    let (vectors, dim) = match gist::loader::read_fvecs(&base_file) {
        Ok(data) => data,
        Err(_) if dataset_size.is_some() => {
            eprintln!("\nError: Subset file not found: {}", base_file);
            eprintln!("Create it first:");
            eprintln!("  cargo run --release --bin create_gist_subset -- {}\n", dataset_size.unwrap());
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("\nError: GIST dataset not found: {}", base_file);
            eprintln!("Download and extract it first:");
            eprintln!("  cd datasets/gist/data && tar -xzf gist.tar.gz\n");
            std::process::exit(1);
        }
    };
    eprintln!("  → {} vectors, {} dimensions\n", vectors.len(), dim);
    
    eprintln!("Loading query vectors...");
    let (queries, _) = gist::loader::read_fvecs("datasets/gist/data/gist/gist_query.fvecs")
        .expect("Failed to load GIST queries");
    eprintln!("  → {} queries\n", queries.len());
    
    // For subsets, compute ground truth via brute force (fast enough for 100K)
    // For full dataset, use precomputed ground truth
    let use_brute_force = vectors.len() < 1_000_000;
    
    let ground_truth = if use_brute_force {
        eprintln!("Computing ground truth via brute force (subset is small enough)...");
        let num_test_queries = 100.min(queries.len());
        let mut gt = Vec::new();
        for i in 0..num_test_queries {
            let gt_100 = compute_ground_truth_brute_force(&vectors, &queries[i], 100);
            gt.push(gt_100);
            if (i + 1) % 20 == 0 {
                eprint!("\r  Progress: {} / {} queries", i + 1, num_test_queries);
                std::io::stderr().flush().unwrap();
            }
        }
        eprintln!("\r  → Computed {} ground truth sets\n", gt.len());
        gt
    } else {
        eprintln!("Loading precomputed ground truth...");
        let gt = gist::loader::read_ivecs("datasets/gist/data/gist/gist_groundtruth.ivecs")
            .expect("Failed to load GIST ground truth");
        eprintln!("  → {} ground truth sets (100-NN each)\n", gt.len());
        gt
    };
    
    (vectors, queries, ground_truth)
}

fn compare_indexes(c: &mut Criterion) {
    let (vectors, queries, ground_truth) = load_gist_dataset();
    let num_vectors = vectors.len();
    
    // Configuration adapts to dataset size
    // Note: On high-dimensional data (960D), RaBitQ should need LOWER rerank_factor than SIFT
    // because error rate is much lower (~5% vs 20%)
    let (branching_factor, target_leaf_size, probes, binary_rerank, rabitq_rerank) = if num_vectors <= 100_000 {
        (100, 100, 3, 10, 10)  // 100K config
    } else {
        (100, 100, 2, 10, 10)  // 1M config
    };
    
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("Configuration:");
    eprintln!("  Vectors: {}", num_vectors);
    eprintln!("  Dimensions: 960 (high-dimensional!)");
    eprintln!("  Branching factor: {}", branching_factor);
    eprintln!("  Target leaf size: {}", target_leaf_size);
    eprintln!("  Probes per level: {}", probes);
    eprintln!("  Binary rerank factor: {}", binary_rerank);
    eprintln!("  RaBitQ rerank factor: {} (same - should work well on 960D)", rabitq_rerank);
    eprintln!("═══════════════════════════════════════════════════════════════\n");
    
    // Build Binary Quantization index
    eprintln!("[1/2] Building Binary Quantization index...");
    let binary_start = std::time::Instant::now();
    let binary_index = ClusteredIndex::build(
        vectors.clone(),
        "gist_comparison_binary_1000000.bin",
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
        "gist_comparison_rabitq_1000000.bin",
        branching_factor,
        target_leaf_size,
        DistanceMetric::L2,
        10,
    ).expect("Failed to build RaBitQ index");
    let rabitq_build_time = rabitq_start.elapsed();
    eprintln!("RaBitQ build time: {:?}\n", rabitq_build_time);
    
    // Calculate recall for both
    eprintln!("═══════════════════════════════════════════════════════════════");
    
    // Use fewer test queries for subsets (ground truth was computed for fewer)
    let num_test_queries = if num_vectors < 1_000_000 {
        ground_truth.len()
    } else {
        100.min(queries.len())
    };
    
    eprintln!("Calculating recall ({} queries)...\n", num_test_queries);
    
    let mut binary_recall10 = 0.0;
    let mut rabitq_recall10 = 0.0;
    let mut binary_recall100 = 0.0;
    let mut rabitq_recall100 = 0.0;
    
    for i in 0..num_test_queries {
        let binary_results = binary_index.search(&queries[i], 100, probes, binary_rerank);
        let rabitq_results = rabitq_index.search(&queries[i], 100, probes, rabitq_rerank);
        
        binary_recall10 += calculate_recall(&ground_truth[i], &binary_results, 10);
        rabitq_recall10 += calculate_recall(&ground_truth[i], &rabitq_results, 10);
        binary_recall100 += calculate_recall(&ground_truth[i], &binary_results, 100);
        rabitq_recall100 += calculate_recall(&ground_truth[i], &rabitq_results, 100);
    }
    
    binary_recall10 /= num_test_queries as f64;
    rabitq_recall10 /= num_test_queries as f64;
    binary_recall100 /= num_test_queries as f64;
    rabitq_recall100 /= num_test_queries as f64;
    
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║                     Recall Results                           ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Recall@10:");
    eprintln!("  Binary:     {:.2}%", binary_recall10 * 100.0);
    eprintln!("  RaBitQ:     {:.2}%", rabitq_recall10 * 100.0);
    eprintln!("  Improvement: {:+.2}%", (rabitq_recall10 - binary_recall10) * 100.0);
    eprintln!();
    eprintln!("Recall@100:");
    eprintln!("  Binary:     {:.2}%", binary_recall100 * 100.0);
    eprintln!("  RaBitQ:     {:.2}%", rabitq_recall100 * 100.0);
    eprintln!("  Improvement: {:+.2}%", (rabitq_recall100 - binary_recall100) * 100.0);
    eprintln!();
    
    // Benchmark search latency
    let dataset_name = if num_vectors >= 1_000_000 {
        "gist_1m".to_string()
    } else {
        format!("gist_{}k", num_vectors / 1000)
    };
    
    let mut group = c.benchmark_group(&dataset_name);
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
    eprintln!("║               GIST Comparison Complete                       ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    eprintln!("Key Findings:");
    eprintln!("  - GIST is 960D (vs SIFT's 128D)");
    eprintln!("  - RaBitQ paper tested on GIST-960D");
    eprintln!("  - Expected: RaBitQ should outperform binary on this dataset");
    eprintln!("  - Concentration effects work better at high dimensions\n");
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(5));
    targets = compare_indexes
}

criterion_main!(benches);
