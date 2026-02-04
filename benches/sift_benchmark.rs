/// SIFT-1M Benchmark (VectorDBBench compatible)
/// 
/// Measures performance on standard SIFT dataset:
/// - 1M vectors, 128 dimensions (or subset)
/// - 10K queries with ground truth
/// - Metrics: QPS, latency (p50/p95/p99), recall@10/100
/// 
/// Loads .fvecs/.ivecs format directly - no conversion needed.
/// 
/// Usage:
///   cargo bench --bench sift_benchmark                    # Full 1M dataset
///   SIFT_SIZE=10000 cargo bench --bench sift_benchmark    # 10K subset
///   SIFT_SIZE=100000 cargo bench --bench sift_benchmark   # 100K subset

// Load the SIFT dataset utilities from datasets/sift
#[path = "../datasets/sift/mod.rs"]
mod sift;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use vectordb::{ClusteredIndexWithRaBitQ, DistanceMetric};
use std::collections::HashSet;
use std::time::{Instant, Duration};
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

fn measure_latency_percentiles(
    index: &ClusteredIndexWithRaBitQ,
    queries: &[Vec<f32>],
    num_queries: usize,
    k: usize,
    probes: usize,
    rerank_factor: usize,
) -> (Duration, f64, f64, f64) {
    let mut latencies = Vec::with_capacity(num_queries);
    
    for i in 0..num_queries {
        let query = &queries[i % queries.len()];
        let start = Instant::now();
        let _results = index.search(query, k, probes, rerank_factor);
        latencies.push(start.elapsed());
    }
    
    latencies.sort();
    
    let total: Duration = latencies.iter().sum();
    let p50 = latencies[num_queries / 2];
    let p95 = latencies[(num_queries * 95) / 100];
    let p99 = latencies[(num_queries * 99) / 100];
    
    (total, p50.as_secs_f64() * 1000.0, p95.as_secs_f64() * 1000.0, p99.as_secs_f64() * 1000.0)
}

fn detect_dataset_size() -> Option<usize> {
    // Check environment variable for subset size
    // This avoids conflicts with Criterion's argument parsing
    if let Ok(size_str) = env::var("SIFT_SIZE") {
        if let Ok(size) = size_str.parse::<usize>() {
            if size >= 1000 && size <= 1_000_000 {
                println!("  Using subset: {} vectors (from SIFT_SIZE env var)\n", size);
                return Some(size);
            }
        }
    }
    println!("  Loaded: {} vectors", 1_000_000);
    None
}

fn benchmark_sift(c: &mut Criterion) {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║           SIFT Benchmark Loading                             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Detect dataset size from command line
    let dataset_size = detect_dataset_size();
    let base_file = match dataset_size {
        Some(size) => format!("datasets/sift/data/sift_base_{}.fvecs", size),
        None => "datasets/sift/data/sift_base.fvecs".to_string(),
    };
    
    // Try loading the dataset
    let (vectors, dims) = match sift::loader::read_fvecs(&base_file) {
        Ok(data) => data,
        Err(_) if dataset_size.is_some() => {
            eprintln!("\nError: Subset file not found: {}", base_file);
            eprintln!("Create it first:");
            eprintln!("  cargo run --release --bin create_sift_subset -- {}\n", dataset_size.unwrap());
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("\nError: SIFT dataset not found: {}", base_file);
            eprintln!("Download it first:");
            eprintln!("  cd datasets/sift && ./download.sh\n");
            std::process::exit(1);
        }
    };
    
    println!("Loading base vectors from {}...", base_file);
    println!("  → {} vectors, {} dimensions\n", vectors.len(), dims);
    
    println!("Loading query vectors from datasets/sift/data/sift_query.fvecs...");
    let (queries, _) = sift::loader::read_fvecs("datasets/sift/data/sift_query.fvecs")
        .expect("Failed to load SIFT queries");
    println!("  → {} queries\n", queries.len());
    
    // For subsets, compute ground truth via brute force (fast enough)
    // For full dataset, use precomputed ground truth
    let use_brute_force = vectors.len() < 1_000_000;
    
    let ground_truth = if use_brute_force {
        println!("Computing ground truth via brute force (subset is small)...");
        let num_test_queries = 100.min(queries.len());
        let mut gt = Vec::new();
        for i in 0..num_test_queries {
            let gt_100 = compute_ground_truth_brute_force(&vectors, &queries[i], 100);
            gt.push(gt_100);
            if (i + 1) % 20 == 0 {
                print!("\r  Progress: {} / {} queries", i + 1, num_test_queries);
                std::io::stdout().flush().unwrap();
            }
        }
        println!("\r  → Computed {} ground truth sets\n", gt.len());
        gt
    } else {
        println!("Loading ground truth from datasets/sift/data/sift_groundtruth.ivecs...");
        let gt = sift::loader::read_ivecs("datasets/sift/data/sift_groundtruth.ivecs")
            .expect("Failed to load ground truth");
        println!("  → {} ground truth sets (100-NN each)\n", gt.len());
        gt
    };
    
    // Build index
    println!("Building index...");
    let build_start = Instant::now();
    let index = ClusteredIndexWithRaBitQ::build(
        vectors.clone(),
        &format!("sift_index_{}.bin", vectors.len()),
        100,  // branching_factor
        100,  // target_leaf_size
        DistanceMetric::L2,
        20,   // max_iterations
    ).unwrap();
    let build_time = build_start.elapsed();
    println!("  → Build time: {:.2}s\n", build_time.as_secs_f64());
    
    // Benchmark different configurations
    // Format: (name, probes, rerank_factor)
    let configs = vec![
        ("low_latency", 50, 25),
        ("balanced", 85, 75),
        ("high_recall", 100, 150),
    ];
    
    let dataset_name = match dataset_size {
        Some(size) => format!("sift_{}k", size / 1000),
        None => "sift_1m".to_string(),
    };
    
    for (name, probes, rerank_factor) in configs {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║ {} configuration (probes={}, rerank={})                    ", name, probes, rerank_factor);
        println!("╚═══════════════════════════════════════════════════════════════╝");
        
        // First, let Criterion benchmark and report
        println!("\nCriterion Performance Metrics:");
        let mut group = c.benchmark_group(format!("{}_{}", dataset_name, name));
        
        // Tell Criterion we're measuring queries per second
        group.throughput(Throughput::Elements(1)); // 1 query per iteration
        
        // Search @ k=10
        group.bench_function("search_k10", |b| {
            b.iter(|| {
                let query = &queries[black_box(0)];
                index.search(query, black_box(10), probes, rerank_factor)
            })
        });
        
        group.finish(); // This triggers Criterion to run and print output
        
        // Now calculate recall and additional metrics
        println!("\nAccuracy Metrics (manual calculation):");
        let mut recall_10 = 0.0;
        let mut recall_100 = 0.0;
        let num_test_queries = if use_brute_force {
            ground_truth.len()
        } else {
            100.min(queries.len())
        };
        
        for i in 0..num_test_queries {
            let results = index.search(&queries[i], 100, probes, rerank_factor);
            recall_10 += calculate_recall(&ground_truth[i], &results, 10);
            recall_100 += calculate_recall(&ground_truth[i], &results, 100);
        }
        
        recall_10 /= num_test_queries as f64;
        recall_100 /= num_test_queries as f64;
        
        // Measure latency percentiles
        let (total, p50, p95, p99) = measure_latency_percentiles(
            &index,
            &queries,
            1000,
            10,
            probes,
            rerank_factor,
        );
        let qps = 1000.0 / total.as_secs_f64();
        
        println!("  Recall@10:  {:.1}%", recall_10 * 100.0);
        println!("  Recall@100: {:.1}%", recall_100 * 100.0);
        println!("  QPS:        {:.0} queries/sec (confirms Criterion 'thrpt' above)", qps);
        println!("  p50:        {:.2}ms (should match Criterion 'time' above)", p50);
        println!("  p95:        {:.2}ms", p95);
        println!("  p99:        {:.2}ms", p99);
    }
    
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║           SIFT Benchmark Complete                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)  // Fewer samples for large dataset
        .warm_up_time(Duration::from_secs(5));
    targets = benchmark_sift
}
criterion_main!(benches);
