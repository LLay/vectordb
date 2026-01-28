/// Detailed query profiling benchmark
/// 
/// Measures time spent in each phase of query execution to identify bottlenecks
/// and track optimization progress.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::Rng;
use vectordb::{ClusteredIndex, DistanceMetric};
use std::time::Instant;

fn generate_random_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Profile query execution phases
fn profile_query_phases(
    index: &ClusteredIndex,
    query: &[f32],
    k: usize,
    probes: usize,
    rerank_factor: usize,
    num_runs: usize,
) -> QueryProfile {
    let mut search_time = 0.0;
    
    for _ in 0..num_runs {
        let start = Instant::now();
        let _results = index.search(query, k, probes, rerank_factor);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0; // ms
        
        search_time += elapsed;
    }
    
    let total_time = search_time / num_runs as f64;
    
    QueryProfile {
        total_ms: total_time,
        num_candidates: probes.pow(index.max_depth() as u32) * 150, // Estimate
    }
}

struct QueryProfile {
    total_ms: f64,
    num_candidates: usize,
}

fn bench_query_profile_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_profile_1M");
    group.sample_size(50);
    
    println!("\n=== Building Index for Profiling (1M vectors) ===");
    let dim = 1024;
    let num_vectors = 1_000_000;
    let vectors = generate_random_vectors(num_vectors, dim);
    
    let index = ClusteredIndex::build(
        vectors,
        "bench_profile_1m.bin",
        10,  // branching
        150, // max_leaf
        DistanceMetric::L2,
        20,  // max_iters
    ).expect("Failed to build index");
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    println!("Index built: max_depth={}, nodes={}", index.max_depth(), index.num_nodes());
    
    // Benchmark with different configurations
    let configs = [
        (10, 1, 2, "low_latency"),
        (10, 2, 3, "balanced"),
        (10, 3, 5, "high_recall"),
        (100, 2, 3, "k100"),
    ];
    
    for (k, probes, rerank, name) in configs.iter() {
        group.bench_function(*name, |b| {
            b.iter(|| {
                index.search(black_box(&query), *k, *probes, *rerank)
            })
        });
        
        // Detailed profiling (not in criterion loop)
        let profile = profile_query_phases(&index, &query, *k, *probes, *rerank, 100);
        println!("\nConfiguration: {}", name);
        println!("  k={}, probes={}, rerank={}", k, probes, rerank);
        println!("  Average latency: {:.3} ms", profile.total_ms);
        println!("  Estimated candidates scanned: {}", profile.num_candidates);
        println!("  Time per candidate: {:.3} μs", 
                 profile.total_ms * 1000.0 / profile.num_candidates as f64);
    }
    
    group.finish();
}

fn bench_query_profile_varying_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_profile_varying_size");
    group.sample_size(20);
    
    let dim = 1024;
    let k = 10;
    let probes = 2;
    let rerank = 3;
    
    for num_vectors in [10_000, 100_000, 1_000_000].iter() {
        println!("\n=== Building Index: {} vectors ===", num_vectors);
        let vectors = generate_random_vectors(*num_vectors, dim);
        
        let build_start = Instant::now();
        let vector_file = format!("bench_varying_{}.bin", num_vectors);
        let index = ClusteredIndex::build(
            vectors,
            &vector_file,
            10,
            150,
            DistanceMetric::L2,
            20,
        ).expect("Failed to build index");
        let build_time = build_start.elapsed().as_secs_f64();
        
        println!("Build time: {:.2}s, depth={}, nodes={}", 
            build_time, index.max_depth(), index.num_nodes());
        
        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    index.search(black_box(&query), k, probes, rerank)
                })
            },
        );
        
        // Profile
        let profile = profile_query_phases(&index, &query, k, probes, rerank, 100);
        println!("Query latency: {:.3} ms", profile.total_ms);
        println!("Candidates: {}", profile.num_candidates);
        
        // Cleanup
        std::fs::remove_file(&vector_file).ok();
    }
    
    group.finish();
}

fn bench_tree_depth_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_depth_impact");
    group.sample_size(20);
    
    let dim = 1024;
    let num_vectors = 100_000;
    let k = 10;
    let probes = 2;
    let rerank = 3;
    
    println!("\n=== Testing Different Branching Factors ===");
    
    for branching in [5, 10, 20].iter() {
        println!("\nBranching factor: {}", branching);
        let vectors = generate_random_vectors(num_vectors, dim);
        
        let vector_file = format!("bench_depth_{}.bin", branching);
        let index = ClusteredIndex::build(
            vectors,
            &vector_file,
            *branching,
            150,
            DistanceMetric::L2,
            20,
        ).expect("Failed to build index");
        
        println!("Tree depth: {}, nodes: {}", index.max_depth(), index.num_nodes());
        
        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        group.bench_with_input(
            BenchmarkId::new("branching", branching),
            branching,
            |bencher, _| {
                bencher.iter(|| {
                    index.search(black_box(&query), k, probes, rerank)
                })
            },
        );
        
        let profile = profile_query_phases(&index, &query, k, probes, rerank, 100);
        println!("Query latency: {:.3} ms", profile.total_ms);
        
        // Cleanup
        std::fs::remove_file(&vector_file).ok();
    }
    
    group.finish();
}

fn bench_cache_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_simulation");
    group.sample_size(30);
    
    let dim = 1024;
    let num_vectors = 100_000;
    let k = 10;
    let probes = 2;
    let rerank = 3;
    
    println!("\n=== Simulating Cache Effects ===");
    let vectors = generate_random_vectors(num_vectors, dim);
    let index = ClusteredIndex::build(
        vectors,
        "bench_cache.bin",
        10,
        150,
        DistanceMetric::L2,
        20
    ).expect("Failed to build index");
    
    let mut rng = rand::thread_rng();
    
    // Test 1: Same query (perfect cache hit simulation)
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    group.bench_function("repeated_query", |bencher| {
        bencher.iter(|| {
            index.search(black_box(&query), k, probes, rerank)
        })
    });
    
    // Test 2: Different queries (cache miss simulation)
    group.bench_function("random_queries", |bencher| {
        bencher.iter(|| {
            let q: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            index.search(black_box(&q), k, probes, rerank)
        })
    });
    
    println!("\nNote: 'repeated_query' shows hot cache performance");
    println!("      'random_queries' shows cold cache performance");
    
    group.finish();
    
    // Cleanup
    std::fs::remove_file("bench_cache.bin").ok();
}

fn bench_parallelism_potential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallelism_potential");
    group.sample_size(20);
    
    let dim = 1024;
    let num_vectors = 100_000;
    let vectors = generate_random_vectors(num_vectors, dim);
    let index = ClusteredIndex::build(
        vectors,
        "bench_parallel.bin",
        10,
        150,
        DistanceMetric::L2,
        20
    ).expect("Failed to build index");
    
    let mut rng = rand::thread_rng();
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    
    println!("\n=== Testing Batch Query Parallelism ===");
    
    // Sequential
    group.bench_function("sequential_100", |bencher| {
        bencher.iter(|| {
            for query in queries.iter() {
                let _ = index.search(black_box(query), 10, 2, 3);
            }
        })
    });
    
    // Parallel (if implemented)
    // Uncomment when batch_search_parallel is available
    /*
    group.bench_function("parallel_100", |bencher| {
        bencher.iter(|| {
            index.batch_search_parallel(black_box(&queries), 10, 2, 3)
        })
    });
    */
    
    println!("Note: Implement batch_search_parallel to see parallel speedup");
    
    group.finish();
    
    // Cleanup
    std::fs::remove_file("bench_param_sweep.bin").ok();
}

criterion_group!(
    benches,
    bench_query_profile_1m,
    bench_query_profile_varying_size,
    bench_tree_depth_impact,
    bench_cache_simulation,
    bench_parallelism_potential,
);
criterion_main!(benches);
