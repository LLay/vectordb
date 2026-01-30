/// Fast speed benchmark for development
/// 
/// Measures query latency without recall calculations.
/// Should complete in < 10 seconds.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::Rng;
use vectordb::{ClusteredIndex, DistanceMetric};

fn generate_random_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn bench_speed_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("speed_fast");
    group.sample_size(10);
    
    println!("\n=== Fast Speed Benchmark ===\n");
    
    let dim = 128;
    let num_vectors = 5_000;
    
    println!("Building index ({} vectors, {} dims)...", num_vectors, dim);
    let vectors = generate_random_vectors(num_vectors, dim);
    
    let index = ClusteredIndex::build(
        vectors.clone(),
        "speed_fast.bin",
        10,
        30,
        DistanceMetric::L2,
        10,
    ).expect("Failed to build index");
    
    let query = generate_random_vectors(1, dim)[0].clone();
    
    println!("Index: depth={}, nodes={}\n", index.max_depth(), index.num_nodes());
    
    // Test different probe/rerank combinations
    let configs = [
        (1, 2, "low_latency"),
        (2, 3, "balanced"),
        (3, 5, "high_recall"),
        (5, 5, "thorough"),
    ];
    
    for (probes, rerank, name) in configs {
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                black_box(index.search(black_box(&query), 10, probes, rerank))
            })
        });
    }
    
    group.finish();
    std::fs::remove_file("speed_fast.bin").ok();
    
    println!("\n💡 Fast latency benchmark - use for quick performance checks");
}

criterion_group!(benches, bench_speed_fast);
criterion_main!(benches);
