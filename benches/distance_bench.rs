use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rand::Rng;
use vectordb::distance::scalar;

#[cfg(target_arch = "aarch64")]
use vectordb::distance::simd_neon;

fn generate_random_vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    (a, b)
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    
    for dim in [128, 256, 512, 768, 1024, 1536, 2048].iter() {
        let (a, b) = generate_random_vectors(*dim);
        
        group.throughput(Throughput::Bytes((*dim * 4 * 2) as u64)); // 2 vectors, f32 = 4 bytes
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bencher, _| {
            bencher.iter(|| scalar::dot_product_scalar(black_box(&a), black_box(&b)))
        });
        
        #[cfg(target_arch = "aarch64")]
        {
            group.bench_with_input(BenchmarkId::new("neon", dim), dim, |bencher, _| {
                bencher.iter(|| simd_neon::dot_product(black_box(&a), black_box(&b)))
            });
            
            group.bench_with_input(BenchmarkId::new("neon_unrolled", dim), dim, |bencher, _| {
                bencher.iter(|| unsafe { 
                    simd_neon::dot_product_neon_unrolled(black_box(&a), black_box(&b)) 
                })
            });
        }
    }
    
    group.finish();
}

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_squared");
    
    for dim in [128, 256, 512, 768, 1024, 1536, 2048].iter() {
        let (a, b) = generate_random_vectors(*dim);
        
        group.throughput(Throughput::Bytes((*dim * 4 * 2) as u64));
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bencher, _| {
            bencher.iter(|| scalar::l2_squared_scalar(black_box(&a), black_box(&b)))
        });
        
        #[cfg(target_arch = "aarch64")]
        {
            group.bench_with_input(BenchmarkId::new("neon", dim), dim, |bencher, _| {
                bencher.iter(|| simd_neon::l2_squared(black_box(&a), black_box(&b)))
            });
            
            group.bench_with_input(BenchmarkId::new("neon_unrolled", dim), dim, |bencher, _| {
                bencher.iter(|| unsafe { 
                    simd_neon::l2_squared_neon_unrolled(black_box(&a), black_box(&b)) 
                })
            });
        }
    }
    
    group.finish();
}

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");
    group.sample_size(50);
    
    let dim = 1024;
    let mut rng = rand::thread_rng();
    
    for num_vectors in [100, 1000, 10000].iter() {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let vectors: Vec<Vec<f32>> = (0..*num_vectors)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::new("sequential", num_vectors),
            &(&query, &vectors),
            |bencher, (q, vecs)| {
                bencher.iter(|| {
                    vectordb::batch_distances(
                        black_box(q),
                        black_box(vecs),
                        vectordb::DistanceMetric::L2,
                    )
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel", num_vectors),
            &(&query, &vectors),
            |bencher, (q, vecs)| {
                bencher.iter(|| {
                    vectordb::batch_distances_parallel(
                        black_box(q),
                        black_box(vecs),
                        vectordb::DistanceMetric::L2,
                    )
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_l2_distance,
    bench_batch_distances,
);
criterion_main!(benches);
