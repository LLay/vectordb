use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rand::Rng;
use vectordb::{KMeans, DistanceMetric};

fn generate_random_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn generate_clustered_vectors(num: usize, dim: usize, num_clusters: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::with_capacity(num);
    
    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect())
        .collect();
    
    // Generate vectors around centers
    for _ in 0..num {
        let center = &centers[rng.gen_range(0..num_clusters)];
        let vector: Vec<f32> = center
            .iter()
            .map(|&c| c + rng.gen_range(-1.0..1.0))
            .collect();
        vectors.push(vector);
    }
    
    vectors
}

fn bench_kmeans_init_plusplus(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_init_plusplus");
    
    let dim = 1024;
    let num_vectors = 10_000;
    let vectors = generate_random_vectors(num_vectors, dim);
    
    for k in [5, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |bencher, &k| {
            bencher.iter(|| {
                KMeans::init_plusplus(
                    black_box(&vectors),
                    k,
                    DistanceMetric::L2,
                )
            })
        });
    }
    
    group.finish();
}

fn bench_kmeans_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_fit");
    group.sample_size(10); // K-means is slow, reduce samples
    
    let dim = 1024;
    let num_vectors = 5_000;
    let max_iters = 20;
    
    for k in [5, 10, 20].iter() {
        let vectors = generate_clustered_vectors(num_vectors, dim, *k);
        
        group.throughput(Throughput::Elements(num_vectors as u64));
        
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |bencher, &k| {
            bencher.iter(|| {
                KMeans::fit(
                    black_box(&vectors),
                    k,
                    DistanceMetric::L2,
                    max_iters,
                )
            })
        });
    }
    
    group.finish();
}

fn bench_kmeans_assign(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_assign");
    
    let dim = 1024;
    let k = 10;
    let vectors_for_training = generate_random_vectors(1_000, dim);
    let (kmeans, _) = KMeans::fit(&vectors_for_training, k, DistanceMetric::L2, 10);
    
    for num_vectors in [100, 1_000, 10_000].iter() {
        let vectors = generate_random_vectors(*num_vectors, dim);
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    kmeans.assign(black_box(&vectors))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_kmeans_nearest_centroid(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_nearest_centroid");
    
    let dim = 1024;
    let vectors = generate_random_vectors(1_000, dim);
    
    for k in [5, 10, 20, 50, 100].iter() {
        let (kmeans, _) = KMeans::fit(&vectors, *k, DistanceMetric::L2, 10);
        
        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |bencher, _| {
            bencher.iter(|| {
                kmeans.nearest_centroid(black_box(&query))
            })
        });
    }
    
    group.finish();
}

fn bench_kmeans_varying_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_varying_dimensions");
    group.sample_size(10);
    
    let num_vectors = 2_000;
    let k = 10;
    let max_iters = 20;
    
    for dim in [128, 512, 1024, 1536].iter() {
        let vectors = generate_clustered_vectors(num_vectors, *dim, k);
        
        group.throughput(Throughput::Bytes((num_vectors * dim * 4) as u64));
        
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bencher, _| {
            bencher.iter(|| {
                KMeans::fit(
                    black_box(&vectors),
                    k,
                    DistanceMetric::L2,
                    max_iters,
                )
            })
        });
    }
    
    group.finish();
}

fn bench_kmeans_get_clusters(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_get_clusters");
    
    let dim = 1024;
    let k = 10;
    let vectors_for_training = generate_random_vectors(1_000, dim);
    let (kmeans, _) = KMeans::fit(&vectors_for_training, k, DistanceMetric::L2, 10);
    
    for num_vectors in [100, 1_000, 10_000].iter() {
        let vectors = generate_random_vectors(*num_vectors, dim);
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    kmeans.get_clusters(black_box(&vectors))
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_kmeans_init_plusplus,
    bench_kmeans_fit,
    bench_kmeans_assign,
    bench_kmeans_nearest_centroid,
    bench_kmeans_varying_dimensions,
    bench_kmeans_get_clusters,
);
criterion_main!(benches);
