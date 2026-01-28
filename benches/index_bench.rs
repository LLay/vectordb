use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rand::Rng;
use vectordb::{ClusteredIndex, DistanceMetric};

fn generate_random_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn bench_index_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_build");
    group.sample_size(10); // Building indices is slow, reduce sample size
    
    let dim = 1024;
    let branching = 10;
    let max_leaf = 150;
    let max_iters = 20;
    
    for num_vectors in [1_000, 10_000, 50_000].iter() {
        let vectors = generate_random_vectors(*num_vectors, dim);
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::new("adaptive", num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    ClusteredIndex::build(
                        black_box(vectors.clone()),
                        branching,
                        max_leaf,
                        DistanceMetric::L2,
                        max_iters,
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn bench_index_search_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_search_k");
    
    let dim = 1024;
    let num_vectors = 10_000;
    let branching = 10;
    let max_leaf = 150;
    let probes = 2;
    let rerank = 3;
    
    let vectors = generate_random_vectors(num_vectors, dim);
    let index = ClusteredIndex::build(vectors, branching, max_leaf, DistanceMetric::L2, 20);
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    for k in [1, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*k as u64));
        
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |bencher, &k| {
            bencher.iter(|| {
                index.search(black_box(&query), k, probes, rerank)
            })
        });
    }
    
    group.finish();
}

fn bench_index_search_varying_probes(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_search_probes");
    
    let dim = 1024;
    let num_vectors = 10_000;
    let branching = 10;
    let max_leaf = 150;
    let k = 10;
    let rerank = 3;
    
    let vectors = generate_random_vectors(num_vectors, dim);
    let index = ClusteredIndex::build(vectors, branching, max_leaf, DistanceMetric::L2, 20);
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    for probes in [1, 2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(probes), probes, |bencher, &probes| {
            bencher.iter(|| {
                index.search(black_box(&query), k, probes, rerank)
            })
        });
    }
    
    group.finish();
}

fn bench_index_search_varying_rerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_search_rerank");
    
    let dim = 1024;
    let num_vectors = 10_000;
    let branching = 10;
    let max_leaf = 150;
    let k = 10;
    let probes = 2;
    
    let vectors = generate_random_vectors(num_vectors, dim);
    let index = ClusteredIndex::build(vectors, branching, max_leaf, DistanceMetric::L2, 20);
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    for rerank_factor in [1, 2, 3, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(rerank_factor),
            rerank_factor,
            |bencher, &rerank_factor| {
                bencher.iter(|| {
                    index.search(black_box(&query), k, probes, rerank_factor)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_index_search_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_search_dimensions");
    
    let num_vectors = 5_000;
    let branching = 10;
    let max_leaf = 150;
    let k = 10;
    let probes = 2;
    let rerank = 3;
    
    for dim in [128, 512, 1024, 1536].iter() {
        let vectors = generate_random_vectors(num_vectors, *dim);
        let index = ClusteredIndex::build(
            vectors,
            branching,
            max_leaf,
            DistanceMetric::L2,
            20,
        );
        
        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..*dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        group.throughput(Throughput::Bytes((*dim * 4) as u64)); // f32 = 4 bytes
        
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bencher, _| {
            bencher.iter(|| {
                index.search(black_box(&query), k, probes, rerank)
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_index_build,
    bench_index_search_varying_k,
    bench_index_search_varying_probes,
    bench_index_search_varying_rerank,
    bench_index_search_dimensions,
);
criterion_main!(benches);
