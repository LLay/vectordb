use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rand::Rng;
use vectordb::{ClusteredIndex, BinaryQuantizer, DistanceMetric};
use vectordb::quantization::binary::hamming_distance;

fn generate_random_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn bench_e2e_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_full_pipeline");
    group.sample_size(10);
    
    let dim = 1024;
    let num_vectors = 10_000;
    let branching = 10;
    let max_leaf = 150;
    let k = 10;
    let probes = 2;
    let rerank = 3;
    
    let vectors = generate_random_vectors(num_vectors, dim);
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    group.throughput(Throughput::Elements(num_vectors as u64));
    
    group.bench_function("build_and_search", |bencher| {
        bencher.iter(|| {
            // Build index
            let index = ClusteredIndex::build(
                black_box(vectors.clone()),
                branching,
                max_leaf,
                DistanceMetric::L2,
                20,
            );
            
            // Search
            index.search(black_box(&query), k, probes, rerank)
        })
    });
    
    group.finish();
}

fn bench_e2e_quantized_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_quantized_search");
    
    let dim = 1024;
    
    for num_vectors in [1_000, 10_000, 50_000].iter() {
        let vectors = generate_random_vectors(*num_vectors, dim);
        
        // Build quantizer
        let quantizer = BinaryQuantizer::from_vectors(&vectors);
        
        // Quantize all vectors
        let binary_vectors = quantizer.quantize_batch_parallel(&vectors);
        
        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let query_binary = quantizer.quantize(&query);
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::new("linear_scan", num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    // Linear scan with Hamming distance
                    let mut distances: Vec<(usize, u32)> = binary_vectors
                        .iter()
                        .enumerate()
                        .map(|(i, b)| {
                            (i, hamming_distance(black_box(&query_binary), black_box(b)))
                        })
                        .collect();
                    
                    // Get top 10
                    distances.sort_by_key(|&(_, dist)| dist);
                    distances.truncate(10);
                    distances
                })
            },
        );
    }
    
    group.finish();
}

fn bench_e2e_mixed_precision_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_mixed_precision");
    
    let dim = 1024;
    let num_vectors = 10_000;
    let k = 10;
    let rerank_factor = 10; // Retrieve 100 candidates
    
    let vectors = generate_random_vectors(num_vectors, dim);
    
    // Build quantizer and quantize
    let quantizer = BinaryQuantizer::from_vectors(&vectors);
    let binary_vectors = quantizer.quantize_batch_parallel(&vectors);
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let query_binary = quantizer.quantize(&query);
    
    group.throughput(Throughput::Elements(num_vectors as u64));
    
    group.bench_function("binary_scan_fp32_rerank", |bencher| {
        bencher.iter(|| {
            // Phase 1: Fast binary scan
            let mut candidates: Vec<(usize, u32)> = binary_vectors
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    (i, hamming_distance(black_box(&query_binary), black_box(b)))
                })
                .collect();
            
            // Get top candidates
            candidates.sort_by_key(|&(_, dist)| dist);
            candidates.truncate(k * rerank_factor);
            
            // Phase 2: Rerank with full precision
            let mut reranked: Vec<(usize, f32)> = candidates
                .iter()
                .map(|&(idx, _)| {
                    let dist = vectordb::distance::simd_neon::l2_squared(
                        black_box(&query),
                        black_box(&vectors[idx]),
                    );
                    (idx, dist)
                })
                .collect();
            
            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            reranked.truncate(k);
            reranked
        })
    });
    
    group.finish();
}

fn bench_e2e_batch_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_batch_queries");
    
    let dim = 1024;
    let num_vectors = 10_000;
    let branching = 10;
    let max_leaf = 150;
    let k = 10;
    let probes = 2;
    let rerank = 3;
    
    let vectors = generate_random_vectors(num_vectors, dim);
    let index = ClusteredIndex::build(vectors, branching, max_leaf, DistanceMetric::L2, 20);
    
    for num_queries in [10, 100, 1_000].iter() {
        let queries = generate_random_vectors(*num_queries, dim);
        
        group.throughput(Throughput::Elements(*num_queries as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_queries),
            num_queries,
            |bencher, _| {
                bencher.iter(|| {
                    queries
                        .iter()
                        .map(|q| index.search(black_box(q), k, probes, rerank))
                        .collect::<Vec<_>>()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_e2e_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_memory_efficiency");
    group.sample_size(10);
    
    let dim = 1024;
    
    for num_vectors in [10_000, 50_000, 100_000].iter() {
        let vectors = generate_random_vectors(*num_vectors, dim);
        
        group.throughput(Throughput::Bytes((num_vectors * dim * 4) as u64)); // f32 size
        
        group.bench_with_input(
            BenchmarkId::new("quantize_all", num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    let quantizer = BinaryQuantizer::from_vectors(black_box(&vectors));
                    quantizer.quantize_batch_parallel(black_box(&vectors))
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_e2e_full_pipeline,
    bench_e2e_quantized_search,
    bench_e2e_mixed_precision_search,
    bench_e2e_batch_queries,
    bench_e2e_memory_efficiency,
);
criterion_main!(benches);
