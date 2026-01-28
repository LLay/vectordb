use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rand::Rng;
use vectordb::{BinaryQuantizer, BinaryVector};
use vectordb::quantization::binary::hamming_distance;

fn generate_random_vectors(num: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn bench_quantize_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_single");
    
    for dim in [128, 256, 512, 768, 1024, 1536, 2048].iter() {
        let mut rng = rand::thread_rng();
        let vector: Vec<f32> = (0..*dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let quantizer = BinaryQuantizer::new(*dim, 0.0);
        
        group.throughput(Throughput::Bytes((*dim * 4) as u64)); // Input: f32 vector
        
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bencher, _| {
            bencher.iter(|| {
                quantizer.quantize(black_box(&vector))
            })
        });
    }
    
    group.finish();
}

fn bench_quantize_batch_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_batch_sequential");
    
    let dim = 1024;
    
    for num_vectors in [100, 1_000, 10_000].iter() {
        let vectors = generate_random_vectors(*num_vectors, dim);
        let quantizer = BinaryQuantizer::new(dim, 0.0);
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    quantizer.quantize_batch(black_box(&vectors))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_quantize_batch_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_batch_parallel");
    
    let dim = 1024;
    
    for num_vectors in [100, 1_000, 10_000].iter() {
        let vectors = generate_random_vectors(*num_vectors, dim);
        let quantizer = BinaryQuantizer::new(dim, 0.0);
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    quantizer.quantize_batch_parallel(black_box(&vectors))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_hamming_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamming_distance");
    
    for dim in [128, 256, 512, 768, 1024, 1536, 2048].iter() {
        let mut rng = rand::thread_rng();
        let v1: Vec<f32> = (0..*dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let v2: Vec<f32> = (0..*dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        let quantizer = BinaryQuantizer::new(*dim, 0.0);
        let b1 = quantizer.quantize(&v1);
        let b2 = quantizer.quantize(&v2);
        
        group.throughput(Throughput::Bytes(((*dim + 7) / 8) as u64)); // Binary size
        
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bencher, _| {
            bencher.iter(|| {
                hamming_distance(black_box(&b1), black_box(&b2))
            })
        });
    }
    
    group.finish();
}

fn bench_hamming_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamming_batch");
    
    let dim = 1024;
    let quantizer = BinaryQuantizer::new(dim, 0.0);
    
    for num_vectors in [100, 1_000, 10_000].iter() {
        let vectors = generate_random_vectors(*num_vectors, dim);
        let binaries: Vec<BinaryVector> = vectors
            .iter()
            .map(|v| quantizer.quantize(v))
            .collect();
        
        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let query_binary = quantizer.quantize(&query);
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    binaries
                        .iter()
                        .map(|b| hamming_distance(black_box(&query_binary), black_box(b)))
                        .collect::<Vec<_>>()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantizer_from_vectors");
    
    let dim = 1024;
    
    for num_vectors in [100, 1_000, 10_000].iter() {
        let vectors = generate_random_vectors(*num_vectors, dim);
        
        group.throughput(Throughput::Elements(*num_vectors as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            num_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    BinaryQuantizer::from_vectors(black_box(&vectors))
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_quantize_single,
    bench_quantize_batch_sequential,
    bench_quantize_batch_parallel,
    bench_hamming_distance,
    bench_hamming_batch,
    bench_compression_ratio,
);
criterion_main!(benches);
