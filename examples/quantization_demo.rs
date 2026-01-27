//! Demo of binary quantization and quantized index

use rand::Rng;
use std::time::Instant;
use vectordb::{BinaryQuantizer, ClusteredIndex, QuantizedClusteredIndex, DistanceMetric};

fn main() {
    println!("=== Binary Quantization Demo ===\n");

    let dim = 512;
    let num_vectors = 10_000;
    let num_queries = 100;
    let k = 10;

    // Generate random vectors
    println!("Generating {} vectors of dimension {}...", num_vectors, dim);
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    // 1. Test Binary Quantization
    println!("\n--- Binary Quantization ---");
    let quantizer = BinaryQuantizer::from_vectors(&vectors);
    
    println!("Original size: {} bytes per vector", dim * 4);
    println!("Quantized size: {} bytes per vector", (dim + 7) / 8);
    println!("Compression ratio: {}x", dim * 4 / ((dim + 7) / 8));
    
    // Show example quantization
    let example_vec = &vectors[0];
    let binary_vec = quantizer.quantize(example_vec);
    println!("\nExample vector (first 8 values): {:?}", &example_vec[..8]);
    println!("Quantized bits (first 8): {:?}", 
        (0..8).map(|i| if binary_vec.get_bit(i) { "1" } else { "0" }).collect::<Vec<_>>()
    );

    // 2. Compare Regular vs Quantized Index
    println!("\n--- Building Indices ---");
    
    // Regular clustered index
    let build_start = Instant::now();
    let regular_index = ClusteredIndex::build(
        vectors.clone(),
        100,
        DistanceMetric::L2,
        20,
    );
    let regular_build_time = build_start.elapsed();
    println!("Regular index built in {:?}", regular_build_time);

    // Quantized clustered index
    let build_start = Instant::now();
    let quantized_index = QuantizedClusteredIndex::build(
        vectors,
        100,
        DistanceMetric::L2,
        20,
    );
    let quantized_build_time = build_start.elapsed();
    println!("Quantized index built in {:?}", quantized_build_time);

    // 3. Compare Query Performance
    println!("\n--- Query Performance (probes=1) ---");
    
    // Regular index
    let start = Instant::now();
    for query in &queries {
        let _ = regular_index.search_parallel(query, k, 1);
    }
    let regular_time = start.elapsed();
    
    println!("Regular index:   {:?} ({:.2} ms avg)", 
        regular_time, 
        regular_time.as_secs_f64() * 1000.0 / num_queries as f64
    );

    // Quantized index with different rerank factors
    for rerank in [2, 3, 5] {
        let start = Instant::now();
        for query in &queries {
            let _ = quantized_index.search(query, k, 1, rerank);
        }
        let quantized_time = start.elapsed();
        
        println!("Quantized ({}x):  {:?} ({:.2} ms avg)", 
            rerank,
            quantized_time, 
            quantized_time.as_secs_f64() * 1000.0 / num_queries as f64
        );
    }

    // 4. Show memory savings
    println!("\n--- Memory Comparison ---");
    let regular_memory = num_vectors * dim * 4; // f32 = 4 bytes
    let quantized_memory = num_vectors * ((dim + 7) / 8); // 1 bit per dim
    let quantized_with_full = quantized_memory + regular_memory; // Both binary + full
    
    println!("Regular index:   {:.2} MB", regular_memory as f64 / 1_000_000.0);
    println!("Quantized only:  {:.2} MB ({:.1}x smaller)", 
        quantized_memory as f64 / 1_000_000.0,
        regular_memory as f64 / quantized_memory as f64
    );
    println!("Quantized+Full:  {:.2} MB (for reranking)", 
        quantized_with_full as f64 / 1_000_000.0
    );

    // 5. Show accuracy/speed tradeoff
    println!("\n--- Key Insights ---");
    println!("✓ Binary quantization: 32x compression");
    println!("✓ Faster distance computation (Hamming vs Euclidean)");
    println!("✓ Two-phase search: fast binary filter → precise rerank");
    println!("✓ Rerank factor controls accuracy/speed tradeoff");
    println!("  - Lower rerank = faster but may miss some neighbors");
    println!("  - Higher rerank = slower but better accuracy");
}
