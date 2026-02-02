/// SIFT Microbenchmarks
/// 
/// Identifies performance bottlenecks using real SIFT data.
/// Use SIFT_SIZE=10000 or SIFT_SIZE=100000 for fast iteration.
/// 
/// Usage:
///   SIFT_SIZE=10000 cargo bench --bench sift_microbench    # Fast (~30s)
///   SIFT_SIZE=100000 cargo bench --bench sift_microbench   # Medium (~5min)

#[path = "../datasets/sift/mod.rs"]
mod sift;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use vectordb::{ClusteredIndex, DistanceMetric, BinaryQuantizer, KMeans};
use std::time::Duration;
use std::env;

fn detect_dataset_size() -> usize {
    if let Ok(size_str) = env::var("SIFT_SIZE") {
        if let Ok(size) = size_str.parse::<usize>() {
            if size >= 1000 && size <= 1_000_000 {
                println!("Using SIFT subset: {} vectors\n", size);
                return size;
            }
        }
    }
    println!("Using full SIFT: 1000000 vectors\n");
    1_000_000
}

fn load_sift_data(size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let base_file = if size < 1_000_000 {
        format!("datasets/sift/data/sift_base_{}.fvecs", size)
    } else {
        "datasets/sift/data/sift_base.fvecs".to_string()
    };
    
    let (vectors, _) = sift::loader::read_fvecs(&base_file)
        .expect("Failed to load SIFT base vectors");
    
    let (queries, _) = sift::loader::read_fvecs("datasets/sift/data/sift_query.fvecs")
        .expect("Failed to load SIFT queries");
    
    println!("Loaded: {} vectors, {} queries", vectors.len(), queries.len());
    (vectors, queries)
}

// Benchmark 1: Distance Calculation (most frequent operation)
fn bench_distance_calculation(c: &mut Criterion) {
    let size = detect_dataset_size();
    let (vectors, queries) = load_sift_data(size);
    
    let mut group = c.benchmark_group("distance_calculation");
    group.throughput(Throughput::Elements(1));
    
    // L2 distance (used everywhere)
    group.bench_function("l2_distance_128d", |b| {
        b.iter(|| {
            let v1 = &vectors[black_box(0)];
            let v2 = &queries[black_box(0)];
            let dist: f32 = v1.iter()
                .zip(v2.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            black_box(dist)
        })
    });
    
    // Batch distance (query vs many vectors)
    group.bench_function("batch_distances_100", |b| {
        b.iter(|| {
            let query = &queries[black_box(0)];
            let batch = &vectors[0..100.min(vectors.len())];
            let distances: Vec<f32> = batch.iter()
                .map(|vec| {
                    vec.iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum()
                })
                .collect();
            black_box(distances)
        })
    });
    
    group.finish();
}

// Benchmark 2: Binary Quantization (critical for search)
fn bench_binary_quantization(c: &mut Criterion) {
    let size = detect_dataset_size();
    let (vectors, queries) = load_sift_data(size);
    
    let mut group = c.benchmark_group("binary_quantization");
    
    // Build quantizer
    println!("Building quantizer for {} vectors...", vectors.len());
    let quantizer = BinaryQuantizer::from_vectors(&vectors);
    
    // Quantize single vector
    group.throughput(Throughput::Elements(1));
    group.bench_function("quantize_single_vector", |b| {
        b.iter(|| {
            let vec = &queries[black_box(0)];
            black_box(quantizer.quantize(vec))
        })
    });
    
    // Quantize batch
    let batch_sizes = [10, 100, 1000];
    for batch_size in batch_sizes {
        if batch_size > vectors.len() {
            continue;
        }
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("quantize_batch", batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    let batch = &vectors[0..size];
                    let quantized: Vec<_> = batch.iter()
                        .map(|v| quantizer.quantize(v))
                        .collect();
                    black_box(quantized)
                })
            },
        );
    }
    
    // Hamming distance (on quantized vectors)
    let q1 = quantizer.quantize(&queries[0]);
    let q2 = quantizer.quantize(&vectors[0]);
    group.throughput(Throughput::Elements(1));
    group.bench_function("hamming_distance", |b| {
        b.iter(|| {
            let dist: u32 = q1.bits.iter()
                .zip(q2.bits.iter())
                .map(|(a, b)| (a ^ b).count_ones())
                .sum();
            black_box(dist)
        })
    });
    
    group.finish();
}

// Benchmark 3: K-Means Clustering (build bottleneck)
fn bench_kmeans_clustering(c: &mut Criterion) {
    let size = detect_dataset_size();
    let (vectors, _) = load_sift_data(size);
    
    let mut group = c.benchmark_group("kmeans_clustering");
    group.sample_size(10); // Fewer samples since k-means is slow
    group.measurement_time(Duration::from_secs(20));
    
    // Test different cluster counts (branching factor scenarios)
    let cluster_counts = if size <= 10_000 {
        vec![10, 50, 100]
    } else {
        vec![100]
    };
    
    for k in cluster_counts {
        if k > vectors.len() / 2 {
            continue;
        }
        
        group.bench_with_input(
            BenchmarkId::new("kmeans_fit", k),
            &k,
            |b, &num_clusters| {
                b.iter(|| {
                    let (kmeans, assignment) = KMeans::fit(
                        black_box(&vectors),
                        black_box(num_clusters),
                        black_box(DistanceMetric::L2),
                        black_box(20), // max_iterations
                    );
                    black_box((kmeans, assignment))
                })
            },
        );
    }
    
    group.finish();
}

// Benchmark 4: Tree Traversal (probe selection)
fn bench_tree_traversal(c: &mut Criterion) {
    let size = detect_dataset_size();
    let (vectors, queries) = load_sift_data(size);
    
    // Build index
    println!("Building index for tree traversal benchmark...");
    let index = ClusteredIndex::build(
        vectors.clone(),
        &format!("sift_microbench_{}.bin", size),
        100,  // branching_factor
        100,  // target_leaf_size
        DistanceMetric::L2,
        20,   // max_iterations
    ).unwrap();
    
    let mut group = c.benchmark_group("tree_traversal");
    group.throughput(Throughput::Elements(1));
    
    // Benchmark just the tree navigation (no leaf search)
    // This isolates the cost of computing distances to centroids
    let probe_counts = [1, 2, 5, 10];
    
    for probes in probe_counts {
        group.bench_with_input(
            BenchmarkId::new("find_top_nodes", probes),
            &probes,
            |b, &p| {
                b.iter(|| {
                    // Simulate what happens in search: find top nodes at each level
                    let query = &queries[black_box(0)];
                    let mut current_nodes = index.root_ids.clone();
                    let mut total_distances = 0;
                    
                    // Traverse tree
                    for _depth in 0..index.max_depth() {
                        if current_nodes.is_empty() {
                            break;
                        }
                        
                        // Compute distances to all current nodes
                        let mut node_distances: Vec<(usize, f32)> = current_nodes
                            .iter()
                            .map(|&node_id| {
                                let node = &index.nodes[node_id];
                                let dist: f32 = node.full_centroid.iter()
                                    .zip(query.iter())
                                    .map(|(a, b)| (a - b).powi(2))
                                    .sum();
                                (node_id, dist)
                            })
                            .collect();
                        
                        total_distances += node_distances.len();
                        
                        // Select top probes
                        node_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        let top_nodes: Vec<usize> = node_distances
                            .iter()
                            .take(p)
                            .map(|(id, _)| *id)
                            .collect();
                        
                        // Get children of internal nodes
                        current_nodes = top_nodes
                            .iter()
                            .filter(|&&node_id| !index.nodes[node_id].children.is_empty())
                            .flat_map(|&node_id| index.nodes[node_id].children.clone())
                            .collect();
                    }
                    
                    black_box(total_distances)
                })
            },
        );
    }
    
    group.finish();
}

// Benchmark 5: Leaf Search (final step)
fn bench_leaf_search(c: &mut Criterion) {
    let size = detect_dataset_size();
    let (vectors, queries) = load_sift_data(size);
    
    // Build index
    println!("Building index for leaf search benchmark...");
    let index = ClusteredIndex::build(
        vectors.clone(),
        &format!("sift_microbench_leaf_{}.bin", size),
        100,
        100,
        DistanceMetric::L2,
        20,
    ).unwrap();
    
    let mut group = c.benchmark_group("leaf_search");
    
    // Quantize query
    let query = &queries[0];
    let query_binary = index.quantizer.quantize(query);
    
    // Test searching different numbers of leaves
    let leaf_counts = [1, 5, 10, 25];
    
    for num_leaves in leaf_counts {
        if num_leaves > index.nodes.len() {
            continue;
        }
        
        // Find actual leaf nodes
        let leaf_ids: Vec<usize> = index.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.children.is_empty())
            .map(|(id, _)| id)
            .take(num_leaves)
            .collect();
        
        group.throughput(Throughput::Elements(num_leaves as u64));
        group.bench_with_input(
            BenchmarkId::new("search_leaves", num_leaves),
            &leaf_ids,
            |b, leaves| {
                b.iter(|| {
                    // Simulate leaf search: binary filter + rerank
                    let mut candidates = Vec::new();
                    
                    // Phase 1: Binary filtering
                    for &leaf_id in leaves {
                        let node = &index.nodes[leaf_id];
                        for &vec_idx in &node.vector_indices {
                            let binary_vec = &index.binary_vectors[vec_idx];
                            let hamming: u32 = query_binary.bits.iter()
                                .zip(binary_vec.bits.iter())
                                .map(|(a, b)| (a ^ b).count_ones())
                                .sum();
                            candidates.push((vec_idx, hamming));
                        }
                    }
                    
                    // Sort by Hamming distance
                    candidates.sort_by_key(|&(_, dist)| dist);
                    
                    // Phase 2: Rerank top-K with L2
                    let rerank_k = 100.min(candidates.len());
                    let mut results: Vec<(usize, f32)> = candidates
                        .iter()
                        .take(rerank_k)
                        .map(|&(idx, _)| {
                            let vec = index.full_vectors.get(idx);
                            let l2_dist: f32 = vec.iter()
                                .zip(query.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum();
                            (idx, l2_dist)
                        })
                        .collect();
                    
                    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    black_box(results)
                })
            },
        );
    }
    
    group.finish();
}

// Benchmark 6: Full Search (end-to-end)
fn bench_full_search(c: &mut Criterion) {
    let size = detect_dataset_size();
    let (vectors, queries) = load_sift_data(size);
    
    println!("Building index for full search benchmark...");
    let index = ClusteredIndex::build(
        vectors.clone(),
        &format!("sift_microbench_full_{}.bin", size),
        100,
        100,
        DistanceMetric::L2,
        20,
    ).unwrap();
    
    let mut group = c.benchmark_group("full_search");
    group.throughput(Throughput::Elements(1));
    
    // Test the configurations from main benchmark
    let configs = [
        ("low_latency", 2, 2),
        ("balanced", 5, 3),
        ("high_recall", 10, 5),
    ];
    
    for (name, probes, rerank_factor) in configs {
        group.bench_with_input(
            BenchmarkId::new("search", name),
            &(probes, rerank_factor),
            |b, &(p, r)| {
                b.iter(|| {
                    let query = &queries[black_box(0)];
                    let results = index.search(query, black_box(10), p, r);
                    black_box(results)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(3));
    targets = 
        bench_distance_calculation,
        bench_binary_quantization,
        bench_kmeans_clustering,
        bench_tree_traversal,
        bench_leaf_search,
        bench_full_search
}
criterion_main!(benches);
