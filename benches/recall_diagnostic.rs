/// Recall Diagnostic Benchmark
/// 
/// Tracks recall loss at each stage of the search pipeline:
/// 1. Tree Traversal → Probe Coverage (% of true NNs in probed leaves)
/// 2. Quantized Filtering → Rerank Coverage (% of true NNs in top candidates)
/// 3. Reranking → Final Recall (% of true NNs in final results)
/// 
/// This identifies WHERE nearest neighbors are lost:
/// - Low probe coverage → Tree structure problem
/// - High probe coverage, low rerank coverage → Quantization problem
/// - High rerank coverage, low final recall → Rerank factor too small
/// 
/// Usage:
///   cargo bench --bench recall_diagnostic

#[path = "../datasets/sift/mod.rs"]
mod sift;

use vectordb::{ClusteredIndex, ClusteredIndexWithRaBitQ, DistanceMetric};
use std::collections::HashSet;
use std::io::Write;

/// Configuration for diagnostic tests
struct DiagnosticConfig {
    dataset_size: usize,
    branching_factor: usize,
    target_leaf_size: usize,
    probes_per_level: usize,
    rerank_factor: usize,
    k: usize,
    num_queries: usize,
}

/// Metrics tracked at each stage
#[derive(Debug, Clone)]
struct StageMetrics {
    /// Total vectors in dataset
    total_vectors: usize,
    /// Number of leaves probed
    leaves_probed: usize,
    /// Total leaves in tree
    total_leaves: usize,
    /// Vectors in probed leaves (candidate pool size)
    vectors_in_probed_leaves: usize,
    /// True NNs found in probed leaves (probe coverage)
    true_nns_in_probed_leaves: usize,
    /// Vectors passed to reranking
    vectors_reranked: usize,
    /// True NNs in rerank candidates (rerank coverage)
    true_nns_in_rerank_candidates: usize,
    /// Final recall
    true_nns_in_final_results: usize,
}

impl StageMetrics {
    fn probe_coverage(&self, k: usize) -> f64 {
        self.true_nns_in_probed_leaves as f64 / k as f64
    }
    
    fn rerank_coverage(&self, k: usize) -> f64 {
        self.true_nns_in_rerank_candidates as f64 / k as f64
    }
    
    fn final_recall(&self, k: usize) -> f64 {
        self.true_nns_in_final_results as f64 / k as f64
    }
    
    fn probe_efficiency(&self) -> f64 {
        self.vectors_in_probed_leaves as f64 / self.total_vectors as f64
    }
}

fn compute_ground_truth(
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
            (idx, dist.sqrt())
        })
        .collect();
    
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.iter().take(k).map(|(idx, _)| *idx).collect()
}

fn diagnose_binary_search(
    index: &ClusteredIndex,
    query: &[f32],
    ground_truth: &[usize],
    config: &DiagnosticConfig,
) -> StageMetrics {
    let k = config.k;
    let gt_set: HashSet<usize> = ground_truth.iter().copied().collect();
    
    // Perform search with statistics
    let (results, stats) = index.search_with_stats(
        query,
        k,
        config.probes_per_level,
        config.rerank_factor,
    );
    
    // Stage 1: Check probe coverage (are true NNs in probed leaves?)
    let mut vectors_in_probed_leaves = HashSet::new();
    for &leaf_id in &stats.leaves_searched_ids {
        let leaf = &index.nodes[leaf_id];
        for &vec_idx in &leaf.vector_indices {
            vectors_in_probed_leaves.insert(vec_idx);
        }
    }
    
    let true_nns_in_probed = gt_set
        .iter()
        .filter(|idx| vectors_in_probed_leaves.contains(idx))
        .count();
    
    // Stage 2: Estimate rerank coverage
    // We reranked `stats.vectors_reranked_full` vectors
    // Approximate which ones by taking top rerank_count from probed leaves
    let rerank_count = (k * config.rerank_factor).min(stats.vectors_scanned_binary);
    
    // Stage 3: Final recall
    let result_set: HashSet<usize> = results.iter().map(|(idx, _)| *idx).collect();
    let true_nns_in_final = gt_set.intersection(&result_set).count();
    
    StageMetrics {
        total_vectors: stats.total_vectors,
        leaves_probed: stats.leaves_searched,
        total_leaves: stats.total_leaves,
        vectors_in_probed_leaves: vectors_in_probed_leaves.len(),
        true_nns_in_probed_leaves: true_nns_in_probed,
        vectors_reranked: stats.vectors_reranked_full,
        true_nns_in_rerank_candidates: true_nns_in_probed.min(rerank_count), // Upper bound estimate
        true_nns_in_final_results: true_nns_in_final,
    }
}

fn diagnose_rabitq_search(
    index: &ClusteredIndexWithRaBitQ,
    query: &[f32],
    ground_truth: &[usize],
    config: &DiagnosticConfig,
) -> StageMetrics {
    let k = config.k;
    let gt_set: HashSet<usize> = ground_truth.iter().copied().collect();
    
    // Perform search with statistics
    let (results, stats) = index.search_with_stats(
        query,
        k,
        config.probes_per_level,
        config.rerank_factor,
    );
    
    // Stage 1: Check probe coverage (are true NNs in probed leaves?)
    let mut vectors_in_probed_leaves = HashSet::new();
    for &leaf_id in &stats.leaves_searched_ids {
        let leaf = &index.nodes[leaf_id];
        for &vec_idx in &leaf.vector_indices {
            vectors_in_probed_leaves.insert(vec_idx);
        }
    }
    
    let true_nns_in_probed = gt_set
        .iter()
        .filter(|idx| vectors_in_probed_leaves.contains(idx))
        .count();
    
    // Stage 2: Estimate rerank coverage
    let rerank_count = (k * config.rerank_factor).min(stats.vectors_scanned_rabitq);
    
    // Stage 3: Final recall
    let result_set: HashSet<usize> = results.iter().map(|(idx, _)| *idx).collect();
    let true_nns_in_final = gt_set.intersection(&result_set).count();
    
    StageMetrics {
        total_vectors: stats.total_vectors,
        leaves_probed: stats.leaves_searched,
        total_leaves: stats.total_leaves,
        vectors_in_probed_leaves: vectors_in_probed_leaves.len(),
        true_nns_in_probed_leaves: true_nns_in_probed,
        vectors_reranked: stats.vectors_reranked_full,
        true_nns_in_rerank_candidates: true_nns_in_probed.min(rerank_count), // Upper bound estimate
        true_nns_in_final_results: true_nns_in_final,
    }
}

fn print_diagnostic_summary(
    binary_metrics: &[StageMetrics],
    rabitq_metrics: &[StageMetrics],
    config: &DiagnosticConfig,
) {
    let k = config.k;
    
    // Average across all queries
    let avg_binary = average_metrics(binary_metrics);
    let avg_rabitq = average_metrics(rabitq_metrics);
    
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║             Recall Diagnostic Summary                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Stage 1: Tree Traversal (Probe Coverage)                   │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    println!("                       Binary        RaBitQ      Difference");
    println!("─────────────────────────────────────────────────────────────");
    println!("Leaves probed:         {:.1} / {}    {:.1} / {}    ",
        avg_binary.leaves_probed as f64, avg_binary.total_leaves,
        avg_rabitq.leaves_probed as f64, avg_rabitq.total_leaves);
    println!("Vectors in leaves:     {:.0}         {:.0}         {:+.0}",
        avg_binary.vectors_in_probed_leaves as f64,
        avg_rabitq.vectors_in_probed_leaves as f64,
        avg_rabitq.vectors_in_probed_leaves as f64 - avg_binary.vectors_in_probed_leaves as f64);
    println!("Probe efficiency:      {:.1}%        {:.1}%       {:+.1}%",
        avg_binary.probe_efficiency() * 100.0,
        avg_rabitq.probe_efficiency() * 100.0,
        (avg_rabitq.probe_efficiency() - avg_binary.probe_efficiency()) * 100.0);
    println!();
    println!("True NNs in leaves:    {:.1} / {}     {:.1} / {}      ",
        avg_binary.true_nns_in_probed_leaves as f64, k,
        avg_rabitq.true_nns_in_probed_leaves as f64, k);
    println!("Probe coverage:        {:.1}%        {:.1}%       {:+.1}%",
        avg_binary.probe_coverage(k) * 100.0,
        avg_rabitq.probe_coverage(k) * 100.0,
        (avg_rabitq.probe_coverage(k) - avg_binary.probe_coverage(k)) * 100.0);
    println!();
    
    if avg_binary.probe_coverage(k) < 0.9 || avg_rabitq.probe_coverage(k) < 0.9 {
        println!("⚠️  LOW PROBE COVERAGE DETECTED!");
        println!("    → Problem: Tree structure or probes_per_level too small");
        println!("    → Solution: Increase probes_per_level or improve tree quality");
        println!();
    }
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Stage 2: Quantized Filtering (Rerank Coverage)             │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    println!("                       Binary        RaBitQ      Difference");
    println!("─────────────────────────────────────────────────────────────");
    println!("Vectors reranked:      {:.0}         {:.0}         {:+.0}",
        avg_binary.vectors_reranked as f64,
        avg_rabitq.vectors_reranked as f64,
        avg_rabitq.vectors_reranked as f64 - avg_binary.vectors_reranked as f64);
    println!("True NNs in rerank:    {:.1} / {}     {:.1} / {}      ",
        avg_binary.true_nns_in_rerank_candidates as f64, k,
        avg_rabitq.true_nns_in_rerank_candidates as f64, k);
    println!("Rerank coverage:       {:.1}%        {:.1}%       {:+.1}%",
        avg_binary.rerank_coverage(k) * 100.0,
        avg_rabitq.rerank_coverage(k) * 100.0,
        (avg_rabitq.rerank_coverage(k) - avg_binary.rerank_coverage(k)) * 100.0);
    println!();
    
    // Calculate loss from probe → rerank
    let binary_probe_to_rerank_loss = avg_binary.probe_coverage(k) - avg_binary.rerank_coverage(k);
    let rabitq_probe_to_rerank_loss = avg_rabitq.probe_coverage(k) - avg_rabitq.rerank_coverage(k);
    
    println!("Loss (probe → rerank): {:.1}%        {:.1}%",
        binary_probe_to_rerank_loss * 100.0,
        rabitq_probe_to_rerank_loss * 100.0);
    println!();
    
    if binary_probe_to_rerank_loss > 0.1 {
        println!("⚠️  BINARY: High loss in quantized filtering!");
        println!("    → Problem: Binary quantization ranking poorly");
        println!("    → Solution: Increase rerank_factor");
        println!();
    }
    
    if rabitq_probe_to_rerank_loss > 0.1 {
        println!("⚠️  RABITQ: High loss in quantized filtering!");
        println!("    → Problem: RaBitQ ranking poorly or rerank_factor too small");
        println!("    → Solution: Increase rerank_factor or check quantization quality");
        println!();
    }
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Stage 3: Final Recall                                       │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    println!("                       Binary        RaBitQ      Difference");
    println!("─────────────────────────────────────────────────────────────");
    println!("True NNs found:        {:.1} / {}     {:.1} / {}      ",
        avg_binary.true_nns_in_final_results as f64, k,
        avg_rabitq.true_nns_in_final_results as f64, k);
    println!("Final recall:          {:.1}%        {:.1}%       {:+.1}%",
        avg_binary.final_recall(k) * 100.0,
        avg_rabitq.final_recall(k) * 100.0,
        (avg_rabitq.final_recall(k) - avg_binary.final_recall(k)) * 100.0);
    println!();
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Recall Loss Breakdown                                       │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    println!("Binary Quantization:");
    println!("  Start:         100.0%");
    println!("  → Probe:       {:.1}%  (loss: {:.1}%)",
        avg_binary.probe_coverage(k) * 100.0,
        (1.0 - avg_binary.probe_coverage(k)) * 100.0);
    println!("  → Rerank:      {:.1}%  (loss: {:.1}%)",
        avg_binary.rerank_coverage(k) * 100.0,
        binary_probe_to_rerank_loss * 100.0);
    println!("  → Final:       {:.1}%  (loss: {:.1}%)",
        avg_binary.final_recall(k) * 100.0,
        (avg_binary.rerank_coverage(k) - avg_binary.final_recall(k)) * 100.0);
    println!();
    
    println!("RaBitQ Quantization:");
    println!("  Start:         100.0%");
    println!("  → Probe:       {:.1}%  (loss: {:.1}%)",
        avg_rabitq.probe_coverage(k) * 100.0,
        (1.0 - avg_rabitq.probe_coverage(k)) * 100.0);
    println!("  → Rerank:      {:.1}%  (loss: {:.1}%)",
        avg_rabitq.rerank_coverage(k) * 100.0,
        rabitq_probe_to_rerank_loss * 100.0);
    println!("  → Final:       {:.1}%  (loss: {:.1}%)",
        avg_rabitq.final_recall(k) * 100.0,
        (avg_rabitq.rerank_coverage(k) - avg_rabitq.final_recall(k)) * 100.0);
    println!();
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Diagnosis                                                    │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    let binary_main_loss = if (1.0 - avg_binary.probe_coverage(k)) > 0.2 {
        "Tree Traversal"
    } else if binary_probe_to_rerank_loss > 0.2 {
        "Quantized Filtering"
    } else {
        "Minimal Loss"
    };
    
    let rabitq_main_loss = if (1.0 - avg_rabitq.probe_coverage(k)) > 0.2 {
        "Tree Traversal"
    } else if rabitq_probe_to_rerank_loss > 0.2 {
        "Quantized Filtering"
    } else {
        "Minimal Loss"
    };
    
    println!("Binary: Primary recall loss in → {}", binary_main_loss);
    println!("RaBitQ: Primary recall loss in → {}", rabitq_main_loss);
    println!();
}

fn average_metrics(metrics: &[StageMetrics]) -> StageMetrics {
    let n = metrics.len() as f64;
    StageMetrics {
        total_vectors: metrics[0].total_vectors,
        leaves_probed: (metrics.iter().map(|m| m.leaves_probed).sum::<usize>() as f64 / n) as usize,
        total_leaves: metrics[0].total_leaves,
        vectors_in_probed_leaves: (metrics.iter().map(|m| m.vectors_in_probed_leaves).sum::<usize>() as f64 / n) as usize,
        true_nns_in_probed_leaves: (metrics.iter().map(|m| m.true_nns_in_probed_leaves).sum::<usize>() as f64 / n) as usize,
        vectors_reranked: (metrics.iter().map(|m| m.vectors_reranked).sum::<usize>() as f64 / n) as usize,
        true_nns_in_rerank_candidates: (metrics.iter().map(|m| m.true_nns_in_rerank_candidates).sum::<usize>() as f64 / n) as usize,
        true_nns_in_final_results: (metrics.iter().map(|m| m.true_nns_in_final_results).sum::<usize>() as f64 / n) as usize,
    }
}

fn main() {
    // Read dataset size from environment variable (default: 10K)
    let dataset_size = std::env::var("SIFT_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10_000);
    
    let config = DiagnosticConfig {
        dataset_size,
        branching_factor: 100,
        target_leaf_size: 100,
        probes_per_level: 5,
        rerank_factor: 10,
        k: 10,
        num_queries: 100,
    };
    
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║         Recall Diagnostic Benchmark                         ║");
    println!("║     Finding WHERE nearest neighbors are lost                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    println!("Configuration:");
    println!("  Dataset size: {}", config.dataset_size);
    println!("  k: {}", config.k);
    println!("  Probes per level: {}", config.probes_per_level);
    println!("  Rerank factor: {}", config.rerank_factor);
    println!("  Queries to test: {}", config.num_queries);
    
    if config.dataset_size >= 1_000_000 {
        println!("\n⏱️  Note: 1M vectors will take ~5-10 minutes to build indices and run diagnostics");
    }
    println!();
    
    // Load dataset
    print!("Loading SIFT vectors...");
    std::io::stdout().flush().unwrap();
    
    // Try to load from subset file first, fallback to full dataset
    let base_file = if config.dataset_size < 1_000_000 {
        let subset_file = format!("datasets/sift/data/sift_base_{}.fvecs", config.dataset_size);
        if std::path::Path::new(&subset_file).exists() {
            subset_file
        } else {
            "datasets/sift/data/sift_base.fvecs".to_string()
        }
    } else {
        "datasets/sift/data/sift_base.fvecs".to_string()
    };
    
    let (base_vectors, _dim) = sift::loader::read_fvecs(&base_file)
        .expect("Failed to load base vectors");
    
    let base_vectors: Vec<Vec<f32>> = base_vectors
        .into_iter()
        .take(config.dataset_size)
        .collect();
    
    let (query_vectors, _) = sift::loader::read_fvecs(
        "datasets/sift/data/sift_query.fvecs"
    ).expect("Failed to load query vectors");
    
    let query_vectors: Vec<Vec<f32>> = query_vectors
        .into_iter()
        .take(config.num_queries)
        .collect();
    
    println!(" ✓");
    println!("  {} vectors, {} queries", base_vectors.len(), query_vectors.len());
    println!();
    
    // Build indices
    print!("[1/2] Building Binary index...");
    std::io::stdout().flush().unwrap();
    let binary_index = ClusteredIndex::build(
        base_vectors.clone(),
        "recall_diagnostic_binary.bin",
        config.branching_factor,
        config.target_leaf_size,
        DistanceMetric::L2,
        10,
    ).expect("Failed to build binary index");
    println!(" ✓");
    
    print!("[2/2] Building RaBitQ index...");
    std::io::stdout().flush().unwrap();
    let rabitq_index = ClusteredIndexWithRaBitQ::build(
        base_vectors.clone(),
        "recall_diagnostic_rabitq.bin",
        config.branching_factor,
        config.target_leaf_size,
        DistanceMetric::L2,
        10,
    ).expect("Failed to build rabitq index");
    println!(" ✓");
    println!();
    
    // Run diagnostics
    println!("Running diagnostics on {} queries...", config.num_queries);
    println!();
    
    let mut binary_metrics = Vec::new();
    let mut rabitq_metrics = Vec::new();
    
    for (i, query) in query_vectors.iter().enumerate() {
        if (i + 1) % 20 == 0 {
            print!("\r  Progress: {} / {} queries", i + 1, config.num_queries);
            std::io::stdout().flush().unwrap();
        }
        
        // Compute ground truth
        let ground_truth = compute_ground_truth(&base_vectors, query, config.k);
        
        // Diagnose binary
        let binary_stage = diagnose_binary_search(
            &binary_index,
            query,
            &ground_truth,
            &config,
        );
        binary_metrics.push(binary_stage);
        
        // Diagnose rabitq
        let rabitq_stage = diagnose_rabitq_search(
            &rabitq_index,
            query,
            &ground_truth,
            &config,
        );
        rabitq_metrics.push(rabitq_stage);
    }
    
    println!("\r  Progress: {} / {} queries ✓", config.num_queries, config.num_queries);
    
    // Print summary
    print_diagnostic_summary(&binary_metrics, &rabitq_metrics, &config);
    
    // Cleanup
    std::fs::remove_file("recall_diagnostic_binary.bin").ok();
    std::fs::remove_file("recall_diagnostic_rabitq.bin").ok();
}
