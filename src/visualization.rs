//! Visualization utilities for understanding vector space and search behavior

use crate::index::hierarchical::{ClusteredIndex, SearchStats};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Simple 2D projection using first two dimensions (fast, no computation)
pub fn reduce_to_2d_simple(vectors: &[Vec<f32>]) -> Vec<(f32, f32)> {
    vectors.iter().map(|v| (v[0], v[1])).collect()
}

/// Random projection to 2D (fast, preserves distances reasonably well)
pub fn reduce_to_2d_random(vectors: &[Vec<f32>]) -> Vec<(f32, f32)> {
    use rand::Rng;
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let d = vectors[0].len();
    let mut rng = rand::thread_rng();
    
    // Generate two random projection vectors
    let proj1: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let proj2: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    // Normalize
    let norm1 = proj1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2 = proj2.iter().map(|x| x * x).sum::<f32>().sqrt();
    let proj1: Vec<f32> = proj1.iter().map(|x| x / norm1).collect();
    let proj2: Vec<f32> = proj2.iter().map(|x| x / norm2).collect();
    
    // Project all vectors
    vectors.iter().map(|v| {
        let x = v.iter().zip(&proj1).map(|(a, b)| a * b).sum();
        let y = v.iter().zip(&proj2).map(|(a, b)| a * b).sum();
        (x, y)
    }).collect()
}

/// Generate vector space visualization CSV
/// Includes: vector positions, whether they're in ground truth, whether they were searched
pub fn visualize_vector_space(
    index: &ClusteredIndex,
    all_vectors: &[Vec<f32>],
    query: &[f32],
    ground_truth_indices: &[usize],
    search_stats: &SearchStats,
    output_file: &str,
) -> std::io::Result<()> {
    println!("Generating vector space visualization...");
    
    // Project to 2D
    let mut all_vecs_with_query = all_vectors.to_vec();
    all_vecs_with_query.push(query.to_vec());
    
    println!("  Computing 2D projection using random projection...");
    let projected = reduce_to_2d_random(&all_vecs_with_query);
    
    let gt_set: HashSet<usize> = ground_truth_indices.iter().copied().collect();
    let searched_set: HashSet<usize> = search_stats.leaves_searched_ids
        .iter()
        .flat_map(|&leaf_id| {
            index.nodes[leaf_id].vector_indices.clone()
        })
        .collect();
    
    // Write CSV
    let file = File::create(output_file)?;
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "x,y,type,in_ground_truth,was_searched,leaf_id")?;
    
    // Write query
    let (qx, qy) = projected[projected.len() - 1];
    writeln!(writer, "{},{},query,0,0,-1", qx, qy)?;
    
    // Write vectors
    for i in 0..all_vectors.len() {
        let (x, y) = projected[i];
        let in_gt = if gt_set.contains(&i) { 1 } else { 0 };
        let searched = if searched_set.contains(&i) { 1 } else { 0 };
        
        // Find which leaf this vector belongs to
        let leaf_id = index.nodes.iter()
            .find(|node| node.vector_indices.contains(&i))
            .map(|node| node.id as i32)
            .unwrap_or(-1);
        
        let vtype = match (in_gt, searched) {
            (1, 1) => "found_neighbor",      // True positive
            (1, 0) => "missed_neighbor",     // False negative
            (0, 1) => "searched_non_neighbor", // False positive area
            (0, 0) => "other",
            _ => "other",  // Catch-all for any other values
        };
        
        writeln!(writer, "{},{},{},{},{},{}", x, y, vtype, in_gt, searched, leaf_id)?;
    }
    
    writer.flush()?;
    println!("  ✓ Saved to {}", output_file);
 
    Ok(())
}

/// Generate tree structure visualization in Graphviz DOT format
pub fn visualize_tree_structure(
    index: &ClusteredIndex,
    search_stats: &SearchStats,
    ground_truth_indices: &[usize],
    output_file: &str,
) -> std::io::Result<()> {
    println!("Generating tree structure visualization...");
    
    let searched_leaves: HashSet<usize> = search_stats.leaves_searched_ids.iter().copied().collect();
    
    // Find which leaves contain ground truth vectors
    let mut gt_leaves = HashSet::new();
    for &gt_idx in ground_truth_indices {
        for node in &index.nodes {
            if !node.children.is_empty() {
                continue; // Skip internal nodes
            }
            if node.vector_indices.contains(&gt_idx) {
                gt_leaves.insert(node.id);
            }
        }
    }
    
    let file = File::create(output_file)?;
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "digraph IndexTree {{")?;
    writeln!(writer, "  rankdir=TB;")?;
    writeln!(writer, "  node [shape=box, style=filled];")?;
    writeln!(writer, "")?;
    
    // Write nodes
    for node in &index.nodes {
        let is_leaf = node.children.is_empty();
        let was_searched = searched_leaves.contains(&node.id);
        let has_gt = gt_leaves.contains(&node.id);
        
        let (color, style) = match (is_leaf, was_searched, has_gt) {
            (true, true, true) => ("lightgreen", "filled,bold"),   // Searched leaf with GT
            (true, true, false) => ("lightblue", "filled"),        // Searched leaf without GT
            (true, false, true) => ("orange", "filled,bold"),      // Missed leaf with GT
            (true, false, false) => ("white", "filled"),           // Unexamined leaf
            (false, _, _) => ("lightgray", "filled"),              // Internal node
        };
        
        let label = if is_leaf {
            format!("L{} ({}v)", node.id, node.vector_indices.len())
        } else {
            format!("N{} ({}c)", node.id, node.children.len())
        };
        
        writeln!(writer, "  n{} [label=\"{}\", fillcolor={}, style=\"{}\"];", 
                 node.id, label, color, style)?;
    }
    
    writeln!(writer, "")?;
    
    // Write edges
    for node in &index.nodes {
        for &child_id in &node.children {
            writeln!(writer, "  n{} -> n{};", node.id, child_id)?;
        }
    }
    
    writeln!(writer, "")?;
    writeln!(writer, "  // Legend")?;
    writeln!(writer, "  subgraph cluster_legend {{")?;
    writeln!(writer, "    label=\"Legend\";")?;
    writeln!(writer, "    legend_searched_gt [label=\"Searched + Has GT\", fillcolor=lightgreen, style=\"filled,bold\"];")?;
    writeln!(writer, "    legend_searched [label=\"Searched\", fillcolor=lightblue, style=filled];")?;
    writeln!(writer, "    legend_missed_gt [label=\"Missed GT\", fillcolor=orange, style=\"filled,bold\"];")?;
    writeln!(writer, "    legend_leaf [label=\"Other Leaf\", fillcolor=white, style=filled];")?;
    writeln!(writer, "  }}")?;
    
    writeln!(writer, "}}")?;
    writer.flush()?;
    
    println!("  ✓ Saved to {}", output_file);
    println!("  Render with: dot -Tpng {} -o tree_structure.png", output_file);
    println!("  Or view online: https://dreampuf.github.io/GraphvizOnline/ (paste file contents)");
    
    Ok(())
}

/// Generate a coverage report comparing searched area vs ground truth location
pub fn print_coverage_report(
    index: &ClusteredIndex,
    ground_truth_indices: &[usize],
    search_stats: &SearchStats,
) {
    let searched_leaves: HashSet<usize> = search_stats.leaves_searched_ids.iter().copied().collect();
    
    // Find which leaves contain ground truth vectors
    let mut gt_leaves = HashSet::new();
    let mut gt_distribution: Vec<(usize, usize)> = Vec::new(); // (leaf_id, count)
    
    for &gt_idx in ground_truth_indices {
        for node in &index.nodes {
            if !node.children.is_empty() {
                continue; // Skip internal nodes
            }
            if node.vector_indices.contains(&gt_idx) {
                gt_leaves.insert(node.id);
                
                // Update count
                if let Some((_, count)) = gt_distribution.iter_mut().find(|(id, _)| *id == node.id) {
                    *count += 1;
                } else {
                    gt_distribution.push((node.id, 1));
                }
            }
        }
    }
    
    let overlap = searched_leaves.intersection(&gt_leaves).count();
    let searched_only = searched_leaves.difference(&gt_leaves).count();
    let missed = gt_leaves.difference(&searched_leaves).count();
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              Search Coverage Analysis                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Ground Truth Distribution:");
    println!("  {} nearest neighbors spread across {} leaves", ground_truth_indices.len(), gt_leaves.len());
    
    gt_distribution.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    println!("  Top leaves with GT vectors:");
    for (leaf_id, count) in gt_distribution.iter().take(10) {
        let was_searched = searched_leaves.contains(leaf_id);
        let marker = if was_searched { "✓" } else { "✗" };
        println!("    {} Leaf {} contains {} GT vectors", marker, leaf_id, count);
    }
    
    println!();
    println!("Search Coverage:");
    println!("  Searched {} leaves total", searched_leaves.len());
    println!("  {} leaves contain GT vectors ({}%)", 
             overlap, (overlap as f64 / gt_leaves.len() as f64 * 100.0) as u32);
    println!("  {} leaves searched but no GT (wasted effort)", searched_only);
    println!("  {} leaves with GT were MISSED", missed);
    
    let coverage = overlap as f64 / gt_leaves.len() as f64 * 100.0;
    println!();
    if coverage < 50.0 {
        println!("⚠️  Low coverage! Search is looking in wrong parts of the tree.");
    } else if coverage < 80.0 {
        println!("⚠️  Moderate coverage. Consider increasing probes.");
    } else {
        println!("✓ Good coverage!");
    }
}
