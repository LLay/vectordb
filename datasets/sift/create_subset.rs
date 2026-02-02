/// Create subset datasets from SIFT-1M for fast development iteration
/// 
/// Usage:
///   cargo run --release --bin create_sift_subset -- 10000
///   cargo run --release --bin create_sift_subset -- 100000

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <num_vectors>", args[0]);
        eprintln!("Example: {} 10000", args[0]);
        eprintln!("Creates subset with first N vectors from SIFT-1M");
        std::process::exit(1);
    }
    
    let subset_size: usize = args[1].parse()
        .expect("Invalid number");
    
    if subset_size > 1_000_000 {
        eprintln!("Error: subset_size cannot exceed 1,000,000");
        std::process::exit(1);
    }
    
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           SIFT Subset Creator                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Creating subset with {} vectors...", subset_size);
    println!();
    
    // Create base vectors subset
    println!("Processing base vectors...");
    let count = copy_fvecs_subset(
        "datasets/sift/data/sift_base.fvecs",
        &format!("datasets/sift/data/sift_base_{}.fvecs", subset_size),
        subset_size,
    )?;
    println!("  ✓ Created {} vectors", count);
    
    // Queries stay the same (use all 10K)
    println!();
    println!("Note: Using full query set (10K queries)");
    println!("      Ground truth will be recomputed for subset");
    println!();
    
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           Subset Created!                                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Files created:");
    println!("  • datasets/sift/data/sift_base_{}.fvecs", subset_size);
    println!();
    println!("To benchmark:");
    println!("  cargo bench --bench sift_benchmark -- {}", subset_size);
    println!();
    println!("Note: Ground truth is computed at benchmark time");
    println!("      (brute force on subset is fast enough)");
    println!();
    
    Ok(())
}

fn copy_fvecs_subset(
    input_path: &str,
    output_path: &str,
    max_vectors: usize,
) -> std::io::Result<usize> {
    let input = File::open(input_path)?;
    let mut reader = BufReader::new(input);
    
    let output = File::create(output_path)?;
    let mut writer = BufWriter::new(output);
    
    let mut count = 0;
    
    loop {
        if count >= max_vectors {
            break;
        }
        
        // Read dimension
        let mut dim_bytes = [0u8; 4];
        match reader.read_exact(&mut dim_bytes) {
            Ok(_) => {},
            Err(_) => break,  // EOF
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        
        // Write dimension
        writer.write_all(&dim_bytes)?;
        
        // Read and write vector
        let mut vec_bytes = vec![0u8; dim * 4];
        reader.read_exact(&mut vec_bytes)?;
        writer.write_all(&vec_bytes)?;
        
        count += 1;
        
        if count % 10_000 == 0 {
            print!("\r  Progress: {} / {} vectors", count, max_vectors);
            std::io::stdout().flush()?;
        }
    }
    
    if count > 0 {
        println!("\r  Progress: {} / {} vectors", count, max_vectors);
    }
    
    writer.flush()?;
    Ok(count)
}
