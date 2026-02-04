#!/usr/bin/env python3
"""Visualize vector space search coverage from CSV and convert DOT files to PNG"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import subprocess
import os

def visualize_vector_space(csv_file, output_png):
    """Generate vector space visualization from CSV file"""
    print(f"Loading {csv_file}...")
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Define visualization properties - order matters for z-order!
    plot_order = ['other', 'searched_non_neighbor', 'missed_neighbor', 'found_neighbor', 'query']
    
    colors = {
        'other': 'lightgray',
        'searched_non_neighbor': 'lightblue', 
        'missed_neighbor': 'orange',
        'found_neighbor': 'green',
        'query': 'red'
    }
    
    sizes = {
        'other': 10,
        'searched_non_neighbor': 20,
        'missed_neighbor': 40,
        'found_neighbor': 40,
        'query': 200
    }
    
    alphas = {
        'other': 0.3,
        'searched_non_neighbor': 0.5,
        'missed_neighbor': 0.8,
        'found_neighbor': 0.8,
        'query': 1.0
    }
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot in order (background to foreground)
    for vtype in plot_order:
        data = df[df['type'] == vtype]
        if len(data) > 0:
            marker = '*' if vtype == 'query' else 'o'
            edgecolor = 'black' if vtype == 'query' else 'none'
            linewidth = 2 if vtype == 'query' else 0
            
            plt.scatter(
                data['x'], data['y'],
                c=colors[vtype],
                label=f"{vtype} ({len(data)})",
                alpha=alphas[vtype],
                s=sizes[vtype],
                marker=marker,
                edgecolors=edgecolor,
                linewidths=linewidth
            )
    
    plt.legend(loc='best', framealpha=0.9)
    plt.title('Vector Space Search Coverage', fontsize=14, fontweight='bold')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f'✓ Saved to {output_png}')
    plt.close()
    
    print(f"\nStatistics:")
    print(f"  Total vectors: {len(df)}")
    for vtype in plot_order:
        count = len(df[df['type'] == vtype])
        if count > 0:
            print(f"  {vtype}: {count}")

def convert_dot_to_png(dot_file, output_png):
    """Convert DOT file to PNG using graphviz"""
    print(f"Converting {dot_file} to PNG...")
    
    try:
        result = subprocess.run(
            ['dot', '-Tpng', dot_file, '-o', output_png],
            check=True,
            capture_output=True,
            text=True
        )
        print(f'✓ Saved to {output_png}')
    except FileNotFoundError:
        print(f'⚠️  dot command not found. Install graphviz to convert DOT files.')
        print(f'   On macOS: brew install graphviz')
        print(f'   On Linux: sudo apt-get install graphviz')
        return False
    except subprocess.CalledProcessError as e:
        print(f'⚠️  Failed to convert {dot_file}: {e.stderr}')
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Visualize vector space from CSV or convert tree DOT files to PNG'
    )
    parser.add_argument(
        '--csv',
        help='Input CSV file for vector space visualization'
    )
    parser.add_argument(
        '--dot',
        help='Input DOT file for tree structure visualization'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output PNG file'
    )
    
    args = parser.parse_args()
    
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found: {args.csv}")
            sys.exit(1)
        visualize_vector_space(args.csv, args.output)
    elif args.dot:
        if not os.path.exists(args.dot):
            print(f"Error: DOT file not found: {args.dot}")
            sys.exit(1)
        if not convert_dot_to_png(args.dot, args.output):
            sys.exit(1)
    else:
        print("Error: Must specify either --csv or --dot")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
