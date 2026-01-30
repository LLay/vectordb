#!/usr/bin/env python3
"""Visualize vector space search coverage from CSV"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import glob

# Find the most recent vector_space CSV file
csv_files = glob.glob('vector_space*.csv')
if not csv_files:
    print("Error: No vector_space*.csv files found")
    sys.exit(1)

csv_file = sorted(csv_files)[-1]  # Use the most recent
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
plt.savefig('vector_space.png', dpi=150, bbox_inches='tight')
print('✓ Saved to vector_space.png')
plt.close()

print(f"\nStatistics:")
print(f"  Total vectors: {len(df)}")
for vtype in plot_order:
    count = len(df[df['type'] == vtype])
    if count > 0:
        print(f"  {vtype}: {count}")
