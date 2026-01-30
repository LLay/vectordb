#!/bin/bash
# Generate visualization images from observability demo output

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           Generating Visualization Images                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check for required tools
echo "Checking dependencies..."

# Check Python packages
if ! python3 -c "import pandas, matplotlib" 2>/dev/null; then
    echo "⚠️  Missing Python packages. Installing..."
    pip3 install pandas matplotlib
    echo ""
fi

# Check Graphviz
if ! command -v dot &> /dev/null; then
    echo "⚠️  Graphviz not found. Install with:"
    echo "    brew install graphviz  # macOS"
    echo "    sudo apt-get install graphviz  # Linux"
    echo ""
    exit 1
fi

echo "✓ All dependencies available"
echo ""

# Change to visualization directory
cd "$(dirname "$0")"

# Generate vector space visualization
echo "1. Generating vector_space.png..."
python3 visualize.py
echo ""

# Generate tree structure visualization
echo "2. Generating tree_structure.png..."
DOT_FILE=$(ls -t tree_structure*.dot 2>/dev/null | head -1)
if [ -n "$DOT_FILE" ]; then
    dot -Tpng "$DOT_FILE" -o tree_structure.png
    echo "   ✓ Rendered $DOT_FILE to tree_structure.png"
else
    echo "   ⚠️  No tree_structure*.dot file found"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  Visualization Complete!                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated files:"
ls -lh vector_space.png tree_structure.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Open with:"
echo "  open vector_space.png tree_structure.png  # macOS"
echo "  xdg-open vector_space.png tree_structure.png  # Linux"
