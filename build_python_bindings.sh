#!/bin/bash
# Build Python bindings for Rust VectorDB using maturin
#
# Usage:
#   ./build_python_bindings.sh          # Development build
#   ./build_python_bindings.sh release  # Release build (optimized)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Building Rust VectorDB Python Bindings                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "❌ maturin not found. Installing..."
    pip install maturin
    echo "✅ maturin installed"
    echo ""
fi

# Determine build mode
BUILD_MODE="${1:-dev}"

if [ "$BUILD_MODE" = "release" ]; then
    echo "🔨 Building RELEASE version (optimized, slower compile)..."
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release --features python
else
    echo "🔨 Building DEVELOPMENT version (fast compile, debug symbols)..."
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --features python
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Testing Python Bindings                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Test the bindings
python3 << 'EOF'
try:
    from vectordb import PyVectorDB
    import sys
    
    print("✅ Python bindings imported successfully!")
    print(f"   Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Quick functionality test
    print("\n🧪 Running quick test...")
    db = PyVectorDB(dimension=128, branching_factor=10, target_leaf_size=50)
    print(f"   Created: {db}")
    
    # Test insert
    vectors = [[float(i+j) for j in range(128)] for i in range(100)]
    count = db.insert_embeddings(vectors)
    print(f"   Inserted: {count} vectors")
    
    # Test optimize
    build_time = db.optimize()
    print(f"   Optimized: {build_time:.3f}s")
    
    # Test search
    query = [1.0] * 128
    results = db.search(query, k=5, probes=5, rerank_factor=3)
    print(f"   Searched: Found {len(results)} neighbors")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║         Build Complete!                                       ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📦 Python package 'vectordb' is now installed"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Test standalone: python vectordbbench_client/rust_vectordb.py"
    echo "   2. Integrate with VectorDBBench (see vectordbbench_client/README.md)"
    echo ""
else
    echo ""
    echo "❌ Build or test failed. Please check the errors above."
    exit 1
fi
