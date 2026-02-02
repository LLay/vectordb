#!/bin/bash
# Download SIFT-1M dataset for VectorDBBench compatibility

set -e  # Exit on error

SIFT_DIR="datasets/sift"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           Downloading SIFT-1M Dataset                         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Create directory
mkdir -p "$SIFT_DIR"
cd "$SIFT_DIR"

# Download SIFT dataset
echo "Downloading SIFT-1M (256 MB)..."
if [ ! -f "sift.tar.gz" ]; then
    wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    echo "✓ Download complete"
else
    echo "✓ Already downloaded"
fi

# Extract
echo ""
echo "Extracting..."
if [ ! -f "sift/sift_base.fvecs" ]; then
    tar -xzf sift.tar.gz
    echo "✓ Extraction complete"
else
    echo "✓ Already extracted"
fi

echo ""
echo "Files extracted:"
ls -lh sift/

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           Dataset Information                                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Base vectors:    datasets/sift/data/sift_base.fvecs (1,000,000 vectors)"
echo "Queries:         datasets/sift/data/sift_query.fvecs (10,000 queries)"
echo "Ground truth:    datasets/sift/data/sift_groundtruth.ivecs (100-NN for each query)"
echo ""
echo "Files are in standard .fvecs/.ivecs format and will be loaded directly."
echo ""
echo "Next step:"
echo "  cargo bench --bench sift_benchmark"
echo ""
