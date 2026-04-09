#!/bin/bash
###############################################################################
# GPU Image Deduplicator - Run Script
#
# Generates test data (if needed), builds, and runs the deduplicator.
###############################################################################

set -e

DATA_DIR="data/generated"
SAMPLE_DIR="sample_images"
OUTPUT_FILE="output/results.txt"

echo "================================================"
echo " GPU Image Deduplicator"
echo "================================================"

# Step 1: Generate test dataset if it doesn't exist
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo ""
    echo "[1/3] Generating test dataset..."
    if [ -z "$(ls -A $SAMPLE_DIR 2>/dev/null)" ]; then
        echo "No sample images found in $SAMPLE_DIR/"
        echo "Please add some .jpg or .png images to $SAMPLE_DIR/ first,"
        echo "or provide your own dataset directory."
        echo ""
        echo "Quick start: download a few images and place them in $SAMPLE_DIR/"
        exit 1
    fi
    python3 generate_dataset.py "$SAMPLE_DIR" "$DATA_DIR" --variants 20
else
    echo ""
    echo "[1/3] Test dataset already exists at $DATA_DIR/"
    echo "      $(find $DATA_DIR -type f | wc -l | tr -d ' ') images found"
fi

# Step 2: Build
echo ""
echo "[2/3] Building..."
make -j

# Step 3: Run
echo ""
echo "[3/3] Running deduplicator..."
mkdir -p output
./bin/gpu_dedup -v -t 0.97 -o "$OUTPUT_FILE" "$DATA_DIR"

echo ""
echo "================================================"
echo " Done! Results saved to $OUTPUT_FILE"
echo "================================================"
cat "$OUTPUT_FILE"
