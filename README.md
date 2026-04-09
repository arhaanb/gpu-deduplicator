# GPU Image Deduplicator

A CUDA-accelerated CLI tool that scans a directory of images and finds duplicate or near-duplicate images using GPU-computed perceptual similarity.

## How It Works

1. **Load & Fingerprint** - each image is loaded, converted to grayscale, and resized to a 32x32 thumbnail (its "perceptual fingerprint")
2. **GPU Similarity** - all fingerprints are uploaded to the GPU, where a CUDA kernel computes pairwise cosine similarity between every image pair in parallel
3. **Clustering** - a Union-Find algorithm groups images whose similarity exceeds a configurable threshold
4. **Report** - duplicate groups are printed with similarity scores

This approach catches common near-duplicates like JPEG recompression, slight crops, brightness changes, flips, and compression artifacts.

## Why GPU?

For N images, there are N*(N-1)/2 pairs to compare. Each comparison involves a dot product over 1024 dimensions (32x32 pixels). For 500 images that's around 125,000 comparisons, which is highly parallel and a natural fit for GPU execution. The GPU kernel computes all pairs simultaneously, giving a significant speedup over a CPU-based approach.

## Project Structure

```
├── src/
│   ├── main.cu              # CUDA source: kernels + CLI
│   ├── stb_image.h          # single-header image loading (stb)
│   └── stb_image_resize2.h  # single-header image resizing (stb)
├── sample_images/            # place your base images here
├── data/generated/           # auto-generated test dataset
├── output/                   # execution output/logs
├── generate_dataset.py       # script to create test data with near-duplicates
├── Makefile                  # build automation
├── run.sh                    # one-command build + run
└── README.md
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)
- Python 3 with Pillow and numpy (for dataset generation only)

## Quick Start

### 1. Add sample images

Place at least a few `.jpg` or `.png` images in `sample_images/`:

```bash
# Example: download some test images
cd sample_images
curl -O https://sipi.usc.edu/database/preview/misc/4.1.01.png
curl -O https://sipi.usc.edu/database/preview/misc/4.1.02.png
curl -O https://sipi.usc.edu/database/preview/misc/4.1.03.png
curl -O https://sipi.usc.edu/database/preview/misc/4.1.04.png
curl -O https://sipi.usc.edu/database/preview/misc/4.1.05.png
curl -O https://sipi.usc.edu/database/preview/misc/4.1.06.png
curl -O https://sipi.usc.edu/database/preview/misc/4.1.07.png
curl -O https://sipi.usc.edu/database/preview/misc/4.1.08.png
curl -O https://sipi.usc.edu/database/preview/misc/4.2.01.png
curl -O https://sipi.usc.edu/database/preview/misc/4.2.02.png
curl -O https://sipi.usc.edu/database/preview/misc/4.2.03.png
curl -O https://sipi.usc.edu/database/preview/misc/4.2.04.png
curl -O https://sipi.usc.edu/database/preview/misc/4.2.05.png
curl -O https://sipi.usc.edu/database/preview/misc/4.2.06.png
curl -O https://sipi.usc.edu/database/preview/misc/4.2.07.png
cd ..
```

### 2. Run everything

```bash
./run.sh
```

This will:
- Generate 200+ near-duplicate test images from your samples
- Build the CUDA binary
- Run the deduplicator and save results to `output/results.txt`

### 3. Or build and run manually

```bash
# Generate test dataset
python3 generate_dataset.py sample_images/ data/generated --variants 20

# Build
make

# Run
./bin/gpu_dedup -v -t 0.92 data/generated
```

## CLI Usage

```
Usage: gpu_dedup [options] <image_directory>

Options:
  -t, --threshold <float>   Similarity threshold (0.0-1.0, default: 0.95)
  -o, --output <file>       Write results to file (default: stdout)
  -v, --verbose             Print detailed progress
  -h, --help                Show this help message
```

### Examples

```bash
# Scan with default threshold (0.95 = very strict, only near-identical)
./bin/gpu_dedup ./my_photos

# Lower threshold to catch more aggressive edits
./bin/gpu_dedup -t 0.85 -v ./my_photos

# Save results to file
./bin/gpu_dedup -t 0.90 -o duplicates.txt ./data/generated
```

## Algorithm Details

### Perceptual Fingerprinting

Each image is reduced to a 32x32 grayscale thumbnail (1024 values normalized to [0,1]). This acts as a content-aware fingerprint that is robust to:
- Resolution changes
- Minor color shifts
- JPEG compression artifacts

### GPU Cosine Similarity

The CUDA kernel maps each unique image pair (i, j) to a GPU thread. Each thread computes:

```
similarity(A, B) = dot(A, B) / (|A| * |B|)
```

This produces a value in [-1, 1], where 1.0 means identical content. The kernel writes results to both `sim[i][j]` and `sim[j][i]` for a symmetric matrix.

### Union-Find Clustering

After the GPU computation, pairs exceeding the similarity threshold are merged into clusters using a Union-Find (disjoint set) data structure with path compression. Each cluster represents a group of duplicate images.

## Lessons Learned

- **Indexing math matters** - mapping a linear thread index to an (i, j) pair for the upper triangle of a matrix requires careful use of the quadratic formula to avoid out-of-bounds accesses
- **Memory planning** - the similarity matrix is N x N floats. For 1000 images that's around 4MB, which is manageable. For 10,000+ images you'd need a tiled approach
- **Threshold tuning** - 0.95 catches obvious duplicates (recompression, resize). 0.85-0.90 catches more aggressive edits (crops, brightness). Below 0.80 you start getting false positives
- **STB simplicity** - using stb single-header libraries avoids complex build dependencies (OpenCV, etc.) while still supporting all common image formats
