// GPU Image Deduplicator
// Computes perceptual fingerprints on GPU and finds duplicate/near-duplicate images.
//
// Algorithm:
//   1. Load all images, resize to a fixed thumbnail (32x32 grayscale)
//   2. Upload all thumbnails to GPU memory
//   3. Compute pairwise cosine similarity between all image pairs on GPU
//   4. Threshold similarity scores to find duplicate clusters
//   5. Report duplicate groups

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

// Thumbnail dimensions for fingerprinting
static const int kThumbSize = 32;
static const int kThumbPixels = kThumbSize * kThumbSize;

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

// Kernel: compute pairwise cosine similarity between all image fingerprints.
// Each thread computes similarity for one pair (i, j) where j > i.
// fingerprints: N x kThumbPixels array of normalized grayscale pixels
// sim_matrix: N x N output similarity matrix
__global__ void CosineSimilarityKernel(const float* fingerprints,
                                       float* sim_matrix,
                                       int n, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = (n * (n - 1)) / 2;
  if (idx >= total_pairs) return;

  // Map linear index to (i, j) pair where j > i
  // Using quadratic formula: i = floor((2*n-1 - sqrt((2*n-1)^2 - 8*idx)) / 2)
  int i = static_cast<int>(
      floor((2.0 * n - 1.0 - sqrt((2.0 * n - 1.0) * (2.0 * n - 1.0) - 8.0 * idx)) / 2.0));
  int j = idx - i * (2 * n - i - 1) / 2 + i + 1;

  // Compute cosine similarity: dot(a,b) / (|a| * |b|)
  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;

  const float* a = fingerprints + i * dim;
  const float* b = fingerprints + j * dim;

  for (int k = 0; k < dim; ++k) {
    dot += a[k] * b[k];
    norm_a += a[k] * a[k];
    norm_b += b[k] * b[k];
  }

  float denom = sqrtf(norm_a) * sqrtf(norm_b);
  float sim = (denom > 1e-8f) ? (dot / denom) : 0.0f;

  sim_matrix[i * n + j] = sim;
  sim_matrix[j * n + i] = sim;
}

// Kernel: set diagonal of similarity matrix to 1.0
__global__ void SetDiagonalKernel(float* sim_matrix, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    sim_matrix[idx * n + idx] = 1.0f;
  }
}

// ---------------------------------------------------------------------------
// Image loading utilities
// ---------------------------------------------------------------------------

// Check if a filename has a supported image extension
bool IsSupportedImage(const std::string& filename) {
  std::string lower = filename;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  return lower.size() > 4 &&
         (lower.substr(lower.size() - 4) == ".jpg" ||
          lower.substr(lower.size() - 4) == ".png" ||
          lower.substr(lower.size() - 4) == ".bmp" ||
          lower.substr(lower.size() - 5) == ".jpeg" ||
          lower.substr(lower.size() - 4) == ".tga");
}

// Recursively collect image file paths from a directory
void CollectImagePaths(const std::string& dir_path,
                       std::vector<std::string>& paths) {
  DIR* dir = opendir(dir_path.c_str());
  if (!dir) {
    fprintf(stderr, "Warning: cannot open directory %s\n", dir_path.c_str());
    return;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") continue;

    std::string full_path = dir_path + "/" + name;
    struct stat st;
    if (stat(full_path.c_str(), &st) != 0) continue;

    if (S_ISDIR(st.st_mode)) {
      CollectImagePaths(full_path, paths);
    } else if (IsSupportedImage(name)) {
      paths.push_back(full_path);
    }
  }
  closedir(dir);
}

// Load an image, resize to kThumbSize x kThumbSize grayscale, normalize to [0,1]
bool LoadAndFingerprint(const std::string& path, float* out_fingerprint) {
  int w, h, channels;
  unsigned char* img = stbi_load(path.c_str(), &w, &h, &channels, 1);  // force grayscale
  if (!img) {
    fprintf(stderr, "Warning: failed to load %s\n", path.c_str());
    return false;
  }

  // Resize to thumbnail
  unsigned char thumb[kThumbPixels];
  stbir_resize_uint8_linear(img, w, h, 0, thumb, kThumbSize, kThumbSize, 0,
                            STBIR_1CHANNEL);

  // Normalize to [0, 1]
  for (int i = 0; i < kThumbPixels; ++i) {
    out_fingerprint[i] = static_cast<float>(thumb[i]) / 255.0f;
  }

  stbi_image_free(img);
  return true;
}

// ---------------------------------------------------------------------------
// Duplicate clustering (Union-Find on CPU after GPU similarity computation)
// ---------------------------------------------------------------------------

class UnionFind {
 public:
  explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
    for (int i = 0; i < n; ++i) parent_[i] = i;
  }

  int Find(int x) {
    while (parent_[x] != x) {
      parent_[x] = parent_[parent_[x]];
      x = parent_[x];
    }
    return x;
  }

  void Unite(int x, int y) {
    int rx = Find(x), ry = Find(y);
    if (rx == ry) return;
    if (rank_[rx] < rank_[ry]) std::swap(rx, ry);
    parent_[ry] = rx;
    if (rank_[rx] == rank_[ry]) rank_[rx]++;
  }

 private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

// ---------------------------------------------------------------------------
// Usage / CLI
// ---------------------------------------------------------------------------

void PrintUsage(const char* prog) {
  printf("GPU Image Deduplicator\n");
  printf("Usage: %s [options] <image_directory>\n\n", prog);
  printf("Options:\n");
  printf("  -t, --threshold <float>   Similarity threshold (0.0-1.0, default: 0.95)\n");
  printf("  -o, --output <file>       Write results to file (default: stdout)\n");
  printf("  -v, --verbose             Print detailed progress\n");
  printf("  -h, --help                Show this help message\n");
  printf("\nExample:\n");
  printf("  %s -t 0.90 -v ./data/generated\n", prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  // Parse arguments
  float threshold = 0.95f;
  const char* output_file = nullptr;
  bool verbose = false;
  const char* input_dir = nullptr;

  for (int i = 1; i < argc; ++i) {
    if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threshold") == 0) && i + 1 < argc) {
      threshold = atof(argv[++i]);
    } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
      output_file = argv[++i];
    } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
      return 0;
    } else {
      input_dir = argv[i];
    }
  }

  if (!input_dir) {
    PrintUsage(argv[0]);
    return 1;
  }

  // ---- Step 1: Discover and load images ----
  printf("Scanning directory: %s\n", input_dir);
  std::vector<std::string> image_paths;
  CollectImagePaths(input_dir, image_paths);
  std::sort(image_paths.begin(), image_paths.end());

  int n = static_cast<int>(image_paths.size());
  if (n < 2) {
    fprintf(stderr, "Error: need at least 2 images, found %d\n", n);
    return 1;
  }
  printf("Found %d images\n", n);

  // Load and compute fingerprints on CPU
  if (verbose) printf("Loading and fingerprinting images...\n");
  std::vector<float> fingerprints(n * kThumbPixels);
  int loaded = 0;
  for (int i = 0; i < n; ++i) {
    if (LoadAndFingerprint(image_paths[i], fingerprints.data() + i * kThumbPixels)) {
      loaded++;
    }
    if (verbose && (i + 1) % 50 == 0) {
      printf("  Loaded %d / %d images\n", i + 1, n);
    }
  }
  printf("Successfully loaded %d / %d images\n", loaded, n);

  // ---- Step 2: Upload fingerprints to GPU ----
  if (verbose) printf("Uploading fingerprints to GPU...\n");

  float* d_fingerprints = nullptr;
  float* d_sim_matrix = nullptr;
  size_t fp_bytes = n * kThumbPixels * sizeof(float);
  size_t sim_bytes = n * n * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_fingerprints, fp_bytes));
  CUDA_CHECK(cudaMalloc(&d_sim_matrix, sim_bytes));
  CUDA_CHECK(cudaMemcpy(d_fingerprints, fingerprints.data(), fp_bytes,
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_sim_matrix, 0, sim_bytes));

  // ---- Step 3: Compute pairwise similarity on GPU ----
  if (verbose) printf("Computing pairwise similarity on GPU...\n");

  int total_pairs = (n * (n - 1)) / 2;
  int block_size = 256;
  int grid_size = (total_pairs + block_size - 1) / block_size;

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  CosineSimilarityKernel<<<grid_size, block_size>>>(
      d_fingerprints, d_sim_matrix, n, kThumbPixels);
  CUDA_CHECK(cudaGetLastError());

  // Set diagonal
  int diag_grid = (n + block_size - 1) / block_size;
  SetDiagonalKernel<<<diag_grid, block_size>>>(d_sim_matrix, n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float gpu_time_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
  printf("GPU similarity computation: %.2f ms (%d pairs)\n", gpu_time_ms, total_pairs);

  // ---- Step 4: Download results and find duplicate clusters ----
  if (verbose) printf("Downloading similarity matrix...\n");
  std::vector<float> sim_matrix(n * n);
  CUDA_CHECK(cudaMemcpy(sim_matrix.data(), d_sim_matrix, sim_bytes,
                         cudaMemcpyDeviceToHost));

  // Free GPU memory
  CUDA_CHECK(cudaFree(d_fingerprints));
  CUDA_CHECK(cudaFree(d_sim_matrix));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Union-Find clustering
  if (verbose) printf("Clustering duplicates (threshold=%.3f)...\n", threshold);
  UnionFind uf(n);
  int dup_pairs = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (sim_matrix[i * n + j] >= threshold) {
        uf.Unite(i, j);
        dup_pairs++;
      }
    }
  }

  // Group by cluster
  std::vector<std::vector<int>> clusters;
  std::vector<int> cluster_map(n, -1);
  for (int i = 0; i < n; ++i) {
    int root = uf.Find(i);
    if (cluster_map[root] == -1) {
      cluster_map[root] = static_cast<int>(clusters.size());
      clusters.push_back({});
    }
    clusters[cluster_map[root]].push_back(i);
  }

  // Filter to clusters with >1 member (actual duplicates)
  std::vector<std::vector<int>> dup_clusters;
  for (auto& c : clusters) {
    if (c.size() > 1) dup_clusters.push_back(c);
  }

  // ---- Step 5: Output results ----
  FILE* out = stdout;
  if (output_file) {
    out = fopen(output_file, "w");
    if (!out) {
      fprintf(stderr, "Error: cannot open output file %s\n", output_file);
      out = stdout;
    }
  }

  fprintf(out, "\n========================================\n");
  fprintf(out, " GPU Image Deduplicator - Results\n");
  fprintf(out, "========================================\n");
  fprintf(out, "Images scanned:    %d\n", n);
  fprintf(out, "Threshold:         %.3f\n", threshold);
  fprintf(out, "Duplicate pairs:   %d\n", dup_pairs);
  fprintf(out, "Duplicate groups:  %d\n", static_cast<int>(dup_clusters.size()));
  fprintf(out, "GPU compute time:  %.2f ms\n", gpu_time_ms);
  fprintf(out, "========================================\n\n");

  if (dup_clusters.empty()) {
    fprintf(out, "No duplicates found.\n");
  } else {
    for (size_t g = 0; g < dup_clusters.size(); ++g) {
      fprintf(out, "Group %zu (%zu images):\n", g + 1, dup_clusters[g].size());
      for (size_t k = 0; k < dup_clusters[g].size(); ++k) {
        int idx = dup_clusters[g][k];
        fprintf(out, "  [%d] %s\n", idx, image_paths[idx].c_str());
        // Print similarity to first image in group
        if (k > 0) {
          int first = dup_clusters[g][0];
          fprintf(out, "       similarity to [%d]: %.4f\n", first,
                  sim_matrix[first * n + idx]);
        }
      }
      fprintf(out, "\n");
    }
  }

  // Summary
  int unique_count = 0;
  for (auto& c : clusters) {
    if (c.size() == 1) unique_count++;
  }
  fprintf(out, "Summary: %d unique images, %d images are duplicates across %d groups\n",
          unique_count, n - unique_count, static_cast<int>(dup_clusters.size()));

  if (output_file && out != stdout) {
    fclose(out);
    printf("Results written to %s\n", output_file);
  }

  return 0;
}
