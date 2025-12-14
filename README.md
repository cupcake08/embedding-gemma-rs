# EmbeddingGemma-RS

🚀 **Fast, Quantized, and Multilingual Text Embeddings for Python & Rust**

This library provides high-performance bindings for the **EmbeddingGemma** model (developed by Google), utilizing the `fastembed-rs` backend with ONNX Runtime. It is designed for efficient, local, and low-latency text embedding and reranking, specifically optimized for:

*   **Multilingual Support:** Excellent performance on Hindi, Hinglish, and English.
*   **Low Resource Usage:** Supports 4-bit and 8-bit quantization (e.g., Q4F16 model is only ~175MB).
*   **Ease of Use:** Models are automatically downloaded and cached on first use.

## Features

*   **Python Bindings:** Seamless integration with Python via `pyo3`.
*   **Auto-Quantization:** Load standard floating-point or compressed quantized models easily.
*   **Reranking:** Integrated BGE-M3 reranker support for improving search relevance.
*   **Fast:** Uses ONNX Runtime with optimization for Apple Silicon (Metal/CoreML) and x86 CPUs.

## Installation

### Prerequisites
*   **Rust Toolchain:** Install from [rustup.rs](https://rustup.rs/).
*   **Python 3.8+**

### Building from Source

This project uses `maturin` to build Python extensions.

```bash
# Install maturin
pip install maturin

# Build and install into current environment
maturin develop --release
```

## Usage (Python)

```python
import embedding_gemma_rs

# 1. Initialize the Embedder
# Recommended: Use the quantized model for speed/size (Q4F16)
embedder = embedding_gemma_rs.TextEmbedder.new_quantized()

# Or load the full FP32 model (~1.2GB) if you need maximum precision
# embedder = embedding_gemma_rs.TextEmbedder()

# 2. Generate Embeddings
texts = ["Namaste world", "Hello kaise ho"]
embeddings = embedder.embed(texts)

print(f"Generated {len(embeddings)} embeddings of dimension {embedder.dimension()}")

# 3. Reranking (for RAG/Search)
reranker = embedding_gemma_rs.Reranker()
query = "Food order"
docs = ["I want a burger", "Mere paas gaadi hai", "Menu dikhao"]

# Returns list of (document, score, index)
results = reranker.rerank(query, docs)
for doc, score, idx in results:
    print(f"Score: {score:.4f} - {doc}")
```

## Models

The library automatically manages model downloads. By default, it uses:

*   **Embedding:** `google/embedding-gemma-300m` (converted to ONNX)
*   **Reranking:** `BAAI/bge-reranker-v2-m3`

Models are cached in `~/.fastembed_cache` (or local `.fastembed_cache` if configured).

## License

MIT
