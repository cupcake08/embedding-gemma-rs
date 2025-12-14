//! EmbeddingGemma-RS: Fast text embeddings using FastEmbed
//!
//! This crate provides Python bindings for FastEmbed text embeddings,
//! using the EmbeddingGemma300M model (supports 100+ languages including Hindi/Hinglish).

mod embedder;
mod reranker;

pub use embedder::{QuantizationType, TextEmbedder};
pub use reranker::TextReranker;

#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python enum for quantization types
#[cfg(feature = "python")]
#[pyclass(name = "QuantizationType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyQuantizationType {
    /// Full FP32 model (~1.2GB)
    Full = 0,
    /// FP16 model (~617MB)
    FP16 = 1,
    /// Q4 quantized (~197MB)
    Q4 = 2,
    /// Q4 with FP16 embeddings (~175MB) - Recommended
    Q4F16 = 3,
    /// INT8 quantized (~309MB)
    Quantized = 4,
}

#[cfg(feature = "python")]
impl From<PyQuantizationType> for QuantizationType {
    fn from(py_quant: PyQuantizationType) -> Self {
        match py_quant {
            PyQuantizationType::Full => QuantizationType::Full,
            PyQuantizationType::FP16 => QuantizationType::FP16,
            PyQuantizationType::Q4 => QuantizationType::Q4,
            PyQuantizationType::Q4F16 => QuantizationType::Q4F16,
            PyQuantizationType::Quantized => QuantizationType::Quantized,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "TextEmbedder")]
pub struct PyTextEmbedder {
    inner: TextEmbedder,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTextEmbedder {
    /// Create a new TextEmbedder using EmbeddingGemma full model (auto-downloaded, ~1.2GB)
    ///
    /// The model will be automatically downloaded on first use.
    #[new]
    fn new() -> PyResult<Self> {
        let inner = TextEmbedder::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create embedder: {}",
                e
            ))
        })?;
        Ok(PyTextEmbedder { inner })
    }

    /// Create a new TextEmbedder using EmbeddingGemma Q4F16 quantized model (auto-downloaded, ~175MB)
    ///
    /// Recommended for low-end CPUs. Same quality, faster inference, smaller download.
    #[staticmethod]
    fn new_quantized() -> PyResult<Self> {
        let inner = TextEmbedder::new_quantized_auto().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create quantized embedder: {}",
                e
            ))
        })?;
        Ok(PyTextEmbedder { inner })
    }

    /// Generate embeddings for a list of texts (for indexing)
    /// Args:
    ///     texts: List of strings to embed
    ///
    /// Returns:
    ///     List of embedding vectors (each is a list of floats)
    fn embed(&mut self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        self.inner.embed(texts).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Embedding failed: {}", e))
        })
    }

    /// Generate embedding for a single text
    fn embed_one(&mut self, text: String) -> PyResult<Vec<f32>> {
        self.inner.embed_one(&text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Embedding failed: {}", e))
        })
    }

    /// Get the embedding dimension (768 for EmbeddingGemma300M)
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

#[cfg(feature = "python")]
#[pyclass]
struct Reranker {
    inner: TextReranker,
}

#[cfg(feature = "python")]
#[pymethods]
impl Reranker {
    #[new]
    fn new() -> PyResult<Self> {
        let inner = TextReranker::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Reranker { inner })
    }

    fn rerank(
        &mut self,
        query: String,
        documents: Vec<String>,
    ) -> PyResult<Vec<(String, f32, usize)>> {
        let results = self
            .inner
            .rerank(query, documents)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let py_results = results
            .into_iter()
            .map(|r| (r.document.unwrap_or_default(), r.score, r.index))
            .collect();

        Ok(py_results)
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn embedding_gemma_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging
    pyo3_log::init();

    m.add_class::<PyTextEmbedder>()?;
    m.add_class::<PyQuantizationType>()?;
    m.add_class::<Reranker>()?;
    Ok(())
}
