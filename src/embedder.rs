//! Core text embedding logic using FastEmbed

use eyre::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Quantization types for EmbeddingGemma models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizationType {
    /// Full FP32 model (~1.2GB)
    Full,
    /// FP16 model (~617MB)
    FP16,
    /// Q4 quantized (~197MB)
    Q4,
    /// Q4 with FP16 embeddings (~175MB) - Recommended
    #[default]
    Q4F16,
    /// INT8 quantized (~309MB)
    Quantized,
}

impl QuantizationType {
    /// Get the ONNX model filename for this quantization type
    pub fn model_filename(&self) -> &'static str {
        match self {
            QuantizationType::Full => "model.onnx",
            QuantizationType::FP16 => "model_fp16.onnx",
            QuantizationType::Q4 => "model_q4.onnx",
            QuantizationType::Q4F16 => "model_q4f16.onnx",
            QuantizationType::Quantized => "model_quantized.onnx",
        }
    }

    /// Get the ONNX data filename for this quantization type
    pub fn data_filename(&self) -> &'static str {
        match self {
            QuantizationType::Full => "model.onnx_data",
            QuantizationType::FP16 => "model_fp16.onnx_data",
            QuantizationType::Q4 => "model_q4.onnx_data",
            QuantizationType::Q4F16 => "model_q4f16.onnx_data",
            QuantizationType::Quantized => "model_quantized.onnx_data",
        }
    }
}

/// Text embedder wrapper around FastEmbed
pub struct TextEmbedder {
    model: TextEmbedding,
}

impl TextEmbedder {
    /// Create a new TextEmbedder using EmbeddingGemma300M full model (auto-downloaded, ~1.2GB)
    /// Supports 100+ languages including Hindi/Hinglish
    pub fn new() -> Result<Self> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::EmbeddingGemma300M).with_show_download_progress(true),
        )
        .map_err(|e| eyre::eyre!("Failed to load model: {}", e))?;

        Ok(TextEmbedder { model })
    }

    /// Create a new TextEmbedder using EmbeddingGemma300M Q4F16 quantized model (auto-downloaded, ~175MB)
    /// Supports 100+ languages including Hindi/Hinglish - RECOMMENDED for low-end CPUs
    pub fn new_quantized_auto() -> Result<Self> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::EmbeddingGemma300MQ).with_show_download_progress(true),
        )
        .map_err(|e| eyre::eyre!("Failed to load quantized model: {}", e))?;

        Ok(TextEmbedder { model })
    }

    /// Create with a specific predefined model
    pub fn with_model(model_type: EmbeddingModel) -> Result<Self> {
        let model =
            TextEmbedding::try_new(InitOptions::new(model_type).with_show_download_progress(true))
                .map_err(|e| eyre::eyre!("Failed to load model: {}", e))?;

        Ok(TextEmbedder { model })
    }

    /// Get the embedding dimension (768 for EmbeddingGemma300M)
    pub fn dimension(&self) -> usize {
        768
    }

    /// Generate embeddings for multiple texts
    pub fn embed(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let embeddings = self
            .model
            .embed(texts, None)
            .map_err(|e| eyre::eyre!("Embedding failed: {}", e))?;

        Ok(embeddings)
    }

    /// Generate embedding for a single text
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let mut embeddings = self.embed(texts)?;
        embeddings
            .pop()
            .ok_or_else(|| eyre::eyre!("No embedding generated"))
    }
}
