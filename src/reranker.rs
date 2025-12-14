use eyre::Result;
use fastembed::{RerankInitOptions, RerankResult, RerankerModel, TextRerank};

/// Reranker wrapper around FastEmbed
pub struct TextReranker {
    model: TextRerank,
}

impl TextReranker {
    /// Create a new TextReranker using BGE-Reranker-V2-M3 (auto-downloaded)
    /// This model is excellent for multilingual/Hinglish reranking
    pub fn new() -> Result<Self> {
        let model = TextRerank::try_new(
            RerankInitOptions::new(RerankerModel::BGERerankerV2M3)
                .with_show_download_progress(true),
        )
        .map_err(|e| eyre::eyre!("Failed to load reranker model: {}", e))?;

        Ok(TextReranker { model })
    }

    /// Rerank a list of documents against a query
    pub fn rerank(&mut self, query: String, documents: Vec<String>) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let results = self
            .model
            .rerank(query, documents, true, None)
            .map_err(|e| eyre::eyre!("Reranking failed: {}", e))?;

        Ok(results)
    }
}
