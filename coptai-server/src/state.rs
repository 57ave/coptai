use std::sync::Arc;
// use tokenizers::Tokenizer;

/// How to source a tokenizer — decoupled from any specific model.
#[derive(Debug, Clone)]
// pub enum TokenizerSource {
//     /// Load from a local file path (e.g. `tokenizer.json`)
//     File(std::path::PathBuf),
//     /// Load from a HuggingFace repo slug (e.g. `"meta-llama/Meta-Llama-3-8B"`)
//     HuggingFace(String),
// }

/// Shared application state threaded through every axum handler.

pub struct AppState {
    /// Opaque identifier returned to clients (e.g. `"coptai-llama3-8b-int8"`).
    pub model_id: String
    // pub tokenizer: Arc<Tokenizer>,
}

impl AppState {
    /// Build and wrap in [`Arc`] in one step.
    pub fn new(
        model_id: impl Into<String>,
        // tokenizer_source: TokenizerSource,
    ) -> Result<Arc<Self>, String> {
        // let tokenizer = match tokenizer_source {
        //     TokenizerSource::File(path) => {
        //         Tokenizer::from_file(&path)
        //             .map_err(|e| format!("failed to load tokenizer from {path:?}: {e}"))?
        //     }
        //     TokenizerSource::HuggingFace(repo) => {
        //         Tokenizer::from_pretrained(&repo, None)
        //             .map_err(|e| format!("failed to load tokenizer from HF repo {repo:?}: {e}"))?
        //     }
        // };

        Ok(Arc::new(Self {
            model_id: model_id.into(),
            // tokenizer: Arc::new(tokenizer),
        }))
    }
}
