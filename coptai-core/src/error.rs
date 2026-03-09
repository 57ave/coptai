use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoptaiError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(Box<dyn std::error::Error + Send + Sync>),

    #[error("Model not loaded — call load_from_safetensors first")]
    NotLoaded,

    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}
