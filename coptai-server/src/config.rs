// coptai-server/src/config.rs

#[derive(Debug, serde::Deserialize)]
pub struct Config {
    /// Port the HTTP server listens on. Default: 8080
    #[serde(default = "default_port")]
    pub port: u16,

    /// Path to `tokenizer.json` on disk.
    /// If unset, falls back to HuggingFace download.
    pub tokenizer_path: Option<String>,

    /// Opaque model identifier echoed back to clients.
    /// e.g. `"coptai-llama3-8b-int8"`
    pub model_id: String,
}

fn default_port() -> u16 {
    8080
}

impl Config {
    pub fn from_env() -> Result<Self, envy::Error> {
        envy::from_env::<Config>()
    }
}