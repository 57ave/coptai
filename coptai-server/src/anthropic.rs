/// Anthropic-compatible response
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,  // always a list, never a raw string
}

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,           // e.g. "coptai-llama3-8b-int8"
    pub system: Option<String>,  // top-level system prompt
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
}

fn default_max_tokens() -> usize {
    512
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
}

/// Token usage counters returned on every response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Serialize)]
pub struct MessagesResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,   // "end_turn" | "max_tokens" | "stop_sequence"
    pub usage: Usage,              // input_tokens, output_tokens — returned on every response
}

// ---------------------------------------------------------------------------
// Models list
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}
