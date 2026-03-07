use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use uuid::Uuid;

use crate::anthropic::{
    ContentBlock, MessagesRequest, MessagesResponse,
    ModelCard, ModelListResponse, StopReason, Usage,
};
use crate::state::AppState;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/messages", post(messages))   // Anthropic-style: text-in, text-out
        .route("/health", get(health))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelListResponse {
        object: "list",
        data: vec![ModelCard {
            id: state.model_id.clone(),
            object: "model",
            created: unix_now(),
            owned_by: "coptai",
        }],
    })
}

/// Accepts a list of messages, returns a single assistant text reply.
async fn messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MessagesRequest>,
) -> impl IntoResponse {
    tracing::info!(model = %req.model, messages = req.messages.len(), "messages");

    // Flatten all user content blocks into one prompt string.
    let prompt_text: String = req
        .messages
        .iter()
        .flat_map(|m| m.content.iter())
        .map(|block| match block {
            ContentBlock::Text { text } => text.as_str(),
        })
        .collect::<Vec<&str>>()
        .join("\n");

    // TODO: replace with real model runner from coptai-core
    let reply = format!(
        "[coptai stub] {} message(s) received.",
        req.messages.len()
    );

    let (input_tokens, output_tokens) = count_tokens(&state, &prompt_text, &reply);

    Json(MessagesResponse {
        id: format!("msg-{}", Uuid::new_v4()),
        model: state.model_id.clone(),
        content: vec![ContentBlock::Text { text: reply }],
        stop_reason: StopReason::EndTurn,
        usage: Usage { input_tokens, output_tokens },
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

//opti-check-need
fn count_tokens_approx(text: &str) -> usize {
    let words = text.split_whitespace().count();
    (words * 4).div_ceil(3)
}

/// count approx nb of tokens for now
fn count_tokens(
    state: &AppState,   // add arc when implement tokenizer
    prompt: &str,
    reply: &str,
) -> (usize, usize) {


    let input_tokens  = count_tokens_approx(&prompt);
    let output_tokens = count_tokens_approx(&reply);

    (input_tokens, output_tokens)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
