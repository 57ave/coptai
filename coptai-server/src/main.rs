use coptai_server::{routes, AppState, config::{Config}};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};



#[tokio::main]
async fn main() {
    let config = Config::from_env().expect("invalid config");
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // let tokenizer_source = std::env::var("TOKENIZER_PATH")
    //     .map(|p| TokenizerSource::File(p.into()))
    //     .unwrap_or_else(|_| TokenizerSource::HuggingFace("meta-llama/Meta-Llama-3-8B".into()));

    let state = AppState::new("coptai-llama3-8b-int8")
        .expect("failed to initialise app state");
    let app = routes::router(state);

    let addr = std::env::var("COPTAI_BIND")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string());

    tracing::info!(%addr, "coptai-server listening");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind");

    axum::serve(listener, app)
        .await
        .expect("server error");
}
