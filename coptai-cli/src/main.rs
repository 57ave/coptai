use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod progress;

#[derive(Parser)]
#[command(name = "coptai", about = "Coptai model optimisation toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Optimise a model and write it to an output directory.
    Optimize {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long)]
        output_dir: PathBuf,

        #[arg(long)]
        int8: bool,

        #[arg(long)]
        int4_awq: bool,
    },
    /// Run a quick benchmark of TTFT / throughput on a loaded model.
    Bench {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long, default_value = "The quick brown fox jumps over the lazy dog.")]
        prompt: String,

        #[arg(long, default_value_t = 10)]
        iters: u32,
    },
    /// Print information about a model directory.
    Info {
        #[arg(long)]
        model_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Optimize {
            model_dir,
            output_dir,
            int8,
            int4_awq,
        } => {
            use coptai_core::{
                candle_core::Device, DeviceConfig, OptimizerConfig, QuantizationConfig,
                QuantizationTarget,
            };

            let device = Device::Cpu;

            if int4_awq {
                return Err(anyhow::anyhow!("INT4/AWQ is Phase 2 — not yet implemented"));
            }

            // Discover shard paths without loading any data.
            let model = coptai_core::model::CoptaiModel::from_dir_lazy(&model_dir)?;
            let shard_count = model.source_paths.len();

            let (optimized, q, kept): (coptai_core::model::OptimizedModel, usize, usize) = if int8 {
                // ── INT8 path: stream each shard with a live progress bar ──
                let cli_progress = progress::CliProgress::new(shard_count as u64);

                let refs: Vec<&std::path::Path> =
                    model.source_paths.iter().map(|p| p.as_path()).collect();

                let shards = coptai_core::loader::load_and_quantize_int8_with_progress(
                    &refs,
                    &device,
                    &cli_progress,
                )?;

                cli_progress.finish();

                let q = shards.quantized.len();
                let kept = shards.other.len();

                let config = OptimizerConfig {
                    quantization: Some(QuantizationConfig {
                        target: QuantizationTarget::Int8,
                        calibration_samples: 512,
                    }),
                    pruning: None,
                    graph_fusion: true,
                    target_device: DeviceConfig::Cpu,
                };
                let optimized = coptai_core::model::OptimizedModel {
                    inner: coptai_core::model::CoptaiModel {
                        weights: shards.other,
                        source_paths: model.source_paths,
                    },
                    device,
                    config,
                    quantized_weights: shards.quantized,
                };
                (optimized, q, kept)
            } else {
                let model = coptai_core::model::CoptaiModel::from_dir(&model_dir, &device)?;
                let total = model.weights.len();
                let optimized = coptai_core::model::OptimizedModel {
                    inner: model,
                    device,
                    config: OptimizerConfig::default(),
                    quantized_weights: std::collections::HashMap::new(),
                };
                (optimized, 0, total)
            };

            std::fs::create_dir_all(&output_dir)?;

            println!();
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!(" Optimisation complete");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!(" Total tensors  : {}", optimized.num_tensors());
            println!(" INT8 quantized : {q}  (2-D weight matrices)");
            println!(" Kept as F32    : {kept}  (embeddings, norms, biases)");
            println!(" Output dir     : {}", output_dir.display());
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!(" ⚠  Weights not yet serialised — save pass is TODO");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!();
        }

        Commands::Bench {
            model_dir,
            prompt,
            iters,
        } => {
            use coptai_core::candle_core::Device;
            use std::time::Instant;

            tracing::info!(model_dir = %model_dir.display(), "Loading model for benchmark");

            let device = Device::Cpu;
            let model = coptai_core::model::CoptaiModel::from_dir(&model_dir, &device)?;

            tracing::info!(tensors = model.weights.len(), prompt = %prompt, iters, "Running TTFT benchmark");

            let mut total_ns: u128 = 0;
            for i in 0..iters {
                let t0 = Instant::now();
                // TODO: replace with actual prefill call once inference is wired
                let _ = std::hint::black_box(&prompt);
                let elapsed = t0.elapsed().as_nanos();
                total_ns += elapsed;
                tracing::debug!(iter = i, elapsed_us = elapsed / 1000, "prefill iteration");
            }

            let avg_ms = total_ns as f64 / iters as f64 / 1_000_000.0;
            println!("TTFT baseline (stub): avg {avg_ms:.3} ms over {iters} iterations");
        }

        Commands::Info { model_dir } => {
            use coptai_core::candle_core::Device;

            let device = Device::Cpu;
            let model = coptai_core::model::CoptaiModel::from_dir(&model_dir, &device)?;

            println!("Model directory : {}", model_dir.display());
            println!("Tensors loaded  : {}", model.weights.len());
            for (name, tensor) in model.weights.iter().take(10) {
                println!("  {name:60} shape={:?}", tensor.shape());
            }
            if model.weights.len() > 10 {
                println!("  … and {} more tensors", model.weights.len() - 10);
            }
        }
    }

    Ok(())
}
