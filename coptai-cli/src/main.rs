use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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
                candle_core::Device,
                optimize, DeviceConfig, OptimizerConfig, QuantizationConfig, QuantizationTarget,
            };

            tracing::info!(model_dir = %model_dir.display(), "Loading model");

            let device = Device::Cpu;
            let model = coptai_core::model::CoptaiModel::from_dir(&model_dir, &device)?;

            tracing::info!(tensors = model.weights.len(), "Model loaded");

            let quant = if int4_awq {
                Some(QuantizationConfig {
                    target: QuantizationTarget::Int4Awq,
                    calibration_samples: 128,
                })
            } else if int8 {
                Some(QuantizationConfig {
                    target: QuantizationTarget::Int8,
                    calibration_samples: 512,
                })
            } else {
                None
            };

            let config = OptimizerConfig {
                quantization: quant,
                pruning: None,
                graph_fusion: true,
                target_device: DeviceConfig::Cpu,
            };

            let _optimized = optimize(model, config)?;

            // TODO: serialise optimised weights to output_dir
            std::fs::create_dir_all(&output_dir)?;
            tracing::info!(output_dir = %output_dir.display(), "Optimisation complete (stub — weights not yet written)");
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
