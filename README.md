# Coptai

> Rust-native LLM compression, AOT optimization, and high-performance inference.  
> One call to make any LLM smaller, faster, cheaper.

```rust
let model = CoptaiModel::from_dir("llama3-8b/", &Device::Cuda(0))?;
let optimized = optimize(model, OptimizerConfig {
    quantization: Some(QuantizationConfig { target: QuantizationTarget::Int8, .. }),
    graph_fusion: true,
    ..Default::default()
})?;
```

---

## What it does

Coptai is inspired by [Pruna AI](https://docs.pruna.ai) but built entirely in Rust with zero Python runtime dependency.

| Pass | What changes | When |
|---|---|---|
| INT8 / INT4 quantization | Weight values compressed | AOT — once offline |
| Structured pruning | Attention heads / layers removed | AOT — once offline |
| Graph fusion | Linear → LayerNorm → Act merged | AOT — once offline |
| FlashAttention | Memory-efficient attention | Runtime — every token |
| Paged KV cache | Past keys/values reused | Runtime — every request |
| Continuous batching | Concurrent requests packed | Runtime — every request |

---

## Workspace

```
coptai/
├── coptai-core/     # optimization passes + model loader
├── coptai-server/   # axum HTTP server (Anthropic Messages API)
└── coptai-cli/      # benchmark runner (criterion: TTFT, TPS, VRAM)
```

---

## Quickstart

```bash
# 1 — optimize a model offline (runs once)
cargo run -p coptai-cli -- optimize \
  --model ./models/llama3-8b \
  --quant int8 \
  --out   ./models/llama3-8b-int8

# 2 — serve it
MODEL_ID=coptai-llama3-8b-int8 \
TOKENIZER_PATH=./models/llama3-8b/tokenizer.json \
cargo run -p coptai-server

# 3 — call it
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coptai-llama3-8b-int8",
    "messages": [{ "role": "user", "content": [{ "type": "text", "text": "Hello" }] }],
    "max_tokens": 256
  }'
```

---

## Design principles

- **Safety by default** — only `.safetensors`, never `.pkl` or `.bin`
- **Zero-copy I/O** — weights loaded via `mmap`, never copied from disk unnecessarily
- **No GIL, no GC** — pure Rust async runtime, no Python in the hot path
- **Declarative optimization** — express *what* to optimize via `OptimizerConfig`, not *how*

---

## Status

> Phase 1 in progress — loader + server stub functional, optimization passes pending.

| Component | Status |
|---|---|
| `safetensors` mmap loader | ✅ |
| Anthropic-compatible HTTP server | ✅ |
| INT8 quantization | ✅ |
| FlashAttention integration | 📋 planned |
| Paged KV cache | 📋 planned |
