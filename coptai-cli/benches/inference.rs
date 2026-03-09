use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};


fn run_prefill(_prompt: &str) {
    // This is an exemple for now when we'll implement the real models
    std::hint::black_box(_prompt);
}

fn bench_ttft(c: &mut Criterion) {
    let prompts = [
        ("short", "Hello, world!"),
        (
            "medium",
            "Explain the transformer architecture in simple terms.",
        ),
        (
            "long",
            "Write a detailed essay on the history of machine learning, \
             covering the perceptron, backpropagation, convolutional neural \
             networks, recurrent neural networks, and the transformer \
             architecture introduced in 'Attention Is All You Need'.",
        ),
    ];

    let mut group = c.benchmark_group("ttft");

    for (label, prompt) in &prompts {
        group.bench_with_input(BenchmarkId::new("prefill", label), prompt, |b, p| {
            b.iter(|| run_prefill(p))
        });
    }

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let tokens_to_generate = [1usize, 32, 128, 512];

    let mut group = c.benchmark_group("throughput");

    for &n in &tokens_to_generate {
        group.bench_with_input(BenchmarkId::new("decode_tokens", n), &n, |b, &n| {
            b.iter(|| {
                // TODO: replace with coptai_core decode loop
                for _ in 0..n {
                    std::hint::black_box("token");
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ttft, bench_throughput);
criterion_main!(benches);
