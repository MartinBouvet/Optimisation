# Optimart benchmark note

Extractive protocol: a gold answer counts as preserved if it still appears in the pruned prompt.
This is **not** LLM Exact Match. Same token budget B for truncation and Optimart.

- Suite: `crafted+hotpot40`
- Cases: 64
- Scorer: `checkpoint:artifacts/train_minilm/best.pt`
- Budget ratio: 0.45

| Arm | Tokens in (mean) | Tokens out (mean) | Reduction | Answer coverage | Support recall | Latency p50 (ms) | Latency p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 855.5 | 855.5 | 0.0% | 98.4% | 100.0% | 1.57 | 3.58 |
| trunc_head | 855.5 | 468.5 | 46.0% | 65.6% | 57.7% | 1.77 | 3.57 |
| trunc_tail | 855.5 | 466.9 | 46.9% | 65.6% | 59.0% | 1.75 | 3.41 |
| optimart | 855.5 | 469.8 | 45.9% | 90.6% | 85.5% | 123.18 | 279.23 |

## Cost model · 100 000 requests · input tokens only

| Model | Full | Optimart | Saved | Trunc head saved |
|---|---:|---:|---:|---:|
| gpt-4o-mini | $12.83 | $7.05 | $5.78 | $5.80 |
| gpt-4o | $213.87 | $117.46 | $96.41 | $96.74 |

Input prices used: gpt-4o-mini $0.15 / 1M, gpt-4o $2.50 / 1M (OpenAI public list, Sept 2026).
Output tokens are ignored: pruning changes the prompt, not the completion length in this model.

## Limits

- Extractive coverage is not LLM Exact Match.
- MiniLM was fine-tuned on 800 HotpotQA train examples (2 epochs).
- Hotpot cases in this suite skip the first 200 validation items so they are not the training val slice.
- Latency is PyTorch MiniLM, not ONNX; long Hotpot contexts are well above the 25 ms design target.
