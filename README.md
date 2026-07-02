# Hybrid TinyStories — Sub-1M Hybrid Language Model

A hybrid Mamba-2 SSM + Transformer language model that produces coherent English
stories at **248K parameters**, less than a third of the size of a pure
transformer that failed to produce coherent text at 800K.

## Results

| Model | Params | Val Loss | Val PPL | Coherent English? |
|-------|--------|----------|---------|-------------------|
| Pure Transformer (baseline) | 800K | 3.40 | ~30 | No — stuck at a permanent wall |
| Hybrid + Muon (other repo) | 652K | 2.17 | 8.85 | Yes — multi-paragraph stories |
| Hybrid + Muon (this repo) | 248K | 2.33 | 10.3 | Yes — coherent sentences & dialogue |

The 248K model writes stories with named characters, dialogue, emotional
arcs, and scene transitions. The 800K pure transformer (trained to
completion on the same dataset) never broke through loss 3.4 and produced
barely coherent output.

### Sample generation (248K model, step 20,000)

> Once upon a time there was a little girl named Sarah. She was very
> creative and loved to make her best friends. One day Sarah and her family
> were all sad. Sarah didn't want to get hurt. She wanted to go outside and
> have a big swing, so she was playing with her ball. She saw a little girl
> and asked her mum, "Mom, can I play with you?"
>
> Her mum smiled and said, "Yes, of course. I will be here and play with my
> ball." Sarah was happy. She saw the puppy looking at the squirrel and
> decided to play with it.


## Why This Works

Three factors combine to break through the wall that pure transformers hit:

### 1. Hybrid Architecture (Falcon-H1 inspired)

Each block runs attention and SSM in parallel on the same input, then applies
an MLP : the Falcon-H1 "SA_M" pattern:

```
r' = r + F_attn(Norm(r)) + F_ssm(Norm(r))    # parallel attention + SSM
r  = r' + F_mlp(Norm(r'))                     # sequential SwiGLU MLP
```

At small scale, a pure transformer's attention heads are too narrow to
compress sequence information efficiently. The Mamba-2 SSM provides
linear-time sequence memory that carries narrative context forward across
tokens — something attention at d_head=16-32 simply cannot do well.

### 2. Muon Optimizer

[Muon](https://github.com/KellerJordan/Muon) (MomentUm Orthogonalized by
Newton-Schulz) orthogonalizes the momentum buffer via a Newton-Schulz
iteration. It uses 1 momentum buffer per parameter (vs AdamW's 2), and
achieves 1.3-1.4x compute efficiency vs well-tuned AdamW at small scale,
with the gain being largest at small scale.

Applied to 2D weight matrices (Mamba projections, attention, MLP). AdamW
handles embeddings, biases, A_log, D, and meta tokens. Pure PyTorch, no
bitsandbytes required.

### 3. Parameter-Efficiency Techniques

- **Block weight sharing** (MobileLLM): 17 effective layers from 9 unique
  blocks: doubles depth without doubling parameters
- **Meta tokens** (Hymba): 2-4 learnable tokens prepended to every input as
  a scratchpad for attention heads
- **MLP ratio 2.77** (PanGu-π Pro): optimal FFN expansion, not 2.0 or 3.0
- **Tied embeddings**: token embedding = LM head
- **Depth over width**: d_model=32 with 17 layers beats wider/shallower
  configs at the same parameter budget

## Architecture

### 248K model (current config)

| Component          | Value          | Note                                |
|--------------------|----------------|-------------------------------------|
| d_model            | 32             | narrow but deep                      |
| n_layers           | 17 (9 unique) | block sharing: 17 effective, 9 unique|
| vocab_size         | 2048           | byte-level BPE on TinyStories       |
| Attention          | 1 head         | d_head=32 (full d_model)             |
| SSM                | Mamba-2        | d_inner=64, d_state=4, d_conv=4     |
| MLP                | SwiGLU         | hidden=88 (ratio 2.77)             |
| Positional         | RoPE           | theta=500K                           |
| Norm               | RMSNorm        | pre-norm                             |
| Embeddings         | Tied          | tok_emb = lm_head                    |
| Meta tokens        | 2              | Hymba-style learnable prepended tokens|

### Parameter breakdown (248K model)

| Component            | Parameters |
|----------------------|------------|
| Embedding (tied)     | 65,536     |
| 9 unique blocks ×9   | 181,056    |
| Meta tokens          | 64         |
| Final norm           | 32         |
| Conv1d biases        | 960        |
| **Total**            | **247,648**|

### Configurable

All model and training settings are in `config.yaml`. The same codebase
scales from 248K to 14M+ parameters by changing a few numbers — no code
edits needed.

## Training Techniques

| # | Technique              | Source              | What it does                       |
|---|------------------------|---------------------|------------------------------------|
| 1 | Muon optimizer          | Keller Jordan 2024  | Newton-Schulz orthogonalized momentum for 2D weights, AdamW for 1D. Pure PyTorch. |
| 2 | WSD schedule            | MiniCPM 2024        | Warmup → Stable (85%) → Decay (15%). No need to commit to total steps upfront. |
| 3 | EMA weights             | Standard            | Exponential moving average (decay=0.999). Evaluate with EMA weights for better generalization. |
| 4 | Block weight sharing    | MobileLLM 2024      | Share weights between non-adjacent layers. 17 effective layers from 9 unique blocks. |
| 5 | Meta tokens             | Hymba/NVIDIA 2025   | 2 learnable tokens prepended to every input. Scratchpad for attention. |
| 6 | MLP ratio 2.77          | PanGu-π Pro 2024    | Optimal FFN expansion (not 2.0, 3.0, or 4.0). |
| 7 | SWA                     | Standard            | Stochastic Weight Averaging — average 10 checkpoints during last 25% of training. |
| 8 | Curriculum learning     | ACL 2024            | Easy → medium → hard phases (40%/35%/25%). Difficulty estimated by token diversity. |
| 9 | Multi-round training    | PanGu-π Pro 2024    | Round 2: upsample hard examples (top 30% loss) by 3x, retrain at lower LR. Not used in the final run, very long. |
| 10| Gradient clipping 0.5   | Standard            | Tighter clip (0.5 vs 1.0) for stability with Muon at high LR. |

## Key Findings

### The Transformer Wall

A pure transformer at 800K parameters, trained to completion on TinyStories
with AdamW, hit a permanent wall at loss 3.4 (perplexity ~30). No amount of
additional training, LR tuning, or architecture changes could break through.
The output was barely coherent, at most one correct sentence per several
attempts.

The hybrid architecture at 248K parameters (less than a third of that
transformer) broke through that wall in 1000 steps and reached loss 2.33
(perplexity 10.3), producing coherent multi-sentence stories with dialogue.

### Muon vs AdamW

The 248K hybrid model with AdamW reached loss ~5.0 at step 3000. The same
model with Muon reached loss 2.66 at step 3000, nearly 2x faster
convergence. The Newton-Schulz orthogonalization means every update step
moves in the steepest useful direction, with no wasted motion in the loss
landscape.

### Block Sharing Works at Small Scale

Block weight sharing (MobileLLM) was originally demonstrated at 125M-350M
parameters. This repo shows it works at 248K: 17 effective layers from 9
unique blocks, with no quality degradation compared to 13 separate blocks.
The depth gain is more valuable than the parameter savings at this scale.

### Single Attention Head is Sufficient

At d_model=32, splitting into 2 attention heads gives d_head=16 — too narrow
for attention to work well. Using 1 head with d_head=32 (full d_model)
produces better results. The SSM handles sequence memory; attention only
needs to handle precise local retrieval, which one wide head can do.

### EMA Beats Raw Weights

Validation loss with EMA weights is consistently lower than training loss
(val 2.33 vs train 2.51 at step 16,500). The EMA (decay=0.999) smooths
weight oscillations during the stable phase and produces better
generalization, free improvement with ~250KB of extra memory.

## Training

### Hardware

Tested on:
- **NVIDIA GTX 750** (Maxwell, 4.3 GB VRAM, 512 cores, no tensor cores)
- **Kaggle T4** (Turing, 16 GB VRAM)

The Muon optimizer is pure PyTorch and works on
any CUDA GPU including older architectures (Maxwell, Pascal, Turing).

### VRAM Usage

| Model | Params | VRAM (GTX 750) | VRAM (Kaggle T4) |
|-------|--------|----------------|-------------------|
| 248K  | 248K   | 0.15 GB        | 2.73 GB (batch=128) |
| 652K  | 652K   | 0.16 GB        | 3.91 GB (batch=128) |

### Speed

| Model | Params | GTX 750 | Kaggle T4 | Speedup |
|-------|--------|---------|-----------|---------|
| 248K  | 248K   | 1,760 tok/s | 8,163 tok/s | 4.6x |
| 652K  | 652K   | 494 tok/s   | 6,868 tok/s  | 13.9x |

## Files

```
hybrid-14m/
├── config.yaml        # All settings — model, training, eval, generation
├── model.py           # Architecture: HybridLM, GQA, MambaSSM, SwiGLU, RMSNorm
├── muon.py            # Muon optimizer (Newton-Schulz + AdamW hybrid)
├── train.py           # Training loop with all 10 upgrades
├── prepare_data.py    # Download TinyStories, train BPE tokenizer, encode to binary
├── generate.py        # Inference: load checkpoint, generate text
├── smoke_test.py      # Verify forward/backward/generation/overfit
├── data/
│   ├── tokenizer.json  # BPE tokenizer (vocab=2048)
│   ├── train.bin       # 530M tokens (1.06 GB)
│   └── val.bin         # 5.3M tokens (10.7 MB)
└── checkpoints/       # Saved during training
```

## Quick Start

```bash
# Install dependencies
pip install torch numpy tokenizers pyyaml requests

# Prepare data (download TinyStories ~1.9GB, train BPE, encode to binary)
python prepare_data.py

# Run smoke test (verify model works)
python smoke_test.py

# Train (edit config.yaml first — set model size, steps, etc.)
python train.py

# Generate text from trained model
python generate.py --prompt "Once upon a time" --tokens 200
```


## Configuration

Everything is in `config.yaml`. Key sections:

```yaml
model:          # Architecture (d_model, n_layers, heads, SSM, MLP, etc.)
data:           # Data paths, sequence length
training:       # Optimizer (Muon, AdamW), LR, batch, upgrades
eval:           # Checkpoint saving, eval frequency, log frequency
generation:     # Sample generation during training
inference:      # Default inference settings
```

### Scaling to different sizes

| Target | d_model | n_layers | n_heads_q | d_state | n_meta | Params |
|--------|---------|----------|-----------|---------|--------|--------|
| 248K   | 32      | 17 (9u)  | 1         | 4       | 2      | 247,648 |
| 436K   | 48      | 15 (8u)  | 2         | 4       | 4      | 436,464 |
| 652K   | 64      | 13 (7u)  | 2         | 4       | 4      | 652,416 |
| 976K   | 64      | 13 (13u)| 2         | 8       | 0      | 976,448 |

(u = unique blocks with block sharing)

## Research Sources

- **Falcon-H1** (arXiv:2507.22448) — parallel hybrid SSM+attention architecture
- **Muon** (Keller Jordan, 2024) — Newton-Schulz orthogonalized momentum optimizer
- **MobileLLM** (ICML 2024) — block-wise weight sharing for sub-billion models
- **Hymba** (NVIDIA, ICLR 2025) — learnable meta tokens for hybrid heads
- **PanGu-π Pro** (ICML 2024) — optimal MLP ratio (2.77), parameter inheritance
- **TinyStories** (Eldan & Li, 2023) — synthetic dataset for tiny LM training
- **WSD schedule** (MiniCPM, 2024) — warmup-stable-decay learning rate

## License

Apache License 2.0