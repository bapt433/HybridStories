import os
import sys
import time
import math
import copy
import argparse
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from tokenizers import Tokenizer

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
from model import ModelConfig, HybridLM, count_parameters

DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

class BinaryDataLoader:
    def __init__(self, bin_path, seq_len, micro_batch_size, device,
                 sample_weights=None):
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = micro_batch_size
        self.device = device
        self.n_samples = len(self.data) // (seq_len + 1)
        self.sample_weights = sample_weights

        if sample_weights is not None:
            self.cum_weights = np.cumsum(sample_weights)
            self.cum_weights /= self.cum_weights[-1]
        else:
            self.cum_weights = None

    def get_batch(self):
        if self.cum_weights is not None:
            r = np.random.random(self.batch_size)
            idxs = np.searchsorted(self.cum_weights, r)
            idxs = np.clip(idxs, 0, self.n_samples - 1)
        else:
            idxs = np.random.randint(0, self.n_samples, size=self.batch_size)
        x = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)
        y = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)
        for i, idx in enumerate(idxs):
            start = idx * (self.seq_len + 1)
            end = start + self.seq_len + 1
            tokens = self.data[start:end].astype(np.int64)
            x[i] = tokens[:-1]
            y[i] = tokens[1:]
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

    def get_eval_batch(self, batch_size=None):
        bs = batch_size or self.batch_size
        idxs = np.random.randint(0, self.n_samples, size=bs)
        x = np.zeros((bs, self.seq_len), dtype=np.int64)
        y = np.zeros((bs, self.seq_len), dtype=np.int64)
        for i, idx in enumerate(idxs):
            start = idx * (self.seq_len + 1)
            end = start + self.seq_len + 1
            tokens = self.data[start:end].astype(np.int64)
            x[i] = tokens[:-1]
            y[i] = tokens[1:]
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

def get_lr(step, max_lr, min_lr, warmup_steps, max_steps, schedule="cosine"):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr

    if schedule == "wsd":
        stable_end = int(max_steps * 0.85)
        if step <= stable_end:
            return max_lr
        else:
            decay_frac = (step - stable_end) / (max_steps - stable_end)
            return max_lr * (0.1 + 0.9 * (1 - decay_frac))
    else:
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_ratio))

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    def update(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def copy_to(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)

def build_curriculum_weights(data, seq_len, phase):
    n_samples = len(data) // (seq_len + 1)
    difficulties = np.zeros(n_samples, dtype=np.float32)
    chunk = 10000
    for i in range(0, n_samples, chunk):
        end = min(i + chunk, n_samples)
        for j in range(i, end):
            start = j * (seq_len + 1)
            segment = data[start:start + seq_len + 1]
            difficulties[j] = len(set(segment.tolist()))

    if phase == 0:
        weights = np.exp(-difficulties / 10.0)
    elif phase == 1:
        weights = np.ones(n_samples, dtype=np.float32)
    else:
        weights = np.exp((difficulties - difficulties.mean()) / 10.0)

    weights /= weights.sum()
    return weights

def train(config_path=DEFAULT_CONFIG_PATH):
    cfg = load_config(config_path)

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    ecfg = cfg["eval"]
    gcfg = cfg["generation"]

    data_dir = PROJECT_DIR / dcfg["data_dir"]
    save_dir = PROJECT_DIR / ecfg["save_dir"]

    use_grad_noise = tcfg.get("grad_noise", True)
    grad_noise_std = tcfg.get("grad_noise_std", 0.01)
    use_swa = tcfg.get("swa", True)
    swa_start_frac = tcfg.get("swa_start_frac", 0.75)
    swa_n_avg = tcfg.get("swa_n_avg", 10)
    use_curriculum = tcfg.get("curriculum", True)
    n_rounds = tcfg.get("n_rounds", 1)
    hard_upsample = tcfg.get("hard_upsample", 3.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: CUDA not available")
    else:
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_total:.1f} GB VRAM)")

    tok_path = data_dir / "tokenizer.json"
    if not tok_path.exists():
        print("ERROR: Tokenizer not found. Run prepare_data.py first.")
        return
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab: {vocab_size}")

    model_cfg = ModelConfig.from_dict(mcfg)
    model_cfg.vocab_size = vocab_size
    model_cfg.max_seq_len = dcfg["seq_len"]
    model_cfg.use_checkpoint = mcfg.get("use_checkpoint", True)

    model = HybridLM(model_cfg).to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Block sharing: {model_cfg.block_sharing}")
    print(f"  Meta tokens: {model_cfg.n_meta_tokens}")
    print(f"  MLP ratio: {model_cfg.mlp_ratio}")

    resume_path = tcfg.get("resume_from")
    start_step = 0
    best_val_loss = float("inf")
    if resume_path:
        resume_full = PROJECT_DIR / resume_path if not Path(resume_path).is_absolute() else Path(resume_path)
        if resume_full.exists():
            ckpt = torch.load(str(resume_full), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            start_step = ckpt.get("step", 0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Resumed from {resume_full} (step {start_step}, val_loss {best_val_loss:.4f})")
        else:
            print(f"WARNING: resume checkpoint not found: {resume_full}")

    use_muon = tcfg.get("use_muon", True)
    muon_lr = tcfg.get("muon_lr", 0.02)
    adam_lr = tcfg.get("max_lr", 3e-4)
    weight_decay = tcfg.get("weight_decay", 0.01)
    ns_steps = tcfg.get("muon_ns_steps", 3)
    muon_momentum = tcfg.get("muon_momentum", 0.9)

    if use_muon:
        try:
            from muon import Muon
            muon_params = []
            adam_params = []
            for name, p in model.named_parameters():
                if p.ndim == 2 and "embed" not in name and "lm_head" not in name and "meta_tokens" not in name:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

            optimizer = Muon(
                muon_params=muon_params,
                adam_params=adam_params,
                lr=muon_lr,
                adam_lr=adam_lr,
                momentum=muon_momentum,
                betas=tuple(tcfg["betas"]),
                weight_decay=weight_decay,
                ns_steps=ns_steps,
            )
            print(f"Optimizer: Muon (lr={muon_lr}) for {len(muon_params)} 2D params")
            print(f"          + AdamW (lr={adam_lr}) for {len(adam_params)} 1D/embed params")
        except ImportError:
            print("Muon not available, using AdamW fallback")
            use_muon = False

    if not use_muon:
        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and "lm_head" not in n]
        nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 or "lm_head" in n]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=adam_lr, betas=tuple(tcfg["betas"]), eps=tcfg["eps"],
        )
        print("Optimizer: torch.optim.AdamW (Muon not available)")

    use_ema = tcfg.get("use_ema", True)
    ema_decay = tcfg.get("ema_decay", 0.999)
    ema = EMA(model, decay=ema_decay) if use_ema else None
    if ema:
        print(f"EMA: enabled (decay={ema_decay})")

    lr_schedule = tcfg.get("lr_schedule", "cosine")
    max_lr = tcfg.get("max_lr", 3e-4)
    min_lr = tcfg.get("min_lr", 1e-5)

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    if not train_path.exists():
        print("ERROR: Training data not found. Run prepare_data.py first.")
        return
    micro_bs = tcfg["micro_batch_size"]
    seq_len = dcfg["seq_len"]

    train_data = np.memmap(str(train_path), dtype=np.uint16, mode="r")

    sample_weights = None
    if use_curriculum:
        print("Building curriculum weights (phase 0: easy)...")
        sample_weights = build_curriculum_weights(train_data, seq_len, phase=0)
        print(f"  Curriculum weights ready ({len(sample_weights):,} samples)")

    train_loader = BinaryDataLoader(
        train_path, seq_len, micro_bs, device,
        sample_weights=sample_weights,
    )
    val_loader = BinaryDataLoader(val_path, seq_len, micro_bs, device) if val_path.exists() else None
    if val_loader:
        print(f"Train tokens: {len(train_loader.data):,} | Val tokens: {len(val_loader.data):,}")
    else:
        print(f"Train tokens: {len(train_loader.data):,}")

    amp_dtype = torch.bfloat16 if tcfg["dtype"] == "bfloat16" else torch.float16
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if (use_amp and amp_dtype == torch.float16) else None

    swa_model = None
    swa_n = 0
    if use_swa:
        swa_start_step = int(tcfg["max_steps"] * swa_start_frac)
        print(f"SWA: will start averaging at step {swa_start_step} ({swa_n_avg} checkpoints)")

    save_dir.mkdir(exist_ok=True)

    max_steps = tcfg["max_steps"]
    warmup_steps = tcfg["warmup_steps"]
    grad_accum = tcfg["grad_accum_steps"]
    grad_clip = tcfg["grad_clip"]

    print(f"\n{'='*60}")
    print(f"Training {total_params/1e6:.2f}M model for {max_steps} steps (starting at {start_step})")
    print(f"  Effective batch size: {micro_bs * grad_accum}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Gradient checkpointing: {model_cfg.use_checkpoint}")
    print(f"  Mixed precision: {tcfg['dtype']}")
    print(f"  Upgrades:")
    print(f"    Optimizer:        {'Muon' if use_muon else 'AdamW'}")
    print(f"    LR schedule:      {lr_schedule}")
    print(f"    EMA:              {use_ema}")
    print(f"    Gradient noise:   {use_grad_noise} (std={grad_noise_std})")
    print(f"    SWA:              {use_swa} (start={swa_start_frac:.0%}, n_avg={swa_n_avg})")
    print(f"    Curriculum:       {use_curriculum}")
    print(f"    Multi-round:      {n_rounds} round(s), hard_upsample={hard_upsample}x")
    print(f"    Block sharing:    {model_cfg.block_sharing}")
    print(f"    Meta tokens:      {model_cfg.n_meta_tokens}")
    print(f"    MLP ratio:        {model_cfg.mlp_ratio}")
    print(f"  Config: {config_path}")
    print(f"{'='*60}\n")

    for round_idx in range(n_rounds):
        if round_idx > 0:
            print(f"\n{'#'*60}")
            print(f"# ROUND {round_idx + 1}/{n_rounds}")
            print(f"# Upsampling hard examples by {hard_upsample}x")
            print(f"{'#'*60}\n")

            print("Computing per-sample losses for hard example upsampling...")
            n_samples = len(train_data) // (seq_len + 1)
            sample_losses = np.zeros(n_samples, dtype=np.float32)
            eval_bs = 32
            model.eval()
            with torch.no_grad():
                for i in range(0, n_samples, eval_bs):
                    end = min(i + eval_bs, n_samples)
                    bs = end - i
                    x = np.zeros((bs, seq_len), dtype=np.int64)
                    y = np.zeros((bs, seq_len), dtype=np.int64)
                    for j, idx in enumerate(range(i, end)):
                        s = idx * (seq_len + 1)
                        e = s + seq_len + 1
                        tokens = train_data[s:e].astype(np.int64)
                        x[j] = tokens[:-1]
                        y[j] = tokens[1:]
                    x_t = torch.from_numpy(x).to(device)
                    y_t = torch.from_numpy(y).to(device)
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            _, loss = model(x_t, y_t)
                    else:
                        _, loss = model(x_t, y_t)
                    sample_losses[i:end] = loss.item()
            model.train()

            threshold = np.percentile(sample_losses, 70)
            weights = np.ones(n_samples, dtype=np.float32)
            weights[sample_losses > threshold] = hard_upsample
            weights /= weights.sum()
            train_loader = BinaryDataLoader(
                train_path, seq_len, micro_bs, device,
                sample_weights=weights,
            )
            print(f"  Hard examples (>loss {threshold:.4f}): upsampled {hard_upsample}x")

            start_step = 0
            max_lr = max_lr * 0.3
            min_lr = min_lr * 0.3

        if use_curriculum:
            phase_boundaries = [
                (0, int(max_steps * 0.40), 0),
                (int(max_steps * 0.40), int(max_steps * 0.75), 1),
                (int(max_steps * 0.75), max_steps, 2),
            ]
        else:
            phase_boundaries = [(0, max_steps, -1)]

        step = start_step
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_steps = 0
        current_phase = 0 if use_curriculum else -1

        while step < max_steps:
            if use_curriculum:
                for p_start, p_end, phase in phase_boundaries:
                    if p_start <= step < p_end and phase != current_phase:
                        current_phase = phase
                        phase_names = ["easy", "medium", "hard"]
                        print(f"\n  [curriculum] switching to phase {phase} ({phase_names[phase]}) at step {step}\n")
                        new_weights = build_curriculum_weights(train_data, seq_len, phase)
                        train_loader = BinaryDataLoader(
                            train_path, seq_len, micro_bs, device,
                            sample_weights=new_weights,
                        )

            lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps, schedule=lr_schedule)
            for pg in optimizer.param_groups:
                if pg.get("use_muon"):
                    pg["lr"] = lr * (muon_lr / max_lr)
                else:
                    pg["adam_lr"] = lr
                    pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            total_loss = 0.0
            for micro_step in range(grad_accum):
                x, y = train_loader.get_batch()
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)
                loss = loss / grad_accum

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += loss.item()

            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if use_grad_noise and step < max_steps * 0.8:
                noise_scale = grad_noise_std * (1.0 - step / max_steps)
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad) * noise_scale)

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if ema is not None:
                ema.update(model)

            step += 1
            running_loss += total_loss
            running_steps += 1

            log_every = ecfg["log_every"]
            if step % log_every == 0:
                avg_loss = running_loss / running_steps
                elapsed = time.time() - t0
                steps_per_sec = (step - start_step) / elapsed if elapsed > 0 else 0
                tokens_per_sec = steps_per_sec * micro_bs * grad_accum * seq_len
                vram_alloc = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                vram_peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                print(f"step {step:>6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                      f"{steps_per_sec:.1f} steps/s | {tokens_per_sec:,.0f} tok/s | "
                      f"VRAM {vram_alloc:.2f}/{vram_peak:.2f} GB", flush=True)
                running_loss = 0.0
                running_steps = 0

            eval_every = ecfg["eval_every"]
            if step % eval_every == 0 and val_loader:
                model.eval()
                if ema is not None:
                    original_weights = {n: p.data.clone() for n, p in model.named_parameters()}
                    ema.copy_to(model)
                val_losses = []
                for _ in range(ecfg["eval_batches"]):
                    x, y = val_loader.get_eval_batch()
                    with torch.no_grad():
                        if use_amp:
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                _, loss = model(x, y)
                        else:
                            _, loss = model(x, y)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)
                if ema is not None:
                    for n, p in model.named_parameters():
                        p.data.copy_(original_weights[n])
                model.train()
                print(f"  -> val_loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}" + (" (EMA)" if ema else ""), flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = save_dir / "best.pt"
                    torch.save({
                        "model": model.state_dict(),
                        "config": model_cfg.__dict__,
                        "step": step,
                        "val_loss": val_loss,
                    }, save_path)
                    print(f"  -> saved best checkpoint to {save_path}", flush=True)

            if use_swa and step >= int(max_steps * swa_start_frac):
                if swa_model is None:
                    swa_model = copy.deepcopy(model)
                    swa_n = 1
                    print(f"  [SWA] started averaging at step {step}", flush=True)
                elif step % (max_steps // swa_n_avg) == 0 and step < max_steps:
                    with torch.no_grad():
                        for (n_swa, p_swa), (_, p_model) in zip(
                            swa_model.named_parameters(), model.named_parameters()
                        ):
                            p_swa.add_(p_model - p_swa, alpha=1.0 / (swa_n + 1))
                    swa_n += 1
                    print(f"  [SWA] averaged checkpoint {swa_n}/{swa_n_avg} at step {step}", flush=True)

            sample_every = gcfg["sample_every"]
            if step % sample_every == 0:
                model.eval()
                prompt_ids = tokenizer.encode(gcfg["sample_prompt"]).ids
                if len(prompt_ids) == 0:
                    prompt_ids = [0]
                idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            generated = model.generate(
                                idx,
                                max_new_tokens=gcfg["sample_max_tokens"],
                                temperature=gcfg["sample_temperature"],
                                top_p=gcfg["sample_top_p"],
                            )
                    else:
                        generated = model.generate(
                            idx,
                            max_new_tokens=gcfg["sample_max_tokens"],
                            temperature=gcfg["sample_temperature"],
                            top_p=gcfg["sample_top_p"],
                        )
                text = tokenizer.decode(generated[0].tolist())
                print(f"\n  [sample @ step {step}] {text}\n", flush=True)
                model.train()

            save_every = ecfg["save_every"]
            if step % save_every == 0:
                save_path = save_dir / f"step_{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "config": model_cfg.__dict__,
                    "step": step,
                }, save_path)
                print(f"  -> saved checkpoint to {save_path}", flush=True)

        save_path = save_dir / f"round_{round_idx+1}_final.pt"
        torch.save({
            "model": model.state_dict(),
            "config": model_cfg.__dict__,
            "step": step,
        }, save_path)
        print(f"\nRound {round_idx+1} complete. Checkpoint: {save_path}")
        print(f"Round time: {(time.time() - t0)/60:.1f} min")

    if use_swa and swa_model is not None and swa_n > 1:
        print(f"\n[SWA] Saving averaged model ({swa_n} checkpoints averaged)")
        save_path = save_dir / "swa.pt"
        torch.save({
            "model": swa_model.state_dict(),
            "config": model_cfg.__dict__,
            "step": step,
            "swa_n": swa_n,
        }, save_path)
        print(f"  Saved SWA checkpoint to {save_path}")

    save_path = save_dir / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "config": model_cfg.__dict__,
        "step": step,
    }, save_path)
    print(f"\nTraining complete. Final checkpoint: {save_path}")
    print(f"Total time: {(time.time() - t0)/60:.1f} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Hybrid LM")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="Path to YAML config file (default: config.yaml)")
    args = parser.parse_args()
    train(args.config)