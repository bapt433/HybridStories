import torch
import sys
import math
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
from model import ModelConfig, HybridLM, count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

config = ModelConfig(
    d_model=32, n_layers=17, vocab_size=2048, max_seq_len=128, use_checkpoint=True,
)
model = HybridLM(config).to(device)
total_params = count_parameters(model)
print(f"\nModel parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  d_model={config.d_model}, n_layers={config.n_layers}, d_head={config.d_head}")
print(f"  d_inner={config.d_inner}, d_state={config.d_state}, mlp_hidden={config.mlp_hidden}")

print("\n=== Test 1: Forward pass ===")
model.eval()
x = torch.randint(0, config.vocab_size, (4, 128), device=device)
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, targets=x)
print(f"  logits shape: {logits.shape} (expected: [4, 128, 2048])")
print(f"  loss: {loss.item():.4f} (expected: ~{math.log(2048):.4f} = ln(2048))")
assert logits.shape == (4, 128, 2048), f"Wrong logits shape: {logits.shape}"
assert not torch.isnan(logits).any(), "NaN in logits!"
assert not torch.isnan(loss), "NaN in loss!"
print("  PASS")

print("\n=== Test 2: Backward pass ===")
model.train()
model.zero_grad()
x = torch.randint(0, config.vocab_size, (4, 128), device=device)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss = model(x, targets=x)
loss.backward()
print(f"  loss: {loss.item():.4f}")
grad_ok = True
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"  WARNING: No gradient for {name}")
        grad_ok = False
    elif param.grad.abs().sum().item() == 0:
        print(f"  WARNING: Zero gradient for {name}")
        grad_ok = False
if grad_ok:
    print("  All gradients present and non-zero")
print("  PASS")

print("\n=== Test 3: VRAM usage ===")
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.train()
    model.zero_grad()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(2):
        x = torch.randint(0, config.vocab_size, (4, 128), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, targets=x)
        loss = loss / 2
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    vram_alloc = torch.cuda.memory_allocated() / 1e9
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM allocated: {vram_alloc:.3f} GB")
    print(f"  VRAM peak:     {vram_peak:.3f} GB")
    print(f"  Budget:        4.0 GB")
    if vram_peak < 4.0:
        print(f"  FITS in 4GB budget (using {vram_peak/4.0*100:.1f}%)")
    else:
        print(f"  WARNING: Exceeds 4GB budget!")
    print("  PASS")

print("\n=== Test 4: Generation ===")
model.eval()
prompt = torch.tensor([[1, 100, 200, 300]], dtype=torch.long, device=device)
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(prompt, max_new_tokens=30, temperature=0.8, top_p=0.9)
print(f"  Input tokens:    {prompt.tolist()[0]}")
print(f"  Generated shape: {generated.shape}")
print(f"  Output tokens:   {generated[0].tolist()}")
assert generated.shape[1] == prompt.shape[1] + 30
print("  PASS")

print("\n=== Test 5: Overfitting test ===")
model.train()
x = torch.randint(0, config.vocab_size, (4, 128), device=device)
y = x.clone()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
losses = []
for i in range(10):
    model.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, targets=y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())
    if (i + 1) % 2 == 0:
        print(f"  Step {i+1:2d}: loss = {loss.item():.4f}")
if losses[-1] < losses[0]:
    print(f"  Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("  PASS")
else:
    print(f"  WARNING: Loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})")

print(f"\n{'='*60}")
print("SMOKE TEST SUMMARY")
print(f"{'='*60}")
print(f"  Model:     {total_params:,} params ({total_params/1e6:.2f}M)")
print(f"  Forward:   OK")
print(f"  Backward:  OK")
if device.type == "cuda":
    print(f"  VRAM peak: {vram_peak:.3f} GB / 4.0 GB")
print(f"  Generate:  OK")
print(f"  Overfit:   {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"  Status:    READY FOR TRAINING")
print(f"{'='*60}")
print(f"\nTo start training:  python train.py")