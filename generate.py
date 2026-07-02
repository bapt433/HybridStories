import argparse
import torch
import sys
from pathlib import Path
from tokenizers import Tokenizer
import yaml

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
from model import ModelConfig, HybridLM

def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--config", type=str, default=str(PROJECT_DIR / "config.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    icfg = cfg.get("inference", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or (PROJECT_DIR / icfg.get("checkpoint", "checkpoints/best.pt"))
    prompt = args.prompt or icfg.get("sample_prompt", cfg["generation"]["sample_prompt"])
    max_tokens = args.tokens or icfg.get("max_new_tokens", 100)
    temperature = args.temperature or icfg.get("temperature", 0.8)
    top_p = args.top_p or icfg.get("top_p", 0.9)

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        ckpt_dir = PROJECT_DIR / "checkpoints"
        if ckpt_dir.exists():
            print("Available checkpoints:")
            for f in ckpt_dir.glob("*.pt"):
                print(f"  {f}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config_dict = ckpt["config"]
    model_config = ModelConfig.from_dict(config_dict)

    model = HybridLM(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded (step={ckpt.get('step', '?')}, val_loss={ckpt.get('val_loss', '?')})")

    tok_path = PROJECT_DIR / "data" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    prompt_ids = tokenizer.encode(prompt).ids
    if len(prompt_ids) == 0:
        prompt_ids = [0]
    print(f"Prompt: {prompt}")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.amp.autocast("cpu", enabled=False):
            generated = model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

    text = tokenizer.decode(generated[0].tolist())
    print(f"\n{'='*60}")
    print(f"Generated text:")
    print(f"{'='*60}")
    print(text)
    print(f"{'='*60}")

if __name__ == "__main__":
    main()