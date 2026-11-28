"""Fine-tune the PaliGemma visual projector on M4 practice JPEGs."""
import argparse
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from dataset import DOORS, partition


def main(argv=None):
    """Optimize answer-token loss on train runs and export a full local checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists() or args.epochs < 1 or not args.lr > 0:
        parser.error("Supply a new output path, positive epochs and learning rate")

    import torch
    from model import inputs_for, load_checkpoint

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    rows = partition(args.data, args.split, "train")
    samples = [(args.data / r["batch_id"] / f"{door}.jpg", r["rooms"][door])
               for r in rows for door in DOORS]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(device, dtype, len(samples), "training images", flush=True)
    processor, model = load_checkpoint(args.base, device, dtype)
    model.requires_grad_(False)
    projector = model.multi_modal_projector.float().requires_grad_(True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    for epoch in range(args.epochs):
        total = 0.0
        rng.shuffle(samples)
        for path, room in samples:
            with Image.open(path) as source:
                inputs = inputs_for(processor, source.convert("RGB"), room, device, dtype)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
                loss = model(**inputs, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss: {path}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        print(f"Epoch {epoch + 1}: mean loss {total / len(samples):.5f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        export = Path(temp) / "checkpoint"
        model.to(dtype).save_pretrained(export)
        processor.save_pretrained(export)
        export.rename(args.output)


if __name__ == "__main__":
    main()
