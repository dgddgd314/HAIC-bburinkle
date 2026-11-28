"""Exercise the real M4 processor, training and evaluation on a tiny random model."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaConfig, PaliGemmaForConditionalGeneration

from dataset import DOORS, ROOMS, make_split, partition, surveys
from evaluate import evaluate
from model import ANSWERS, inputs_for, load_checkpoint, load_scorer
from train import main as train


def check_processor(processor):
    """Verify the target tokenizer's supervised suffix and the native image size."""
    image = Image.new("RGB", (64, 48))
    for room, answer in ANSWERS.items():
        inputs = inputs_for(processor, image, room, "cpu", torch.float32)
        assert inputs.pixel_values.shape == (1, 3, 224, 224)
        assert (inputs.input_ids == processor.image_token_id).sum().item() == 256
        labels = inputs.labels[inputs.labels != -100].tolist()
        assert labels[-1] == processor.tokenizer.eos_token_id
        assert processor.tokenizer.decode(labels[:-1]).strip() == answer


def make_tiny_checkpoint(source, dest, dtype):
    """Keep the target tokenizer and Gemma 2 architecture with one small layer."""
    proc = AutoProcessor.from_pretrained(source, local_files_only=True)
    check_processor(proc)
    proc.image_processor.size = {"height": 28, "width": 28}
    proc.image_processor.image_seq_length = 4
    cfg = PaliGemmaConfig.from_pretrained(source, local_files_only=True)
    cfg.projection_dim = 16
    for sub in (cfg.vision_config, cfg.text_config):
        sub.hidden_size = 16
        sub.intermediate_size = 32
        sub.num_hidden_layers = 1
        sub.num_attention_heads = 2
    cfg.vision_config.image_size = 28
    cfg.vision_config.projection_dim = 16
    cfg.text_config.num_image_tokens = 4
    cfg.text_config.head_dim = 8
    cfg.text_config.num_key_value_heads = 1
    PaliGemmaForConditionalGeneration(cfg).to(dtype).save_pretrained(dest)
    proc.save_pretrained(dest)


def main():
    """Run a complete update and verify exported weights and report counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processor", type=Path, help="224 config and processor files")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        base, tuned = root / "base", root / "tuned"
        dtype = torch.float32
        if args.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        make_tiny_checkpoint(args.processor, base, dtype)
        for run in ("a", "b"):
            batch = root / run
            batch.mkdir()
            labels = dict(zip(DOORS, ROOMS, strict=True))
            for door, color in zip(DOORS, ("red", "green", "blue"), strict=True):
                Image.new("RGB", (40, 30), color).save(batch / f"{door}.jpg")
            (batch / "survey.json").write_text(json.dumps(
                {"run_id": run, "batch_id": run, "rooms": labels}), encoding="utf-8")
        split = root / "split.json"
        split.write_text(json.dumps(make_split(surveys(root), 42)), encoding="utf-8")
        held = partition(root, split, "eval")
        first = evaluate(root, held, load_scorer(base, device=args.device))
        train([str(root), str(split), str(base), str(tuned), "--device", args.device, "--epochs", "1"])
        second = evaluate(root, held, load_scorer(tuned, device=args.device))
        for report in (first, second):
            assert report["layouts"] == 1
            assert all(sum(row.values()) == 1 for row in report["confusion_true_by_pred"].values())
        _, before = load_checkpoint(base, "cpu", torch.float32)
        _, after = load_checkpoint(tuned, "cpu", torch.float32)
        changed = [name for name, p in after.named_parameters()
                   if not torch.equal(p, before.get_parameter(name))]
        assert changed and all("multi_modal_projector" in name for name in changed), changed
    print(f"PASS: target processor, {args.device} training, projector update, export, reload and scoring")


if __name__ == "__main__":
    main()
