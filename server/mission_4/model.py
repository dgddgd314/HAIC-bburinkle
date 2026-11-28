"""Shared M4 processor, model loading and deployed candidate scoring."""
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

PROMPT = "answer en Is this room a classroom, an infirmary, or a cafeteria?"
ANSWERS = {"classroom": "classroom", "medical": "infirmary", "cafeteria": "cafeteria"}


def load_checkpoint(path, device, dtype):
    """Open the processor and full model from a local checkpoint."""
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    return processor, model


def inputs_for(processor, image, room, device, dtype):
    """Mask the prompt and supervise the server's exact candidate answer."""
    inputs = processor(images=image, text=PROMPT, suffix=ANSWERS[room], return_tensors="pt")
    inputs = inputs.to(device=device)
    inputs["labels"] = inputs["labels"].long()
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    return inputs


def load_scorer(path, device=None, projector=None):
    """Load the scorer, optionally replacing the base model's visual projector."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor, model = load_checkpoint(path, device, dtype)
    if projector is not None:
        from safetensors.torch import load_file

        prefix = "multi_modal_projector."
        weights = load_file(str(projector))
        expected = {prefix + key for key in model.multi_modal_projector.state_dict()}
        if set(weights) != expected:
            raise ValueError("Projector file must contain exactly the visual projector tensors")
        model.multi_modal_projector.load_state_dict(
            {key[len(prefix):]: value for key, value in weights.items()}, strict=True)

    @torch.inference_mode()
    def score(image: Image.Image):
        """Score each answer independently by negative mean answer-token loss."""
        scores = {}
        for room in ANSWERS:
            inputs = inputs_for(processor, image, room, device, dtype)
            scores[room] = -model(**inputs, use_cache=False).loss.item()
        return scores

    return score
