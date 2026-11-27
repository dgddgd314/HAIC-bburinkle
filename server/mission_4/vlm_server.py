"""Serve M4 doorway scores from a local PaliGemma checkpoint."""
import argparse
import io
import json
from wsgiref.simple_server import make_server

from PIL import Image


def main():
    """Load the checkpoint once and serve candidate answer scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Local PaliGemma mix checkpoint directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8044)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, torch_dtype=dtype).to(device).eval()

    def score_request(env, start_response):
        """Return mean answer-token log likelihoods for the submitted view."""
        if env["REQUEST_METHOD"] != "POST" or env["PATH_INFO"] != "/scores":
            start_response("404 Not Found", [])
            return [b"POST a JPEG to /scores"]
        size = int(env.get("CONTENT_LENGTH") or 0)
        if not 0 < size <= 4_000_000:
            start_response("400 Bad Request", [])
            return [b"Invalid image size"]
        with Image.open(io.BytesIO(env["wsgi.input"].read(size))) as source:
            image = source.convert("RGB")
        prompt = "answer en Is this room a classroom, an infirmary, or a cafeteria?"
        scores = {}
        with torch.inference_mode():
            for room, answer in (("classroom", "classroom"), ("medical", "infirmary"),
                                 ("cafeteria", "cafeteria")):
                inputs = processor(images=image, text=prompt, suffix=answer, return_tensors="pt")
                inputs = inputs.to(device=device)
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                scores[room] = -model(**inputs, use_cache=False).loss.item()
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps(scores, allow_nan=False).encode()]

    with make_server(args.host, args.port, score_request) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
