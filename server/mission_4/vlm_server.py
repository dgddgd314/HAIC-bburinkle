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
    parser.add_argument("--projector", help="Apply the team's projector.safetensors to the base checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8044)
    args = parser.parse_args()

    from model import load_scorer

    score = load_scorer(args.checkpoint, projector=args.projector)

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
        scores = score(image)
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps(scores, allow_nan=False).encode()]

    with make_server(args.host, args.port, score_request) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
