from __future__ import annotations

import argparse
import json

from nlp_pipeline.config import load_config
from nlp_pipeline.logging_utils import configure_logging
from nlp_pipeline.pipeline import train_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NewsBot intelligence system entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and persist the model")
    train_parser.add_argument("--config", required=True, help="Path to pipeline config")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI service")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        manifest = train_pipeline(load_config(args.config))
        print(json.dumps(manifest, indent=2))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "nlp_pipeline.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
