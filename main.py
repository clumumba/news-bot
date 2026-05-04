#!/usr/bin/env python
"""
NewsBot Intelligence System - Unified CLI
Run tests, train models, serve API, and manage Docker deployment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a shell command and return exit code."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    return subprocess.call(cmd)


def run_tests() -> int:
    """Run pytest test suite."""
    return run_command(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        "Running Tests"
    )


def train_model(config: str = "configs/pipeline.yaml") -> int:
    """Train the ML model."""
    return run_command(
        [sys.executable, "src/nlp_pipeline/main.py", "train", "--config", config],
        "Training Model"
    )


def serve_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
) -> int:
    """Start the FastAPI server."""
    cmd = [sys.executable, "src/nlp_pipeline/main.py", "serve", "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    return run_command(cmd, f"Starting API Server (http://{host}:{port})")


def build_docker(tag: str = "newsbot") -> int:
    """Build Docker image."""
    return run_command(
        ["docker", "build", "-t", tag, "."],
        f"Building Docker Image ({tag})"
    )


def docker_compose(service: str = "api", detach: bool = True) -> int:
    """Run docker-compose."""
    cmd = ["docker-compose", "up"]
    if detach:
        cmd.append("-d")
    cmd.append(service)
    return run_command(cmd, f"Running Docker Compose ({service})")


# def push_docker(tag: str, registry: str) -> int:
#     """Push Docker image to registry."""
#     full_tag = f"{registry}/{tag}:latest"
#     print(f"\n{'='*60}")
#     print(f"▶ Pushing Docker Image to {registry}")
#     print(f"{'='*60}")
#     print(f"Make sure you're logged in: docker login {registry}")
#     print(f"Image: {full_tag}")
    
#     # Tag the image
#     subprocess.call(["docker", "tag", tag, full_tag])
#     # Push the image
#     return subprocess.call(["docker", "push", full_tag])


def status() -> int:
    """Check service status."""
    print(f"\n{'='*60}")
    print("▶ Service Status")
    print(f"{'='*60}")
    
    # Check if containers are running
    result = subprocess.call(["docker-compose", "ps"])
    
    # Check if API is responding
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("\n✅ API Health: OK")
            print(f"   Response: {response.json()}")
        else:
            print(f"\n⚠️  API Health: {response.status_code}")
    except Exception as e:
        print(f"\n❌ API Health: UNAVAILABLE ({e})")
    
    return 0


def full_setup() -> int:
    """Complete setup: test, train, build Docker, run."""
    steps = [
        ("Tests", run_tests),
        ("Training", train_model),
        ("Docker Build", lambda: build_docker()),
        ("Docker Compose", lambda: docker_compose()),
    ]
    
    for name, func in steps:
        code = func()
        if code != 0:
            print(f"\n❌ {name} failed with code {code}")
            return code
        print(f"✅ {name} passed")
    
    print(f"\n{'='*60}")
    print("✅ Full Setup Complete!")
    print(f"{'='*60}")
    print("\nAPI is running at http://localhost:8000")
    print("  - Health: http://localhost:8000/health")
    print("  - Docs: http://localhost:8000/docs")
    print("  - Metrics: http://localhost:8000/metrics")
    
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NewsBot Intelligence System - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test                    # Run tests
  python main.py train                   # Train model
  python main.py serve                   # Start API server
  python main.py serve --port 8080       # Start on custom port
  python main.py docker build            # Build Docker image
  python main.py docker compose          # Run with docker-compose
  python main.py docker push <user>      # Push to Docker Hub
  python main.py setup                   # Full setup (test, train, docker)
  python main.py status                  # Check service status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Test command
    subparsers.add_parser("test", help="Run pytest test suite")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the ML model")
    train_parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file path")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Docker command
    docker_parser = subparsers.add_parser("docker", help="Docker operations")
    docker_subparsers = docker_parser.add_subparsers(dest="docker_cmd", required=True)
    docker_subparsers.add_parser("build", help="Build Docker image")
    docker_subparsers.add_parser("compose", help="Run with docker-compose")
    push_parser = docker_subparsers.add_parser("push", help="Push to Docker registry")
    push_parser.add_argument("registry", help="Docker registry (e.g., yourusername)")
    push_parser.add_argument("--tag", default="newsbot", help="Image tag")
    
    # Status command
    subparsers.add_parser("status", help="Check service status")
    
    # Setup command
    subparsers.add_parser("setup", help="Full setup: test, train, build, run")
    
    args = parser.parse_args()
    
    # Ensure PYTHONPATH is set
    import os
    pythonpath = os.environ.get("PYTHONPATH", "")
    src_path = str(Path(__file__).parent / "src")
    if src_path not in pythonpath:
        os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{pythonpath}" if pythonpath else src_path
        # Re-run with updated PYTHONPATH
        return subprocess.call([sys.executable, __file__] + sys.argv[1:])
    
    # Execute command
    if args.command == "test":
        return run_tests()
    
    elif args.command == "train":
        return train_model(args.config)
    
    elif args.command == "serve":
        return serve_api(args.host, args.port, args.reload)
    
    elif args.command == "docker":
        if args.docker_cmd == "build":
            return build_docker()
        elif args.docker_cmd == "compose":
            return docker_compose()
        # elif args.docker_cmd == "push":
        #     return push_docker(args.tag, args.registry)
    
    elif args.command == "status":
        return status()
    
    elif args.command == "setup":
        return full_setup()
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
