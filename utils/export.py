from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Exporter:
    output_dir: Path = Path("reports")

    def export_json(self, payload: dict, filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def export_markdown(self, title: str, sections: list[tuple[str, str]], filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        lines = [f"# {title}", ""]
        for heading, body in sections:
            lines.extend([f"## {heading}", body, ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

