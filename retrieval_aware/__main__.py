"""Safe bootstrap entry point for the E7 experiment workspace."""

from __future__ import annotations

import json

from .config import E7Config
from .protocol import PROTOCOL_SPECS, validate_protocol_registry
from .utils import ensure_output_layout


def main() -> None:
    """Validate E7 paths and create only the isolated output layout."""
    config = E7Config()
    config.validate()
    validate_protocol_registry()
    paths = ensure_output_layout(config)
    print(
        json.dumps(
            {
                "status": "skeleton_ready",
                "config": config.to_dict(),
                "output_directories": {
                    name: str(path) for name, path in paths.items()
                },
                "protocols": [protocol.value for protocol in PROTOCOL_SPECS],
                "retrieval_or_llm_calls_made": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
