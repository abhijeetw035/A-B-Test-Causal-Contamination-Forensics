"""OpenEnv-compatible server entrypoint forwarding to FastAPI application."""

from __future__ import annotations

import uvicorn

from api.app import app


def main() -> None:
	"""Run the FastAPI app on the expected OpenEnv-compatible port."""
	uvicorn.run("api.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
	main()


__all__ = ["app", "main"]
