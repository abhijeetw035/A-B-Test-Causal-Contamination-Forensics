"""FastAPI application entrypoint for the environment API."""

from fastapi import FastAPI

from api.routes import router


app = FastAPI(title="A/B Test Causal Contamination Forensics", version="0.1.0")
app.include_router(router)
