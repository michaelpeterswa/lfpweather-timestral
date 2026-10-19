# lfpweather-timestral

Forecasts a weather station's temperature hourly with Mistral's Timestral-209M and stores results in TimescaleDB.

## Pinned dependencies

- timestral: `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` (mistralai/timestral HEAD as of 2026-10-19)
- torch: PyTorch cu128 index on Linux (Blackwell needs CUDA >= 12.8), PyPI on macOS.

## Development

`uv sync && uv run pytest`

## Configuration

- `API_KEY` (optional): when set, `POST /forecast` (port 8080) requires an `X-API-Key` header equal to it and returns 401 otherwise. When unset the endpoint is open (development only). `/healthcheck` and `/metrics` on port 8081 are never authenticated.
