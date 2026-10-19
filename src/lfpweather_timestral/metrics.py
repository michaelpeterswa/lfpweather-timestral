from prometheus_client import Counter, Gauge, Histogram

RUN_LAST_SUCCESS = Gauge(
    "timestral_run_last_success_timestamp_seconds",
    "Unix time of the last successful forecast run",
    ["source"],
)
# stage: history, nws, model, write, insufficient_history
RUN_FAILURES = Counter(
    "timestral_run_failures_total",
    "Forecast run failures by stage",
    ["stage"],
)
FORECAST_SECONDS = Histogram(
    "timestral_forecast_seconds",
    "Time spent producing a forecast",
)
