-- Vendored verbatim from timescale-migrations-timestral/migrations/022_create_sensors_temperature_forecast_table.up.sql
-- Hourly temperature forecasts for the home weather station, one row per
-- (source, issue time, target hour), so NWS and Timestral can be scored against
-- the same observations.
--
-- source is 'nws' (the National Weather Service hourly forecast), 'timestral'
-- (Mistral's Timestral model on the station's own history) or 'timestral_nws'
-- (Timestral with NWS forecast covariates). issued_at is when the forecast was
-- made and valid_at the hour it is for; lead_hours is their difference in whole
-- hours (1-24), stored so scoring can group by lead without recomputing it.
--
-- temperature is °F, the same units as sensors.vantagepro2plus.temperature.
-- For Timestral it is the median. quantiles holds Timestral's 21 quantile
-- values in ascending level order and is NULL for NWS. covariate_coverage is
-- the fraction of non-NaN NWS covariate values in the past window and is only
-- set for timestral_nws. model records the model version that produced the row.
CREATE TABLE IF NOT EXISTS
    sensors.temperature_forecast (
        issued_at TIMESTAMPTZ NOT NULL,
        source TEXT NOT NULL CHECK (source IN ('nws', 'timestral', 'timestral_nws')),
        valid_at TIMESTAMPTZ NOT NULL,
        lead_hours SMALLINT NOT NULL CHECK (lead_hours BETWEEN 1 AND 24),
        temperature DOUBLE PRECISION NOT NULL,
        quantiles REAL[],
        covariate_coverage REAL,
        model TEXT,
        PRIMARY KEY (source, issued_at, valid_at)
    );

SELECT
    create_hypertable ('sensors.temperature_forecast', 'issued_at', if_not_exists => TRUE);

-- Scoring joins forecasts to observations on the target hour, per source.
CREATE INDEX IF NOT EXISTS temperature_forecast_valid_at_source_idx ON sensors.temperature_forecast (valid_at, source);
