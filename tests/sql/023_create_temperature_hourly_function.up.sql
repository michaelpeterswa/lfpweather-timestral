-- Vendored verbatim from timescale-migrations-timestral/migrations/023_create_temperature_hourly_function.up.sql
-- Hourly observed temperature, the ground truth forecasts are scored against.
--
-- Each row is labelled with the top of an hour and is the mean of the raw
-- readings within 30 minutes either side of it (a centered window: the 10:00
-- row averages 09:30 <= time < 10:30). A forecast for 10:00 is a point estimate
-- for that instant, so a centered mean is a fairer target than the mean of
-- 10:00-11:00. Hours with no readings produce no row.
--
-- The time_bucket origin offset of -30 minutes shifts buckets to start at :30,
-- and the +30 minutes puts the label back on the hour. The filter is on raw
-- time so it can use the hypertable's chunk and time indexes.
CREATE OR REPLACE FUNCTION sensors.temperature_hourly (from_ts TIMESTAMPTZ, to_ts TIMESTAMPTZ) RETURNS TABLE (valid_at TIMESTAMPTZ, temperature DOUBLE PRECISION) LANGUAGE sql STABLE AS $$
    SELECT
        time_bucket('1 hour', v.time, '-30 minutes'::interval) + interval '30 minutes' AS valid_at,
        avg(v.temperature)::double precision AS temperature
    FROM
        sensors.vantagepro2plus v
    WHERE
        v.time >= from_ts - interval '30 minutes'
        AND v.time < to_ts + interval '30 minutes'
    GROUP BY
        1
    ORDER BY
        1
$$;
