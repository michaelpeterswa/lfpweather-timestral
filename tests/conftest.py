from pathlib import Path

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer

SQL_DIR = Path(__file__).parent / "sql"
MIGRATIONS = [
    "001_create_sensors_schema.up.sql",
    "006_create_sensors_vantagepro2plus_table.up.sql",
    "022_create_sensors_temperature_forecast_table.up.sql",
    "023_create_temperature_hourly_function.up.sql",
]


@pytest.fixture(scope="session")
def _timescale_url():
    with PostgresContainer(
        "timescale/timescaledb:latest-pg16", driver=None, dbname="weather"
    ) as pg:
        yield pg.get_connection_url()


@pytest.fixture
def db_conn(_timescale_url):
    """psycopg 3 connection to a fresh database with the sensors schema applied."""
    admin = psycopg.connect(_timescale_url, autocommit=True)
    name = f"t_{id(admin):x}"
    admin.execute(f'CREATE DATABASE "{name}"')
    url = _timescale_url.rsplit("/", 1)[0] + "/" + name
    conn = psycopg.connect(url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    for m in MIGRATIONS:
        conn.execute((SQL_DIR / m).read_text())
    try:
        yield conn
    finally:
        conn.close()
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.close()
