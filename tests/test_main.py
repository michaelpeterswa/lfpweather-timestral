import psycopg
import pytest

from lfpweather_timestral import main
from lfpweather_timestral.config import Settings


def _settings(url="postgresql://x"):
    return Settings(database_url=url, nws_user_agent="test")


def test_connect_sets_timeouts(monkeypatch):
    seen = {}

    def fake_connect(url, **kwargs):
        seen["url"] = url
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    main.connect(_settings())
    assert seen["url"] == "postgresql://x"
    assert seen["kwargs"] == {
        "autocommit": True,
        "connect_timeout": 10,
        "options": "-c statement_timeout=60000",
    }


@pytest.mark.db
def test_statement_timeout_raises_instead_of_hanging(_timescale_url):
    conn = main.connect(_settings(_timescale_url), statement_timeout_ms=500)
    try:
        with pytest.raises(psycopg.errors.QueryCanceled):
            conn.execute("SELECT pg_sleep(10)")
    finally:
        conn.close()
