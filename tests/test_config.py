from lfpweather_timestral.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.setenv("NWS_USER_AGENT", "test (a@b.c)")
    s = Settings()
    assert s.grid_point == "SEW/127,75"
    assert s.history_days == 30
    assert s.require_cuda is True


def test_api_key_optional(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.setenv("NWS_USER_AGENT", "test (a@b.c)")
    monkeypatch.delenv("API_KEY", raising=False)
    assert Settings().api_key is None
    monkeypatch.setenv("API_KEY", "k")
    assert Settings().api_key == "k"


def test_empty_api_key_is_unset(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.setenv("NWS_USER_AGENT", "test (a@b.c)")
    for raw in ("", "   "):
        monkeypatch.setenv("API_KEY", raw)
        assert Settings().api_key is None
