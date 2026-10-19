from lfpweather_timestral.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.setenv("NWS_USER_AGENT", "test (a@b.c)")
    s = Settings()
    assert s.grid_point == "SEW/127,75"
    assert s.history_days == 30
    assert s.require_cuda is True
