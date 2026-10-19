from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", protected_namespaces=("settings_",))

    database_url: str
    grid_point: str = "SEW/127,75"
    nws_user_agent: str
    history_days: int = 30
    model_id: str = "mistralai/Timestral-209M"
    require_cuda: bool = True
    schedule_enabled: bool = True
    api_port: int = 8080
    metrics_port: int = 8081
    log_level: str = "info"
