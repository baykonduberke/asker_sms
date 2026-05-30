from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "FastAPI"
    app_env: str = "development"

    openrouter_api_key: str
    openrouter_model: str

    postgres_url: str

settings = Settings()