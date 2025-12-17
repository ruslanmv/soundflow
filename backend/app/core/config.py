from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # R2 / S3
    r2_endpoint: str
    r2_bucket: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_region: str = "auto"

    # Catalog object keys in bucket
    catalog_free_key: str = "catalog/free-index.json"
    catalog_premium_key: str = "catalog/premium-index.json"

    # Security
    admin_api_key: str = ""
    premium_api_key: str = ""  # optional simple premium gating

    # CORS
    cors_origins: str = "*"  # comma-separated or "*"

    # Signed URL TTL (seconds)
    signed_url_ttl_sec: int = 900  # 15 minutes

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
