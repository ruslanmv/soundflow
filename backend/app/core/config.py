# backend/app/core/config.py
from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Production-ready settings with safe local defaults.

    Key change vs your current file:
    - R2 settings are OPTIONAL unless STORAGE_MODE=r2
    - This prevents import-time crashes in local dev / CI
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,  # makes Windows/Linux env handling less painful
    )

    # Environment
    env: str = Field(default="dev", alias="ENV")

    # Storage
    # local = no R2 required (uses local dev stubs / sample catalogs)
    # r2    = Cloudflare R2 (S3 compatible) is required
    storage_mode: str = Field(default="local", alias="STORAGE_MODE")  # local | r2

    # R2 / S3 (required only if STORAGE_MODE=r2)
    r2_endpoint: Optional[str] = Field(default=None, alias="R2_ENDPOINT")
    r2_bucket: Optional[str] = Field(default=None, alias="R2_BUCKET")
    r2_access_key_id: Optional[str] = Field(default=None, alias="R2_ACCESS_KEY_ID")
    r2_secret_access_key: Optional[str] = Field(default=None, alias="R2_SECRET_ACCESS_KEY")
    r2_region: str = Field(default="auto", alias="R2_REGION")

    # Catalog object keys in bucket (or local dev equivalents)
    catalog_free_key: str = Field(default="catalog/free-index.json", alias="CATALOG_FREE_KEY")
    catalog_premium_key: str = Field(default="catalog/premium-index.json", alias="CATALOG_PREMIUM_KEY")

    # Security
    admin_api_key: str = Field(default="", alias="ADMIN_API_KEY")
    premium_api_key: str = Field(default="", alias="PREMIUM_API_KEY")  # optional simple premium gating
    signing_secret: str = Field(default="dev-secret-change-me", alias="SIGNING_SECRET")

    # CORS
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")  # comma-separated or "*"

    # Signed URL TTL (seconds)
    signed_url_ttl_sec: int = Field(default=900, alias="SIGNED_URL_TTL_SEC")  # 15 minutes

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def r2_required(self) -> bool:
        return self.storage_mode.strip().lower() == "r2"

    def validate_r2_or_raise(self) -> None:
        """
        Call this ONLY when you actually use R2.
        This avoids boot-time failures in local/dev/CI.
        """
        if not self.r2_required():
            return

        missing = []
        if not self.r2_endpoint:
            missing.append("R2_ENDPOINT")
        if not self.r2_bucket:
            missing.append("R2_BUCKET")
        if not self.r2_access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not self.r2_secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")

        if missing:
            raise RuntimeError(
                "R2 is enabled (STORAGE_MODE=r2) but required env vars are missing: "
                + ", ".join(missing)
            )


settings = Settings()
