"""Unit tests for api/config.py — settings, SECRET_KEY check, CORS parsing."""

import os
import sys
from unittest.mock import patch

import pytest

from codeverify_api.config import Settings


class TestCorsOriginsParsing:
    """Test CORS_ORIGINS field_validator."""

    def test_parses_comma_separated_string(self):
        s = Settings(CORS_ORIGINS="http://a.com, http://b.com")
        assert s.CORS_ORIGINS == ["http://a.com", "http://b.com"]

    def test_parses_single_origin_string(self):
        s = Settings(CORS_ORIGINS="http://localhost:3000")
        assert s.CORS_ORIGINS == ["http://localhost:3000"]

    def test_accepts_list_directly(self):
        s = Settings(CORS_ORIGINS=["http://a.com", "http://b.com"])
        assert s.CORS_ORIGINS == ["http://a.com", "http://b.com"]

    def test_strips_whitespace(self):
        s = Settings(CORS_ORIGINS="  http://a.com ,  http://b.com  ")
        assert s.CORS_ORIGINS == ["http://a.com", "http://b.com"]

    def test_default_origins(self):
        s = Settings()
        assert "http://localhost:3000" in s.CORS_ORIGINS
        assert "http://localhost:8000" in s.CORS_ORIGINS


class TestSecretKeyProductionCheck:
    """Test get_settings() SECRET_KEY safety check."""

    def test_development_allows_default_secret(self):
        s = Settings(ENVIRONMENT="development", SECRET_KEY="change-this-in-production")
        assert s.SECRET_KEY == "change-this-in-production"

    def test_production_exits_with_default_secret(self):
        """get_settings() should sys.exit(1) when production uses default secret."""
        from codeverify_api.config import get_settings

        # Clear the lru_cache so get_settings() re-executes
        get_settings.cache_clear()

        with (
            patch.dict(os.environ, {
                "ENVIRONMENT": "production",
                "SECRET_KEY": "change-this-in-production",
            }),
            pytest.raises(SystemExit) as exc_info,
        ):
            get_settings()

        assert exc_info.value.code == 1

        # Restore cache for subsequent tests
        get_settings.cache_clear()

    def test_production_works_with_custom_secret(self):
        """Non-default SECRET_KEY should be accepted in production."""
        from codeverify_api.config import get_settings

        get_settings.cache_clear()

        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "my-secure-random-key-12345",
        }):
            s = get_settings()
            assert s.SECRET_KEY == "my-secure-random-key-12345"

        get_settings.cache_clear()


class TestSettingsDefaults:
    """Test default values are sane."""

    def test_default_environment(self):
        s = Settings()
        assert s.ENVIRONMENT == "development"

    def test_default_jwt_algorithm(self):
        s = Settings()
        assert s.JWT_ALGORITHM == "HS256"

    def test_default_jwt_expiration(self):
        s = Settings()
        assert s.JWT_EXPIRATION_HOURS == 24

    def test_default_api_port(self):
        s = Settings()
        assert s.API_PORT == 8000
