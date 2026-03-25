"""Tests for web viewer."""

import pytest
from fastapi.testclient import TestClient


class TestWebApp:
    def test_home_page(self, tmp_path, monkeypatch):
        """Home page loads without error."""
        monkeypatch.setenv("EC_DATA_DIR", str(tmp_path))

        # Create empty DB
        from extracosmic_commons.database import Database
        Database(tmp_path / "extracosmic.db").close()

        from extracosmic_commons.web.app import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "Extracosmic Commons" in response.text

    def test_api_stats(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EC_DATA_DIR", str(tmp_path))
        from extracosmic_commons.database import Database
        Database(tmp_path / "extracosmic.db").close()

        from extracosmic_commons.web.app import app
        client = TestClient(app)
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert "chunks" in data
