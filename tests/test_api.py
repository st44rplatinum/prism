"""Serving-path tests.

These exist because of a failure the unit tests could not see. The saturation
route asked for the trained head before it looked in the cache, so on a fresh
clone - which ships 183 precomputed probability matrices but no feature cache,
and therefore cannot construct a predictor - every one of those matrices came
back 503. The data was on disk and readable; the route simply refused to reach
it. Nothing in a pure-unit suite catches a route requiring more than it uses.

They run against the committed artefacts only: proteins.json, variants.parquet
and the saturation cache. No GPU, no network, no feature cache.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient       # noqa: E402


@pytest.fixture
def client(monkeypatch):
    """The app as a fresh clone sees it: artefacts present, no model loadable."""
    monkeypatch.setenv("VEP_NO_WARM", "1")
    # Constructing a VariantPredictor needs the 250 MB feature cache, which is
    # not committed. Returning None is exactly what the real store does there.
    monkeypatch.setattr("api.store.Store.predictor", lambda self: None)

    from api.main import app

    with TestClient(app) as c:
        yield c


class TestServesWithoutAModel:
    def test_cached_probability_matrix_is_served(self, client):
        r = client.get("/genes/TP53/saturation", params={"value": "probability"})
        assert r.status_code == 200, r.text

        body = r.json()
        assert body["value"] == "probability"
        assert body["gene"] == "TP53"
        assert len(body["matrix"]) == body["length"]

        cells = [v for row in body["matrix"] for v in row if v is not None]
        assert cells, "matrix came back empty"
        # Calibrated probabilities, so anything outside [0, 1] means the wrong
        # array was read or the scale was mangled in transport.
        assert 0.0 <= min(cells) and max(cells) <= 1.0

    def test_uncached_probability_still_needs_the_model(self, client, monkeypatch):
        """The gate is correct, it was just in the wrong place."""
        monkeypatch.setattr("api.store.Store.get_saturation",
                            lambda self, symbol, kind: None)
        r = client.get("/genes/TP53/saturation", params={"value": "probability"})
        assert r.status_code == 503
        assert "no pathogenicity model loaded" in r.json()["detail"]

    def test_catalog_endpoints_need_no_model(self, client):
        for path in ["/genes", "/genes/TP53", "/metrics", "/health"]:
            assert client.get(path).status_code == 200, path

    def test_unknown_gene_is_404_not_500(self, client):
        assert client.get("/genes/NOTAGENE/saturation").status_code == 404
