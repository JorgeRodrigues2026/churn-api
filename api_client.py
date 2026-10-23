"""
Cliente REST da Churn Prediction API.

O dashboard NUNCA carrega o modelo diretamente: toda predição passa pela API
via HTTP (mesmo contrato que qualquer aplicação externa usaria). Esta classe
isola a comunicação HTTP da interface Streamlit.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BATCH_LIMIT = 100  # mesmo limite do schema BatchRequest da API


@dataclass
class ApiResult:
    ok: bool
    status_code: int | None
    data: Any = None
    elapsed_ms: float = 0.0
    error: str | None = None
    request_payload: Any = field(default=None, repr=False)


class ChurnApiClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json",
                                     "Accept": "application/json"})

    # ---------- núcleo ----------
    def _request(self, method: str, path: str, payload: Any = None) -> ApiResult:
        url = f"{self.base_url}{path}"
        start = time.perf_counter()
        try:
            resp = self.session.request(method, url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            return ApiResult(False, None, error=f"Não foi possível conectar em {self.base_url}. "
                                                "A API está rodando?", request_payload=payload)
        except requests.exceptions.Timeout:
            return ApiResult(False, None, error="Tempo de resposta excedido.", request_payload=payload)

        elapsed = (time.perf_counter() - start) * 1000
        try:
            data = resp.json()
        except ValueError:
            data = {"detail": resp.text}

        error = None
        if not resp.ok:
            detail = data.get("detail") if isinstance(data, dict) else data
            error = self._format_detail(detail)
        return ApiResult(resp.ok, resp.status_code, data, round(elapsed, 1), error, payload)

    @staticmethod
    def _format_detail(detail: Any) -> str:
        # 422 do Pydantic vem como lista de erros -> transforma em texto legível
        if isinstance(detail, list):
            parts = []
            for err in detail:
                loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
                parts.append(f"{loc}: {err.get('msg')}")
            return " | ".join(parts)
        return str(detail)

    # ---------- endpoints ----------
    def health(self) -> ApiResult:
        return self._request("GET", "/health")

    def model_info(self) -> ApiResult:
        return self._request("GET", "/v1/model/info")

    def predict(self, customer: dict) -> ApiResult:
        return self._request("POST", "/v1/predict", customer)

    def predict_batch(self, customers: list[dict]) -> ApiResult:
        return self._request("POST", "/v1/predict/batch", {"customers": customers})

    def predict_many(self, customers: list[dict], progress_cb=None) -> ApiResult:
        """Divide listas grandes em lotes de até 100 (limite da API) e junta as respostas."""
        predictions: list[dict] = []
        total_ms = 0.0
        chunks = range(0, len(customers), BATCH_LIMIT)
        for i, start in enumerate(chunks, 1):
            res = self.predict_batch(customers[start:start + BATCH_LIMIT])
            if not res.ok:
                res.error = f"Lote {i} (linhas {start + 1}–{start + BATCH_LIMIT}): {res.error}"
                return res
            predictions.extend(res.data["predictions"])
            total_ms += res.elapsed_ms
            if progress_cb:
                progress_cb(i / len(chunks))
        return ApiResult(True, 200, {"total": len(predictions), "predictions": predictions},
                         round(total_ms, 1))
