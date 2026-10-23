"""
# Fases 2 e 3 — Ponto de entrada, inclusão dos routers e ciclo de vida (lifespan)

Evolução do `main.py` do código base: lá os dados eram gerados no import do
módulo (`products = generate_products()`), em escopo global. Aqui o modelo é
carregado UMA única vez dentro do `lifespan` e guardado em `app.state`,
liberado no desligamento. Nenhuma rota lê o .joblib do disco.

Executar:  uvicorn app.main:app --reload
"""
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.config import APP_NAME, APP_VERSION, METRICS_PATH, MODEL_PATH
from app.routers import health, predict
from app.services.model_service import ModelService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- startup: roda 1 vez antes de aceitar requisições ----------
    service = ModelService(MODEL_PATH)
    try:
        service.load()
    except Exception:
        # A API sobe mesmo assim: /health mostra "degraded" e /v1/predict devolve 503
        logger.exception("Não foi possível carregar o modelo")
    app.state.model_service = service
    app.state.metrics = (json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                         if METRICS_PATH.exists() else None)
    yield
    # ---------- shutdown: libera a memória ----------
    service.unload()
    logger.info("Modelo descarregado. API encerrada.")


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API REST para prever cancelamento (churn) de clientes de telecom.",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", tags=["Monitoramento"], summary="Boas-vindas")
def root() -> dict:
    return {"api": APP_NAME, "version": APP_VERSION, "docs": "/docs", "health": "/health"}


@app.get("/v1/model/info", tags=["Inferência"], summary="Métricas do modelo em produção")
def model_info(request: Request) -> dict:
    if request.app.state.metrics is None:
        return JSONResponse(status_code=404, content={"detail": "metrics.json não encontrado."})
    return request.app.state.metrics


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Rede de segurança: qualquer erro não previsto vira 500 em JSON, sem derrubar a API."""
    logger.exception("Erro não tratado em %s", request.url.path)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"detail": "Erro interno do servidor."})
