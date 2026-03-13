from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback

from app.model import HFModelEngine
from app.utils.logger import get_logger

logger = get_logger()

app = FastAPI(title="HF CPU Model Inference API")


# =========================
# 初始化模型
# =========================

model_engine = HFModelEngine(
    model_path="/home/liuyuan/AIGC-text-detect/archive/checkpoint-6250",
    max_length=1024,
    batch_size=8
)



class PredictRequest(BaseModel):
    texts: str


class PredictResponse(BaseModel):
    results: list



@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):

    logger.info("Received API request")

    try:

        results = await model_engine.predict([req.texts])

        return PredictResponse(results=results)

    except HTTPException as e:

        logger.warning(f"Request error: {e.detail}")

        raise e

    except Exception:

        logger.error("Unexpected API error")
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail="internal server error"
        )