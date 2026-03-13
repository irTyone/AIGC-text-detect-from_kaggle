import torch
import pandas as pd
import numpy as np
import asyncio
import traceback

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)

from datasets import Dataset
from fastapi import HTTPException

from app.utils.logger import get_logger
from app.error_code import *
from datetime import datetime

logger = get_logger()
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

class HFModelEngine:

    def __init__(self, model_path: str, max_length: int = 128, batch_size: int = 8):

        logger.info("Initializing model engine")

        self.device = "cpu"
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_char_length = 5000

        try:

            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info("Loading model")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=1
            )

            self.model.to(self.device)
            self.model.eval()

            self.data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding="longest"
            )

            logger.info("Model loaded successfully")

        except Exception:

            logger.error(traceback.format_exc())

            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": MODEL_LOAD_ERROR,
                    "message": "model load failed"
                }
            )

    def preprocess_function(self, examples):

        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length
        )

    async def predict(self, texts: list):

        logger.info(f"Received {len(texts)} texts")

        try:

            # =====================
            # 长度检查
            # =====================

            for text in texts:

                if len(text) > self.max_char_length:

                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error_code": TEXT_TOO_LONG_CHAR,
                            "message": f"text too long (char) {len(text)} > {self.max_char_length}"
                        }
                    )

                tokens = self.tokenizer(text)

                if len(tokens["input_ids"]) > self.max_length:

                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error_code": TEXT_TOO_LONG_TOKEN,
                            "message": f"text too long (token) {len(tokens['input_ids'])} > {self.max_length}"
                        }
                    )

            # =====================
            # 构造数据
            # =====================

            df = pd.DataFrame({"text": texts})
            df["id"] = np.arange(len(df))

            ds = Dataset.from_pandas(df)

            tokenized_ds = ds.map(
                self.preprocess_function,
                batched=True,
                remove_columns=ds.column_names
            )

            # =====================
            # 推理
            # =====================

            loop = asyncio.get_event_loop()

            pred_output = await loop.run_in_executor(
                None,
                self._sync_predict,
                tokenized_ds
            )

            logits = pred_output.predictions.astype(float)

            scores = -1 * logits[:, 0]

            probs = 1 / (1 + np.exp(-scores))

            results = [
                {
                
                    "score": float(scores[i]),
                    "probability": float(probs[i]),
                    "is_ai": bool(probs[i] > 0.5)
                }
                for i in range(len(df))
            ]

            return results

        except HTTPException:
            raise

        except Exception:

            logger.error(traceback.format_exc())

            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": MODEL_INFERENCE_ERROR,
                    "message": "model inference failed"
                }
            )

    def _sync_predict(self, dataset):

        training_args = TrainingArguments(
            output_dir="tmp",
            per_device_eval_batch_size=self.batch_size,
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        return trainer.predict(dataset)

# ======================
# CPU 测试 main
# ======================
if __name__ == "__main__":
    import asyncio
    
    model_path = "/home/liuyuan/AIGC-text-detect/archive/checkpoint-6250"
    print(model_path)
    engine = HFModelEngine(model_path=model_path, max_length=1024, batch_size=4)

    sample_texts = [
        "In this article we analyze the impact of B-physics and Higgs physics at LEPon standard and non-standard Higgs bosons searches at the Tevatron and the LHC,within the framework of minimal flavor violating supersymmetric models. TheB-physics constraints we consider come from the experimental measurements ofthe rare B-decays b -> s gamma and B_u -> tau nu and the experimental limit onthe B_s -> mu+ mu- branching ratio. We show that these constraints are severefor large values of the trilinear soft breaking parameter A_t, rendering thenon-standard Higgs searches at hadron colliders less promising. On the contrarythese bounds are relaxed for small values of A_t and large values of theHiggsino mass parameter mu, enhancing the prospects for the direct detection ofnon-standard Higgs bosons at both colliders. We also consider the availableATLAS and CMS projected sensitivities in the standard model Higgs searchchannels, and we discuss the LHC's ability in probing the whole MSSM parameterspace. In addition we also consider the expected Tevatron collidersensitivities in the standard model Higgs h -> b bbar channel to show that itmay be able to find 3 sigma evidence in the B-physics allowed regions for smallor moderate values of the stop mixing parameter",
        "这是第二条测试文本",
        "这是第三条测试文本"
    ]

    async def test_cpu():
        log("Starting test_cpu...")
        results = await engine.predict(sample_texts)
        for r in results:
            log(
                f"score={r['score']:.4f} | "
                f"prob={r['probability']:.4f} | "
                f"is_ai={r['is_ai']}"
                )

    asyncio.run(test_cpu())