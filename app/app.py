from fastapi import FastAPI
import pandas as pd
import yaml
import os
from .ml.classifier import clf
from .ml.pipeline import OneShotModel
from .models import ClassifyRequest, ClassifyResponse, TaskResult, ChunkResult
from .utils.preprocess import clean_email
from .utils.text_utils import join_subject_body

app = FastAPI(title="Email Classification API")

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "resources", "prompts.yaml")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    MODEL_DICT = yaml.safe_load(f)


@app.post("/classify", response_model=ClassifyResponse)
def classify_email(request: ClassifyRequest):
    """
    Run email classification across all configured tasks.

    Parameters
    ----------
    request : ClassifyRequest
        The incoming request containing the email subject and body to be classified.

    Returns
    -------
    ClassifyResponse
        A structured response containing, for each classification task, the selected label, its score, and chunk-level prediction details.
    """
    full_text = join_subject_body(request.subject, request.body)
    full_text = clean_email(full_text)

    df = pd.DataFrame({"full_text": [full_text]})
    model = OneShotModel(
        df=df,
        text_col="full_text",
        model_dict=MODEL_DICT,
        clf_pipe=clf,
        display_time=False
    )

    results = []
    for task_idx, _ in enumerate(MODEL_DICT):
        cache = model._chunk_cache[task_idx][df.index[0]]
        chunks_out = [
            ChunkResult(
                chunk=c,
                top_label=r["labels"][0],
                top_score=float(r["scores"][0]),
            )
            for c, r in zip(cache["chunks"], cache["results"])
        ]
        results.append(
            TaskResult(
                task_index=task_idx,
                selected_label=cache["selected_label"],
                selected_score=None if pd.isna(cache["selected_score"]) else float(cache["selected_score"]),
                chunks=chunks_out,
            )
        )

    return ClassifyResponse(results=results)