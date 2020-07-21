import os
import sys

sys.path.append(".")

from fastapi import FastAPI, Path
from fastapi.responses import RedirectResponse

from http import HTTPStatus
import json

from pydantic import BaseModel
import wandb

from text_classification import config, data, predict, utils


app = FastAPI(
    title="app",
    description="PyTorch Tutorial for Text Classification",
    version="1.0.0",
)

best_run = utils.get_best_run(project="mahjouri-saamahn/mwml-tutorial-app",
                              metric="test_loss", objective="minimize")
best_run_dir = utils.load_run(run=best_run)

model, word_map = predict.get_run_components(run_dir=best_run_dir)


@utils.construct_response
@app.get("/")
async def _index():
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {}
    }
    config.logger.info(json.dumps(response, indent=2))
    return response


@app.get("/experiments")
async def _experiments():
    return RedirectResponse("https://app.wandb.ai/mahjouri-saamahn/mwml-tutorial-app")


class PredictPayload(BaseModel):
    experiment_id: str = 'latest'
    inputs: list = [{"text": ""}]


@utils.construct_response
@app.post("/predict")
async def _predict(payload: PredictPayload):
    prediction = predict(inputs=payload.inputs, model=model, word_map=word_map)

    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {"prediction": prediction}
    }

    config.logger.info(json.dumps(response, indent=2))

    return response
