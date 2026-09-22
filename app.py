import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import torch
import laya
try:
    import laya_npu
except ImportError:
    laya_npu = None
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get('LAYA_CHECKPOINT', 'multilingual')
DEVICE = os.environ.get('LAYA_DEVICE', 'npu')
agent = None
lock = asyncio.Lock()

class Question(BaseModel):
    type: Literal['choice', 'score', 'noul']
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode='after')
    def validate_options(self):
        if self.type == 'noul' and self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError('noul criteria must be a dictionary')
        if self.type != 'noul':
            if self.criteria is None or not 2 <= len(self.criteria) <= 12:
                raise ValueError('choice/score require 2–12 criteria')
            if self.type == 'score' and not isinstance(self.criteria, list):
                raise ValueError('score criteria must be a list')
            if any(not str(k).strip() for k in self.criteria):
                raise ValueError('criteria must not be empty')
            if len(json.dumps(self.criteria, ensure_ascii=False)) > 4000:
                raise ValueError('criteria too long')
        return self

class PredictRequest(BaseModel):
    state: str | dict | list
    questions: dict[str, Question] = Field(min_length=1, max_length=4)

    @model_validator(mode='after')
    def validate_size(self):
        if len(json.dumps(self.state, ensure_ascii=False)) > 16000:
            raise ValueError('state exceeds 16000 characters')
        return self

@asynccontextmanager
async def lifespan(app):
    global agent
    torch.set_num_threads(int(os.environ.get('LAYA_THREADS', '4')))
    torch.set_num_interop_threads(1)
    path = ROOT / 'models'
    if CHECKPOINT == 'multilingual':
        path /= 'multilingual'
    elif CHECKPOINT != 'english':
        raise ValueError('LAYA_CHECKPOINT must be multilingual or english')
    if DEVICE == 'npu' and laya_npu is not None:
        agent = await asyncio.to_thread(laya_npu.load, str(path), device='npu')
    else:
        agent = await asyncio.to_thread(laya.load, str(path), device=DEVICE)
    yield

app = FastAPI(title='Laya on Rubik Pi 3', lifespan=lifespan)

@app.get('/health')
def health():
    return {'status': 'ready', 'checkpoint': CHECKPOINT, 'device': DEVICE, 'threads': torch.get_num_threads(), 'revision': (ROOT / 'models/revision.txt').read_text().strip()}

@app.post('/predict')
async def predict(body: PredictRequest):
    if lock.locked():
        raise HTTPException(503, 'Model busy; retry after current request', headers={'Retry-After': '2'})
    async with lock:
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(agent.predict, body.state, {k: v.model_dump(exclude_none=True) for k, v in body.questions.items()})
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        result['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 1)
        result['checkpoint'] = CHECKPOINT
        return result

@app.get('/', response_class=HTMLResponse)
def home():
    return (ROOT / 'index.html').read_text()
