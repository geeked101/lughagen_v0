import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "plutoverse/LughaGen-v0-nllb"

app = FastAPI(
    title="LughaGen API",
    description="LughaGen multilingual translation API",
    version="0.1.0"
)

# Will be filled at startup (avoids importing/downloading during module import/build)
tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    # Read common env var names for convenience
    hf_token = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )

    print("HUGGINGFACE token present:", bool(hf_token))

    if hf_token:
        try:
            # quick HEAD to see if HF accepts the token for this repo
            headers = {"Authorization": f"Bearer {hf_token}"}
            r = requests.head(f"https://huggingface.co/{MODEL_ID}/resolve/main/config.json", headers=headers, timeout=10)
            print("HF repo HEAD status:", r.status_code)
        except Exception as e:
            print("HF HEAD check failed:", repr(e))

    try:
        print("Loading LughaGen tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=hf_token)

        print("Loading LughaGen model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_auth_token=hf_token)

        print("LughaGen loaded successfully!")
    except Exception as e:
        # Keep server running so you can inspect logs / health endpoint
        print("LughaGen failed to load at startup:", repr(e))
        tokenizer = None
        model = None


class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str


@app.get("/")
def root():
    return {
        "name": "LughaGen API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "model_loaded": tokenizer is not None and model is not None
    }


@app.post("/translate")
def translate(request: TranslationRequest):

    if tokenizer is None or model is None:
        # Not ready to serve translations
        raise HTTPException(status_code=503, detail="Model not loaded yet; check HUGGINGFACE_HUB_TOKEN and logs.")

    tokenizer.src_lang = request.source_language

    inputs = tokenizer(
        request.text,
        return_tensors="pt"
    )

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(
        request.target_language
    )

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=100
    )

    translation = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]

    return {
        "source": request.text,
        "translation": translation,
        "source_language": request.source_language,
        "target_language": request.target_language
    }
