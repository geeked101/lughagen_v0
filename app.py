import os
import requests
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "plutoverse/LughaGen-v0-nllb"

app = FastAPI(
    title="LughaGen API",
    description="LughaGen multilingual translation API",
    version="0.1.0"
)

# Will be filled when loaded on demand
tokenizer = None
model = None

# Loading state and lock for thread-safety
_load_lock = threading.Lock()
_loading = False

def _download_and_init_model():
    global tokenizer, model, _loading
    hf_token = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )

    try:
        _loading = True
        print("HUGGINGFACE token present:", bool(hf_token))
        if hf_token:
            try:
                headers = {"Authorization": f"Bearer {hf_token}"}
                r = requests.head(f"https://huggingface.co/{MODEL_ID}/resolve/main/config.json", headers=headers, timeout=10)
                print("HF repo HEAD status:", r.status_code)
            except Exception as e:
                print("HF HEAD check failed:", repr(e))

        print("Loading LughaGen tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=hf_token
        )

        print("Tokenizer loaded!")

        # Temporarily disabled model loading to reduce startup memory/time while testing.
        # model = AutoModelForSeq2SeqLM.from_pretrained(
        #     MODEL_ID,
        #     use_auth_token=hf_token,
        #     low_cpu_mem_usage=True
        # )

        # print("LughaGen loaded successfully!")
    except Exception as e:
        print("LughaGen failed to load:", repr(e))
        tokenizer = None
        model = None
    finally:
        _loading = False

def ensure_loading_in_background():
    """Start background loader if not loaded and not already loading."""
    global _loading
    if tokenizer is not None and model is not None:
        return
    with _load_lock:
        if tokenizer is not None and model is not None:
            return
        if not _loading:
            t = threading.Thread(target=_download_and_init_model, daemon=True)
            t.start()

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

@app.on_event("startup")
def startup_event():
    # Start loading in background so the server can bind immediately
    ensure_loading_in_background()

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
        "model_loaded": tokenizer is not None and model is not None,
        "loading": _loading
    }

@app.post("/translate")
def translate(request: TranslationRequest):
    # If model not ready, start background load and return 503 while it loads
    if tokenizer is None or model is None:
        ensure_loading_in_background()
        raise HTTPException(status_code=503, detail="Model is loading in background; try again in a minute.")

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
