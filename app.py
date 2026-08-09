from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "plutoverse/LughaGen-v0-nllb-bucket"

app = FastAPI(
    title="LughaGen API",
    description="LughaGen multilingual translation API",
    version="0.1.0"
)

print("Loading LughaGen tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading LughaGen model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

print("LughaGen loaded successfully!")


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
        "model": MODEL_ID
    }


@app.post("/translate")
def translate(request: TranslationRequest):

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