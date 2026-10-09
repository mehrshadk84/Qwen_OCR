import os
import re
import gc
import json
import time
import unicodedata
from io import BytesIO
from contextlib import asynccontextmanager

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_PATH = "./output/qwen_receipt_merged"
MAX_SIDE = 1024
MAX_NEW_TOKENS = 1024
CONFIDENCE_THRESHOLD = 0.70

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

SYSTEM_PROMPT = """You are an expert Document Intelligence and OCR model specialized in Iranian financial documents, including POS receipts, multi-account payment slips, bank deposit slips (فیش نقدی), bank account statements (صورتحساب سپرده), cash register sheets (صورت شمارش صندوق), check deposits (دستور وصول چک), postal shipping manifests, and sales invoices.

YOUR TASK:
Extract all financial data and metadata into a strict, single JSON object containing a "records" array. Every independent transaction or document slice must be a distinct object in "records".

FIELD EXTRACTION & TERMINOLOGY MAPPING:
1. "document_type": Classify as one of:
   - "رسید پوز" (Standard POS receipt)
   - "رسید پوز چندحسابی" (Multi-account postal/treasury POS slip with multiple amounts)
   - "فیش واریز نقدی" (Handwritten/printed bank deposit slips)
   - "صورتحساب سپرده" (Multi-row bank statement; append "(ردیف X)" if row number is present)
   - "صورت شمارش صندوق" (Cash register count sheets)
   - "گزارش مرسولات پستی" (Postal manifest sheets)
   - "دستور وصول چک" (Bank check deposit orders)
   - "فاکتور فروش" (Sales invoices)
   - "رسید انتقال وجه خودپرداز" (ATM Card-to-Card slips)

2. "bank_or_issuer": Bank, PSP, or entity issuing the document.
   - Look for: بانک سامان, به پرداخت ملت, آسان پرداخت (آپ), پرداخت الکترونیک سامان, بانک ملی, بانک تجارت, بانک پارسیان, بانک آینده, انتشارات جیحون, شرکت درصد, شرکت ملی پست.

3. "date" & "time":
   - "date": Solar Hijri date formatted strictly as "YYYY/MM/DD". Convert 2-digit years (e.g., 98/01/10 -> 1398/01/10).
   - "time": Machine time formatted as "HH:MM:SS" or "HH:MM". If absent, use null.

4. "total_amount":
   - Numerical integer ONLY (no Rial, commas, or decimals).
   - Synonyms: مبلغ, مبلغ کل, جمع کل, مبلغ قابل پرداخت, بستانکار, بدهکار.

5. "tracking_code":
   - Store as EXACT STRING. Never drop leading zeros.
   - Comprehensive Persian synonyms:
     * شماره پیگیری / کد پیگیری / پیگیری
     * شماره مرجع / کد مرجع / مرجع / ارجاع
     * شماره ارجاع / ش.م / ش.پ / پ.ک
     * شماره فرم / شماره برگه / شماره سند
     * شماره تراکنش / کد رهگیری مرسوله پستی

6. "payer":
   - The originating party, sender, or source card/account.
   - Look for: واریزکننده, پرداخت‌کننده, خریدار, فرستنده, از حساب, از کارت (e.g., "621986***9215").

7. "payee":
   - The destination entity, recipient, or terminal.
   - Look for: به حساب, به نام, پذیرنده, فروشگاه, مقصد, گیرنده, شماره حساب یا شبای مقصد.

8. "sub_items":
   - An array of objects `{"title": string, "amount": integer}` if the document contains itemized breakdowns:
     * Multi-account POS slips: "مبلغ ۱ (حق‌السهم) - شناسه واریز: ...", "مبلغ ۲ (مالیات) - شناسه واریز: ..."
     * Cash register count: Banknote counts, e.g., "اسکناس ۱۰۰۰۰۰ ریالی - ۴ عدد"
     * Postal manifests: Individual recipient parcels and postal fees
     * Invoices: Individual item prices
   - If no sub-items exist, set strictly to null.

9. "handwritten_note":
   - Capture ONLY physical pen/pencil handwritten notes, manual corrections, or signatures.
   - If there is NO visible pen/pencil handwriting on the document, set strictly to null.
   - Never place printed text, machine data, or summary explanations in this field.

OUTPUT JSON SCHEMA:
{
  "records": [
    {
      "document_type": string,
      "bank_or_issuer": string or null,
      "date": "YYYY/MM/DD" or null,
      "time": string or null,
      "total_amount": integer or null,
      "tracking_code": string or null,
      "payer": string or null,
      "payee": string or null,
      "sub_items": [
        {
          "title": string,
          "amount": integer
        }
      ] or null,
      "handwritten_note": string or null
    }
  ]
}

STRICT OUTPUT CONSTRAINTS:
- Output MUST be valid, raw JSON starting with "{" and ending with "}".
- Do NOT output markdown code blocks (no ```json or ```).
- Absolutely NO conversational sentences, greetings, notes, explanations, or disclaimers outside the JSON object.
- Never hallucinate data that is not explicitly printed or written on the document.""".strip()

USER_PROMPT_TEXT = "اطلاعات مالی و تراکنش‌های این سند را با دقت کامل و طبق ساختار مشخص‌شده استخراج کن."

model = None
processor = None
eos_ids = None

def crop_whitespace(image, padding=15):
    img_gray = np.array(image.convert("L"))
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 80 and h > 80:
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(image.width, x + w + padding)
            y_max = min(image.height, y + h + padding)
            return image.crop((x_min, y_min, x_max, y_max))
    return image

def apply_clahe_enhancement(pil_img):
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))

def preprocess_image_buffer(image_bytes):
    with Image.open(BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = crop_whitespace(img)
        img = apply_clahe_enhancement(img)
        w, h = img.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img.copy()

def normalize_text(val):
    if val is None: return None
    val = unicodedata.normalize("NFKC", str(val))
    rep = {"ي": "ی", "ك": "ک", "ؤ": "و", "ۀ": "ه", "ة": "ه", "\u200c": " ", "\n": " ", "\t": " "}
    for k, v in rep.items(): val = val.replace(k, v)
    for p, e in zip("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"): val = val.replace(p, e)
    return re.sub(r"\s+", " ", val).strip() or None

def normalize_amount(val):
    if val is None: return None
    cleaned = normalize_text(val)
    if not cleaned: return None
    cleaned = re.sub(r"[^\d]", "", cleaned.replace(",", "").replace("٬", ""))
    return int(cleaned) if cleaned else None

def validate_and_fix_extracted_data(parsed_json):
    if not parsed_json or not isinstance(parsed_json, dict): return parsed_json
    records = parsed_json.get("records", [])
    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict): continue
            if rec.get("time") and isinstance(rec["time"], str):
                parts = rec["time"].split(":")
                if len(parts) >= 2: rec["time"] = ":".join([p.zfill(2) for p in parts])
            if "total_amount" in rec: rec["total_amount"] = normalize_amount(rec["total_amount"])
            if rec.get("sub_items") and isinstance(rec["sub_items"], list):
                for item in rec["sub_items"]:
                    if isinstance(item, dict) and "amount" in item:
                        item["amount"] = normalize_amount(item["amount"])
    return parsed_json

def extract_json_from_text(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s: return None
    raw_sub = text[s:e + 1]
    try:
        return json.loads(raw_sub)
    except Exception:
        clean_str = re.sub(r",\s*([\]}])", r"\1", raw_sub)
        try:
            return json.loads(clean_str)
        except Exception:
            return None

def extract_word_confidences(processor_obj, generated_ids, token_probs):
    syntax_chars = {'{', '}', '[', ']', ':', ',', '"', 'null', ''}
    words_data = []
    curr_tokens, curr_probs = [], []
    for t_id, prob in zip(generated_ids, token_probs):
        tok_str = processor_obj.decode([t_id], skip_special_tokens=True)
        if not tok_str: continue
        if re.search(r'\s', tok_str):
            parts = re.split(r'(\s+)', tok_str)
            for part in parts:
                if not part: continue
                if re.match(r'^\s+$', part):
                    if curr_tokens:
                        clean_word = re.sub(r'["\':,]', '', "".join(curr_tokens).strip()).strip()
                        if clean_word and clean_word not in syntax_chars:
                            words_data.append({"word": clean_word, "confidence_percent": round(float(np.mean(curr_probs)) * 100, 2)})
                        curr_tokens, curr_probs = [], []
                else:
                    curr_tokens.append(part)
                    curr_probs.append(prob)
        else:
            curr_tokens.append(tok_str)
            curr_probs.append(prob)
    if curr_tokens:
        clean_word = re.sub(r'["\':,]', '', "".join(curr_tokens).strip()).strip()
        if clean_word and clean_word not in syntax_chars:
            words_data.append({"word": clean_word, "confidence_percent": round(float(np.mean(curr_probs)) * 100, 2)})
    return words_data

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, eos_ids
    print("\n--- [1/2] در حال بارگذاری مدل ادغام‌شده ۱۶ بیتی روی GPU ---")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map={"": 0},
        torch_dtype=COMPUTE_DTYPE,
        attn_implementation="sdpa",
        trust_remote_code=True
    ).eval()

    eos_ids = [
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]
    eos_ids = [t for t in eos_ids if t is not None]

    print("--- [2/2] اجرای Warm-up اولیه روی GPU ---")
    dummy_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    dummy_messages = [
        {"role": "system", "content": "You are a warm-up agent."},
        {"role": "user", "content": [{"type": "image", "image": dummy_img}, {"type": "text", "text": "ping"}]}
    ]
    dummy_prompt = processor.apply_chat_template(dummy_messages, tokenize=False, add_generation_prompt=True)
    dummy_inputs = processor(text=[dummy_prompt], images=[dummy_img], padding=True, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        _ = model.generate(
            **dummy_inputs,
            max_new_tokens=5,
            eos_token_id=eos_ids,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True
        )

    print("✅ مدل با سرعت بالا لود و کامپایل شد و روی پورت 8000 آماده است!\n")
    yield
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

app = FastAPI(title="Qwen Fast Financial OCR Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "running", "model": MODEL_PATH}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    
    # خواندن بایت‌های فایل بدون تکیه بر هدر Content-Type کلاینت
    try:
        image_bytes = file.file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="فایل ارسالی خالی است.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطا در خواندن فایل: {str(e)}")

    # اعتبارسنجی واقعی تصویر از طریق لود باینری
    try:
        img = preprocess_image_buffer(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"فایل ارسالی یک تصویر معتبر نیست: {str(e)}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": USER_PROMPT_TEXT}]}
    ]
    
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[img], padding=True, return_tensors="pt").to("cuda:0")
    input_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_ids,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    generated_ids = outputs.sequences[0, input_len:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    token_probs = torch.exp(transition_scores[0]).cpu().tolist()
    mean_confidence = round(float(np.mean(token_probs)) * 100, 2) if token_probs else 0.0

    word_confidences = extract_word_confidences(processor, generated_ids, token_probs)
    low_conf_words = [w for w in word_confidences if (w["confidence_percent"] / 100.0) < CONFIDENCE_THRESHOLD]

    parsed_json = extract_json_from_text(output_text)
    parsed_json = validate_and_fix_extracted_data(parsed_json)
    
    latency = round(time.perf_counter() - t0, 3)

    return {
        "status": "success",
        "latency_seconds": latency,
        "mean_confidence_percent": mean_confidence,
        "has_low_confidence_warning": len(low_conf_words) > 0,
        "low_confidence_words": low_conf_words,
        "data": parsed_json,
        "raw_text": output_text
    }

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, workers=1)