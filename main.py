import os
import gc
import re
import json
import time
import argparse
import unicodedata
from pathlib import Path

# بهینه‌سازی‌های مدیریت حافظه CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image, ImageOps

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

# ============================================================
# مسیرها و تنظیمات سخت‌افزاری 
# ============================================================

MODEL_ID_OR_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

IMAGE_DIR = Path("./dataset/images")
LABEL_DIR = Path("./dataset/labels")

OUTPUT_DIR = Path("./output")
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed_images"
SPLIT_DIR = OUTPUT_DIR / "splits"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = OUTPUT_DIR / "reports"
ADAPTER_DIR = OUTPUT_DIR / "qwen_receipt_lora"
AUTO_SPLIT_DIR = OUTPUT_DIR / "auto_splitted_receipts"

# فعال‌سازی سرعت بهینه روی هسته‌های تانسور
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# رزولوشن بهینه 896 جهت آزادسازی ۲.۵ گیگابایت حافظه VRAM
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 896 * 28 * 28  
MAX_SIDE = 896              
MAX_NEW_TOKENS = 1024        
CONFIDENCE_THRESHOLD = 0.70

# ============================================================
# سیستم پرامپت تخصصی و مقید (بدون خروجی توضیحی اضافه)
# ============================================================

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

# ============================================================
# توابع پیش‌پردازش با فیلتر کنتراست موضعی (CLAHE)
# ============================================================

def auto_split_long_receipts(image_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_exts = [".jpg", ".jpeg", ".png", ".webp"]
    
    print("\n--- بررسی و برش خودکار رسیدهای طولی ---")
    processed_count = 0
    
    for img_path in Path(image_dir).iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue
            
        img = cv2.imread(str(img_path))
        if img is None: continue
            
        h, w = img.shape[:2]
        if h / w > 2.0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
            row_sums = np.sum(thresh, axis=1)
            blank_threshold = np.max(row_sums) * 0.05
            is_blank = row_sums < blank_threshold
            
            split_points = []
            gap_start = None
            for y, blank in enumerate(is_blank):
                if blank and gap_start is None:
                    gap_start = y
                elif not blank and gap_start is not None:
                    gap_length = y - gap_start
                    if gap_length > 40:
                        split_points.append(gap_start + (gap_length // 2))
                    gap_start = None
            
            valid_splits = [0]
            for pt in split_points:
                if pt - valid_splits[-1] > 300:
                    valid_splits.append(pt)
            valid_splits.append(h)
            
            if len(valid_splits) > 2:
                for i in range(len(valid_splits) - 1):
                    y1, y2 = valid_splits[i], valid_splits[i+1]
                    cropped = img[y1:y2, 0:w]
                    out_name = output_dir / f"{img_path.stem}_part{i+1}{img_path.suffix}"
                    cv2.imwrite(str(out_name), cropped)
                processed_count += 1
                
    print(f"عملیات تمام شد. {processed_count} رسید تفکیک شدند.")

def make_dirs():
    for d in [OUTPUT_DIR, PREPROCESSED_DIR, SPLIT_DIR, PREDICTION_DIR, REPORT_DIR, ADAPTER_DIR, AUTO_SPLIT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

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

def preprocess_image(input_path, output_path):
    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = crop_whitespace(img)
        img = apply_clahe_enhancement(img)

        width, height = img.size
        if max(width, height) > MAX_SIDE:
            scale = MAX_SIDE / max(width, height)
            new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img.save(output_path, format="JPEG", quality=95)

def normalize_text(val):
    if val is None: return None
    val = unicodedata.normalize("NFKC", str(val))
    rep = {"ي": "ی", "ك": "ک", "ؤ": "و", "ۀ": "ه", "ة": "ه", "\u200c": " ", "\n": " ", "\t": " "}
    for k, v in rep.items():
        val = val.replace(k, v)
    for p, e in zip("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"):
        val = val.replace(p, e)
    val = re.sub(r"\s+", " ", val).strip()
    return val if val else None

def normalize_amount(val):
    if val is None: return None
    cleaned = normalize_text(val)
    if not cleaned: return None
    cleaned = re.sub(r"[^\d]", "", cleaned.replace(",", "").replace("٬", ""))
    return int(cleaned) if cleaned else None

def normalize_payload(data):
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if k in ["total_amount", "amount", "cash_breakdown_total", "row_number"]:
                new_dict[k] = normalize_amount(v)
            else:
                new_dict[k] = normalize_payload(v)
        return new_dict
    elif isinstance(data, list):
        return [normalize_payload(item) for item in data]
    else:
        return normalize_text(data)

def load_samples():
    samples = []
    valid_exts = [".jpg", ".jpeg", ".png", ".webp"]
    for label_file in sorted(LABEL_DIR.glob("*.json")):
        sample_id = label_file.stem.strip()
        img_path = None
        for ext in valid_exts:
            cand = IMAGE_DIR / f"{sample_id}{ext}"
            if cand.exists():
                img_path = cand
                break
        if not img_path: continue
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                raw_label = json.load(f)
        except Exception: continue

        out_img = PREPROCESSED_DIR / f"{sample_id}.jpg"
        if not out_img.exists():
            preprocess_image(img_path, out_img)

        samples.append({
            "id": sample_id,
            "image_path": str(out_img),
            "label": normalize_payload(raw_label)
        })
    return samples

class ReceiptDataset(TorchDataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class QwenVLDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        full_texts, images = [], []
        for sample in batch:
            answer = json.dumps(sample["label"], ensure_ascii=False, separators=(",", ":"))
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image_path"]},
                        {"type": "text", "text": USER_PROMPT_TEXT}
                    ]
                },
                {"role": "assistant", "content": answer}
            ]
            full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            img = Image.open(sample["image_path"]).convert("RGB")
            full_texts.append(full_text)
            images.append(img)

        batch_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        for img in images:
            img.close()

        labels = batch_inputs["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        assistant_tokens = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        seq_len = len(assistant_tokens)

        for i in range(len(batch)):
            ids = batch_inputs["input_ids"][i].tolist()
            found = -1
            for idx in range(len(ids) - seq_len + 1):
                if ids[idx:idx + seq_len] == assistant_tokens:
                    found = idx + seq_len
                    break
            if found != -1:
                labels[i, :found] = -100

        image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_pad_id is not None and image_pad_id != self.processor.tokenizer.unk_token_id:
            labels[labels == image_pad_id] = -100

        batch_inputs["labels"] = labels
        return batch_inputs

# ============================================================
# بارگذاری مدل و پیکربندی LoRA
# ============================================================

def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_model_and_processor(for_training=True):
    clean_gpu_memory()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID_OR_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID_OR_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=COMPUTE_DTYPE,
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    
    model.config.use_cache = False if for_training else True

    if for_training:
        if hasattr(model, "visual"):
            model.visual.requires_grad_(False)

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.10,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor

# ============================================================
# آموزش مدل 
# ============================================================

def train_model(train_samples):
    train_dataset = ReceiptDataset(train_samples)
    model, processor = load_model_and_processor(for_training=True)

    training_args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=7,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        weight_decay=0.02,
        max_grad_norm=0.5,
        lr_scheduler_type="cosine",
        warmup_steps=9,
        logging_steps=2,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=(COMPUTE_DTYPE == torch.float16),
        bf16=(COMPUTE_DTYPE == torch.bfloat16),
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QwenVLDataCollator(processor)
    )

    print(f"\n--- آغاز آموزش مدل با {len(train_samples)} نمونه و تنظیمات پایدار ۱۶ گیگابایت ---")
    start_time = time.perf_counter()
    trainer.train()
    print(f"آموزش در {(time.perf_counter() - start_time) / 60:.2f} دقیقه به پایان رسید.")

    trainer.model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)

    del trainer, model
    clean_gpu_memory()

# ============================================================
# پساپردازش و استنتاج
# ============================================================

def validate_and_fix_extracted_data(parsed_json):
    if not parsed_json or not isinstance(parsed_json, dict): return parsed_json
    records = parsed_json.get("records", [])
    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict): continue
            
            if rec.get("time") and isinstance(rec["time"], str):
                parts = rec["time"].split(":")
                if len(parts) >= 2:
                    rec["time"] = ":".join([p.zfill(2) for p in parts])
                    
            if "total_amount" in rec:
                rec["total_amount"] = normalize_amount(rec["total_amount"])
                
            if rec.get("sub_items") and isinstance(rec["sub_items"], list):
                for item in rec["sub_items"]:
                    if isinstance(item, dict) and "amount" in item:
                        item["amount"] = normalize_amount(item["amount"])
    return parsed_json

def repair_json_string(json_str):
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
    json_str = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', json_str)
    return json_str

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
    except json.JSONDecodeError:
        try:
            return json.loads(repair_json_string(raw_sub))
        except Exception:
            return None

def extract_word_confidences(processor, generated_ids, token_probs):
    syntax_chars = {'{', '}', '[', ']', ':', ',', '"', 'null', ''}
    words_data = []
    curr_tokens, curr_probs = [], []

    for t_id, prob in zip(generated_ids, token_probs):
        tok_str = processor.decode([t_id], skip_special_tokens=True)
        if not tok_str: continue
        
        if re.search(r'\s', tok_str):
            parts = re.split(r'(\s+)', tok_str)
            for part in parts:
                if not part: continue
                if re.match(r'^\s+$', part):
                    if curr_tokens:
                        full_word = "".join(curr_tokens).strip()
                        clean_word = re.sub(r'["\':,]', '', full_word).strip()
                        if clean_word and clean_word not in syntax_chars:
                            mean_p = float(np.mean(curr_probs))
                            words_data.append({
                                "word": clean_word,
                                "confidence_percent": round(mean_p * 100, 2)
                            })
                        curr_tokens, curr_probs = [], []
                else:
                    curr_tokens.append(part)
                    curr_probs.append(prob)
        else:
            curr_tokens.append(tok_str)
            curr_probs.append(prob)

    if curr_tokens:
        full_word = "".join(curr_tokens).strip()
        clean_word = re.sub(r'["\':,]', '', full_word).strip()
        if clean_word and clean_word not in syntax_chars:
            mean_p = float(np.mean(curr_probs))
            words_data.append({
                "word": clean_word,
                "confidence_percent": round(mean_p * 100, 2)
            })

    return words_data

def predict_with_confidence(model, processor, image_path):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": USER_PROMPT_TEXT}
        ]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[prompt],
        images=[img],
        padding=True,
        return_tensors="pt"
    ).to("cuda:0")
    img.close()

    input_len = inputs.input_ids.shape[1]

    eos_ids = [
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]
    eos_ids = [t for t in eos_ids if t is not None]

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

    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True
    )

    token_probs = torch.exp(transition_scores[0]).cpu().tolist()
    mean_confidence = round(float(np.mean(token_probs)) * 100, 2) if token_probs else 0.0

    word_confidences = extract_word_confidences(processor, generated_ids, token_probs)
    low_confidence_words = [
        w for w in word_confidences if (w["confidence_percent"] / 100.0) < CONFIDENCE_THRESHOLD
    ]

    parsed_json = extract_json_from_text(output_text)
    parsed_json = validate_and_fix_extracted_data(parsed_json)

    del inputs, outputs
    return output_text, parsed_json, mean_confidence, word_confidences, low_confidence_words

def evaluate(val_samples):
    clean_gpu_memory()
    base_model, processor = load_model_and_processor(for_training=False)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, device_map={"": 0}).eval()

    print("\n" + "=" * 70)
    print("--- آغاز ارزیابی مدل و سنجش قطعیت کلمات استخراج‌شده ---")
    print("=" * 70)

    summary_warnings = []

    for idx, s in enumerate(val_samples, 1):
        t0 = time.perf_counter()
        raw_out, parsed, mean_conf, words_conf, low_conf_words = predict_with_confidence(
            model, processor, s["image_path"]
        )
        elapsed = time.perf_counter() - t0

        print(f"\n[{idx}/{len(val_samples)}] تصویر: {s['id']} | زمان: {elapsed:.2f}s | میانگین قطعیت: {mean_conf}%")

        if low_conf_words:
            print(f"  [⚠️ هشدار: {len(low_conf_words)} کلمه مشکوک یا کم‌احتمال]")
            summary_warnings.append({
                "id": s["id"],
                "mean_confidence": mean_conf,
                "low_confidence_count": len(low_conf_words),
                "low_confidence_words": low_conf_words
            })

        out_record = {
            "id": s["id"],
            "image_path": s["image_path"],
            "mean_confidence_percent": mean_conf,
            "has_low_confidence_warning": len(low_conf_words) > 0,
            "word_confidences": words_conf,
            "low_confidence_words": low_conf_words,
            "ground_truth": s["label"],
            "prediction": parsed,
            "raw_output": raw_out,
            "latency_seconds": round(elapsed, 3)
        }

        with open(PREDICTION_DIR / f"{s['id']}.json", "w", encoding="utf-8") as f:
            json.dump(out_record, f, ensure_ascii=False, indent=2)

    with open(REPORT_DIR / "confidence_warnings_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_evaluated": len(val_samples),
            "samples_with_warnings": len(summary_warnings),
            "threshold_used": f"{CONFIDENCE_THRESHOLD * 100}%",
            "warning_list": summary_warnings
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"ارزیابی پایان یافت. گزارش در {REPORT_DIR / 'confidence_warnings_summary.json'} ثبت گردید.")
    print("=" * 70 + "\n")

    del model, base_model
    clean_gpu_memory()

# ============================================================
# تابع اصلی (Main)
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "all", "split"], default="all")
    args, _ = parser.parse_known_args()

    make_dirs()

    if args.mode == "split":
        auto_split_long_receipts(IMAGE_DIR, AUTO_SPLIT_DIR)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("پردازنده گرافیکی فعال (CUDA) شناسایی نشد.")

    samples = load_samples()
    if len(samples) < 2:
        raise RuntimeError(f"تعداد داده‌های معتبر در دایرکتوری برچسب‌ها کافی نیست ({len(samples)} نمونه یافت شد).")

    if args.mode in ["train", "all"]:
        train_model(samples)

    if args.mode in ["evaluate", "all"]:
        evaluate(samples)

if __name__ == "__main__":
    main()