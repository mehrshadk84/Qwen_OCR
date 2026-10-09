import os
import gc
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "./output/qwen_receipt_lora"
MERGED_OUTPUT_PATH = "./output/qwen_receipt_merged"

print("\n" + "=" * 60)
print("--- ۱. بارگذاری مدل پایه (FP16/BF16) در RAM ---")
print("=" * 60)

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("\n--- ۲. بارگذاری وزن‌های آموزش‌دیده LoRA ---")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("\n--- ۳. ادغام قطعی وزن‌ها (Merge and Unload) ---")
merged_model = model.merge_and_unload()

print(f"\n--- ۴. ذخیره مدل ادغام‌شده در: {MERGED_OUTPUT_PATH} ---")
os.makedirs(MERGED_OUTPUT_PATH, exist_ok=True)
merged_model.save_pretrained(MERGED_OUTPUT_PATH, safe_serialization=True)

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
processor.save_pretrained(MERGED_OUTPUT_PATH)

print("\n" + "=" * 60)
print("✅ مدل ۱۶ بیتی با موفقیت ساخته و ذخیره شد!")
print("=" * 60 + "\n")