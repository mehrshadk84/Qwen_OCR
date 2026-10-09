# Qwen2.5-VL Iranian Financial Document Intelligence & OCR Engine

An end-to-end deep learning pipeline for fine-tuning, merging, and deploying **Qwen2.5-VL-3B-Instruct** for Iranian financial documents, Point-of-Sale (POS) thermal receipts, multi-account treasury slips, postal manifests, check orders, and banking statements.

The system features hardware-optimized QLoRA fine-tuning with thermal-print enhancement, zero-latency 16-bit standalone weight merging, token transition confidence scoring, and an asynchronous, non-blocking FastAPI serving engine.

## Table of Contents

* [Architecture & Key Highlights](#architecture--key-highlights)

* [System Architecture Flow](#system-architecture-flow)

* [Project Directory Structure](#project-directory-structure)

* [Hardware & Software Requirements](#hardware--software-requirements)

* [Dataset Specifications & Label Schema](#dataset-specifications--label-schema)

* [Step-by-Step Production Guide](#step-by-step-production-guide)

  * [1. Environment & Docker Build](#1-environment--docker-build)

  * [2. Model Fine-Tuning (QLoRA)](#2-model-fine-tuning-qlora)

  * [3. 16-Bit Standalone Weight Merging](#3-16-bit-standalone-weight-merging)

  * [4. Production Daemon API Deployment](#4-production-daemon-api-deployment)

* [API Reference & Benchmarks](#api-reference--benchmarks)

* [Troubleshooting & Performance Tuning](#troubleshooting--performance-tuning)

## Architecture & Key Highlights

### 1. Vision-Language Fine-Tuning (QLoRA)

* **Base Architecture:** `Qwen/Qwen2.5-VL-3B-Instruct` loaded in 4-bit NormalFloat (`NF4`) with double quantization.

* **LoRA Adaptation:** Target modules include all attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP gates (`gate_proj`, `up_proj`, `down_proj`).

* **Selective Loss Masking:** Masks prompt tokens and special image-pad tokens (`<|image_pad|>`) with label ID `-100`, computing backpropagation loss strictly on assistant response tokens.

* **Thermal Print Contrast Booster:** Integrates an OpenCV **Contrast Limited Adaptive Histogram Equalization (CLAHE)** filter and Laplacian edge sharpening to render faded, dot-matrix thermal receipts razor-sharp prior to patch tokenization.

### 2. Zero-Overhead 16-Bit Merging

* Combines the quantized base model and LoRA adapter into full-precision `bfloat16`/`float16` weights (`merge_and_unload`).

* Eliminates adapter switching and dynamic dequantization latency, reducing serving latency by **\~40%**.

### 3. Production Serving Engine

* **Pre-Warmed Inference (Zero Cold-Start):** Automatically runs a dummy warm-up inference cycle upon FastAPI startup, pre-compiling CUDA kernels and cuDNN paths.

* **Threadpool Architecture:** Utilizes synchronous FastAPI handler offloading (`def predict`) to prevent blocking the async event loop during compute-intensive PyTorch execution.

* **Token Confidence Diagnostics:** Employs `compute_transition_scores` to calculate word-level statistical certainty and flags low-confidence OCR entities below threshold ($70\%$).

* **Self-Healing Output Formatter:** Persian/Arabic numeral NFKC normalization, whitespace stripping, comma removal from currency amounts, zero-padding for tracking IDs, and fallback regex-based JSON decoders.

## System Architecture Flow

```
                      +------------------------------------------+
                      |         Raw Financial Documents          |
                      |   (POS, Bank Slips, Postal Manifests)    |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |       Image Preprocessing Pipeline       |
                      |   - Whitespace Bounding Box Truncation   |
                      |   - CLAHE (Tile Grid: 8x8, Clip: 2.0)    |
                      |   - Kernel Sharpening & Aspect Scaling   |
                      +------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v (Training Mode)                             v (Inference Mode)
+---------------------------------------+     +---------------------------------------+
|          QLoRA Fine-Tuning            |     |        16-Bit Merged Inference        |
| - 4-bit NF4 Quantization              |     | - Direct VRAM GPU Resident            |
| - Gradient Checkpointing              |     | - Pre-Compiled CUDA Warmup            |
| - Loss Masking (Assistant Only)       |     | - Token Transition Scoring            |
+---------------------------------------+     +---------------------------------------+
                    |                                             |
                    v                                             v
+---------------------------------------+     +---------------------------------------+
|       LoRA Adapter Checkpoints        |     |     Self-Healing Response Parsing     |
|   (output/qwen_receipt_lora)          |     | - Regex JSON Repair & Validation      |
+---------------------------------------+     | - Persian Normalization (NFKC)        |
                    |                         | - Low-Confidence Diagnostics (<0.70)  |
                    v                                             |
+---------------------------------------+                         v
|    merge.py (Merge & Unload)          |     +---------------------------------------+
|  => output/qwen_receipt_merged (FP16) |     |       Strict JSON Response API        |
+---------------------------------------+     +---------------------------------------+

```

## Project Directory Structure

```
qwen-ocr/
|-- dataset/
|   |-- images/                   # Training/Evaluation images (.jpg, .png, .webp)
|   `-- labels/                   # Ground-truth JSON annotations
|-- output/
|   |-- preprocessed_images/      # Cached CLAHE-enhanced resized images
|   |-- qwen_receipt_lora/        # Trained LoRA adapter weights & tokenizer
|   |-- qwen_receipt_merged/      # Standalone 16-bit production model
|   |-- predictions/              # Validation evaluation dumps
|   `-- reports/                  # Confidence & discrepancy analytics
|-- api_server.py                 # FastAPI microservice for high-throughput inference
|-- main.py                       # Training, preprocessing & evaluation orchestrator
|-- merge.py                      # Weight fusion & conversion script
|-- Dockerfile                    # Containerization manifest
|-- requirements.txt              # Python runtime dependencies
|-- .gitignore                    # Prevents tracking large binary artifacts
`-- README.md                     # Technical documentation

```

## Hardware & Software Requirements

| Component | Minimum Specification | Recommended Specification | 
| ----- | ----- | ----- | 
| **GPU** | NVIDIA GPU with $\ge 16\text{ GB}$ VRAM | NVIDIA RTX 4090, A5000, A100, or H100 | 
| **CUDA Driver** | Version $\ge 535.xx$ | Version $\ge 550.xx$ | 
| **System Memory** | 32 GB RAM | 64 GB RAM | 
| **Storage** | 30 GB SSD Space | 50 GB NVMe SSD | 
| **Environment** | Docker Engine with `nvidia-container-toolkit` | Ubuntu 22.04 LTS / 24.04 LTS | 

## Dataset Specifications & Label Schema

Every image file in `dataset/images/` must have a corresponding `.json` file in `dataset/labels/` with identical base filenames (e.g., `receipt_0102.jpg` $\leftrightarrow$ `receipt_0102.json`).

### Target Schema Definition

```
{
  "records": [
    {
      "document_type": "رسید پوز چندحسابی",
      "bank_or_issuer": "به پرداخت ملت / بانک سامان",
      "date": "1405/03/21",
      "time": "13:14:00",
      "total_amount": 5720000,
      "tracking_code": "802435238388-221012710",
      "payer": "621986***9215",
      "payee": "ناحیه پستی یوسفیان",
      "sub_items": [
        {
          "title": "مبلغ ۱ (شناسه حساب: 100003)",
          "amount": 5200000
        },
        {
          "title": "مبلغ ۲ (شناسه حساب: 100002)",
          "amount": 520000
        }
      ],
      "handwritten_note": null
    }
  ]
}

```

### Supported Document Types

* `"رسید پوز"`: Single-transaction thermal POS payment slip.

* `"رسید پوز چندحسابی"`: Treasury/postal multi-account split slip.

* `"فیش واریز نقدی"`: Bank cash deposit voucher.

* `"صورتحساب سپرده"`: Bank statement or ledger excerpt.

* `"صورت شمارش صندوق"`: End-of-day cash drawer register sheet.

* `"گزارش مرسولات پستی"`: Postal logistics distribution manifest.

* `"دستور وصول چک"`: Bank check clearing order.

* `"فاکتور فروش"`: Commercial itemized sales invoice.

* `"رسید انتقال وجه خودپرداز"`: ATM Card-to-Card receipt.

## Step-by-Step Production Guide

### 1. Environment & Docker Build

Build the optimized container image:

```
docker build -t qwen-ocr-app .

```

Ensure NVIDIA Container Runtime is functioning:

```
docker run --rm --gpus all qwen-ocr-app nvidia-smi

```

### 2. Model Fine-Tuning (QLoRA)

Clean legacy cache directories and launch training:

```
sudo rm -rf output/preprocessed_images/ output/qwen_receipt_lora/ output/predictions/ output/reports/

sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/main.py:/app/main.py \
  -it qwen-ocr-app python main.py --mode train

```

#### Training Configuration Table:

| Parameter | Value | Rationale | 
| ----- | ----- | ----- | 
| `MAX_SIDE` | `896 px` | Fits batch context into 16 GB VRAM with zero OOM spikes | 
| `num_train_epochs` | `7` | Optimal loss convergence without over-fitting | 
| `per_device_train_batch_size` | `1` | Minimizes memory allocation overhead | 
| `gradient_accumulation_steps` | `8` | Effective batch size of 8 | 
| `learning_rate` | `3e-5` | Cosine decay schedule with 9 warmup steps | 
| `optim` | `paged_adamw_8bit` | Offloads optimizer states to dynamic memory segments | 

### 3. 16-Bit Standalone Weight Merging

Once training finishes, fuse the adapter into base model weights:

```
sudo docker run --rm --ipc=host \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/merge.py:/app/merge.py \
  -it qwen-ocr-app python merge.py

```

*The merged model will be saved at `./output/qwen_receipt_merged`.*

### 4. Production Daemon API Deployment

Run the FastAPI engine as a persistent, self-healing background container:

```
sudo docker run -d \
  --name qwen-api-server \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/api_server.py:/app/api_server.py \
  qwen-ocr-app python api_server.py

```

#### Tail Live Logs:

```
sudo docker logs -f qwen-api-server

```

When you see `Uvicorn running on http://0.0.0.0:8000`, the server is pre-warmed and ready.

## API Reference & Benchmarks

### Health Check

```
GET / HTTP/1.1
Host: localhost:8000

```

#### Response:

```
{
  "status": "running",
  "model": "./output/qwen_receipt_merged"
}

```

### Extract Receipt Data

```
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

```

#### `curl` Example:

```
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/receipt.jpg"

```

#### Response Body:

```
{
  "status": "success",
  "latency_seconds": 0.842,
  "mean_confidence_percent": 96.34,
  "has_low_confidence_warning": false,
  "low_confidence_words": [],
  "data": {
    "records": [
      {
        "document_type": "رسید پوز چندحسابی",
        "bank_or_issuer": "به پرداخت ملت / بانک سامان",
        "date": "1405/03/21",
        "time": "13:14:00",
        "total_amount": 5720000,
        "tracking_code": "802435238388-221012710",
        "payer": "621986***9215",
        "payee": "ناحیه پستی یوسفیان",
        "sub_items": [
          {
            "title": "مبلغ ۱ (شناسه حساب: 100003)",
            "amount": 5200000
          },
          {
            "title": "مبلغ ۲ (شناسه حساب: 100002)",
            "amount": 520000
          }
        ],
        "handwritten_note": null
      }
    ]
  },
  "raw_text": "{\"records\":[...]}"
}

```

## Troubleshooting & Performance Tuning

### CUDA Out of Memory (`CUDA OOM`) During Training

1. **Clear Preprocessing Cache:**
   Ensure old scaled images are cleared:

   ```
   rm -rf output/preprocessed_images/
   
   ```

2. **Cap Image Dimensions:**
   In `main.py`, maintain `MAX_SIDE = 896` or reduce to `840`.

3. **Disable Dataloader Pin Memory:**
   Ensure `dataloader_pin_memory=False` and `dataloader_num_workers=0` inside `TrainingArguments`.

### Multi-Row or Extremely Long Receipt Degradation

If your receipts exceed an aspect ratio of $H/W > 2.5$:

* Use `python main.py --mode split` to automatically partition long tapes along horizontal whitespace regions without truncating text lines.

### Response Latency Optimization

* **TF32 Mode:** Both `torch.backends.cuda.matmul.allow_tf32` and `cudnn.allow_tf32` are enabled by default for Ampere, Ada Lovelace, and Hopper architectures.

* **Pre-warmed GPU:** The lifespan context manager ensures the model stays permanently cached in VRAM, eliminating execution jitters.

## License

Proprietary & Confidential. Unauthorized distribution, modification, or reverse engineering of trained weights and source code is strictly prohibited.