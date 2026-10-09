# 1. استفاده از موتور پایتورچ 2.5.1 (پایدارترین نسخه برای مدل‌های نسل 2025)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app

# 2. نصب پیش‌نیازهای سیستمی برای OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 3. حذف پکیج‌های پایه از فایل نیازمندی‌ها تا دانلود مجدد و تداخل C++ رخ ندهد
RUN sed -i '/torch/d' requirements.txt
RUN sed -i '/torchvision/d' requirements.txt
RUN sed -i '/torchaudio/d' requirements.txt

# 4. نصب دقیق torchvision نسخه 0.20.1 (دقیقاً هماهنگ و جفت‌شده با PyTorch 2.5.1)
RUN pip install --no-cache-dir --no-deps torchvision==0.20.1 torchaudio==2.5.1

# 5. نصب اکوسیستم هوش مصنوعی و رابط کاربری از مخزن رسمی بدون ارور 402 پولی
RUN pip install --no-cache-dir -i https://pypi.org/simple/ -r requirements.txt
RUN pip install --no-cache-dir -i https://pypi.org/simple/ \
    "transformers>=4.49.0" \
    "numpy<2" \
    huggingface_hub \
    qwen-vl-utils \
    bitsandbytes \
    accelerate \
    peft \
    gradio \
    opencv-python-headless

# 6. دانلود پایدار مدل هوش مصنوعی با استفاده از Mirror ضدتحریم
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-VL-3B-Instruct', resume_download=True)"

COPY . .
CMD ["python", "main.py", "--mode", "all"]