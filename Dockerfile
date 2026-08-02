# استخدام نسخة بايثون خفيفة
FROM python:3.9

# تحديد مكان العمل جوه السيرفر
WORKDIR /code

# نسخ ملف المكتبات وتسطيبها
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# نسخ باقي ملفات المشروع (الموديل، الكود، الـ HTML)
COPY . .

# تشغيل الـ FastAPI باستخدام uvicorn
# CMD ["uvicorn", "app.py:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]