import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from deep_translator import GoogleTranslator
import uvicorn
from fastapi.responses import RedirectResponse
import re
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langdetect import detect, DetectorFactory

# ضمان نتائج ثابتة لفحص اللغة
DetectorFactory.seed = 0

# ============================
# 1. API Security & Rate Limiting
# ============================
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Medical Specialty Predictor API", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    error_msg = exc.errors()[0]["msg"]

    return JSONResponse(
        status_code=400,
        content={
            "predicted_specialty": None,
            "confidence": 0.0,
            "message": error_msg
        }
    )

def error_response(message: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={
            "predicted_specialty": None,
            "confidence": 0.0,
            "message": message
        }
    )


# ============================
# 2. Pydantic Models with Validations
# ============================
from pydantic import BaseModel, Field, validator
import re
from langdetect import detect

class MedicalInput(BaseModel):
    # هنا الـ Swagger هيعرض تلقائياً "string" كـ placeholder
    text: str = Field(..., min_length=10, max_length=500, example="string")

    @validator('text')
    def validate_content(cls, v):
        # 1. تنظيف أولي
        val = v.strip()
        # 2. منع اللينكات
        if re.search(r'http[s]?://\S+|www\.\S+', val):
            raise ValueError('غير مسموح بإرسال روابط.')
    
        # 3. منع تكرار أي حرف أكتر من 3 مرات (Spam) مثل "goooood" أو "asdfgh"
        if re.search(r'(.)\1{3,}', val):
            raise ValueError('يوجد تكرار حروف غير طبيعي.')
    
        
        digits_count = sum(c.isdigit() for c in val)
        if digits_count > len(val) * 0.3:
            raise ValueError('النص يحتوي على أرقام كثيرة، يرجى كتابة وصف الحالة فقط.')
        # 4. منع حروف إنجليزي عشوائية ورا بعضها (Gibberish Detection)
        # لو النص إنجليزي، بنفحص لو فيه كلمات طويلة جداً مفيهاش حروف متحركة (Vowels)
        # دي بتقفش كلمات زي "asdfghjkl" أو "qwertyuiop"
        if bool(re.search(r'[a-zA-Z]', val)):
            words = val.split()
            for word in words:
                if len(word) > 5 and not re.search(r'[aeiouAEIOU]', word):
                    raise ValueError('The text contains English words that are not understood.')
                
                # منع تكرار الحروف الساكنة المبالغ فيه في كلمة واحدة
                if re.search(r'[^aeiouAEIOU\s]{6,}', word):
                    raise ValueError('The English words seem random and unreal.')
    
        # 5. فحص الـ Gibberish العام (رموز فقط)
        clean_text = re.sub(r'[^\w\s]', '', val).strip()
        if len(clean_text) < 5:
            raise ValueError('الوصف قصير جداً أو يحتوي على رموز فقط.')
        # 6. فحص اللغة (Language Check)
        try:
            lang = detect(val)
            if lang not in ['ar', 'en']:
                raise ValueError('يدعم العربية والإنجليزية فقط.')
        except:
            pass 
        
        return val







# ============================
# 3. Config & Rules Data
# ============================
ARABIC_SPECIALTY_RULES = {

    # 1️⃣ جهاز هضمي
    "Gastroenterologist": [
        "بطن", "مغص", "معده", "معدة", "حرقان", "قيء", "ترجيع",
        "اسهال", "إسهال", "امساك", "إمساك", "انتفاخ", "حموضة"
    ],

    # 2️⃣ قلب
    "Cardiologist": [
        "قلب", "خفقان", "نهجان", "ألم الصدر", "وجع صدر","صدري","صدرى","قلبي",
        "ضغط", "ضغط عالي", "تسارع ضربات"
    ],

    # 3️⃣ صدرية
    "Pulmonologist": [
        "كحه", "كحة", "سعال", "ضيق تنفس", "بلغم", "صدري","الم شديد في الصدر","الم في الصدر",
        "صدر", "صفير", "ربو", "حساسية صدر"
    ],

    # 4️⃣ مخ وأعصاب
    "Neurologist": [
        "صداع", "دوخه", "دوخة", "اغماء", "إغماء",
        "تنميل", "شلل", "تشنج", "رعشه", "رعشة","مصدعة","مصدع","مصدعه",
        "زغللة", "صداع نصفي"
    ],

    # 5️⃣ عظام
    "Orthopedic": [
        "عظم", "عظام", "مفصل", "ركبه", "ركبة","ركبتي","ركبتى","رجلي","رجلى",
        "كتف", "ضهر", "ظهر", "عمود فقري","ظهرى","ظهري","ضهري","ضهرى","كتفي","كتفى",
        "غضروف", "خشونة"
    ],

    # 6️⃣ جلدية
    "Dermatologist": [
        "جلد", "حكه", "حكة", "هرش", "طفح", "وشي", "وشى","وجهي","وجهى",
        "حبوب", "بثور", "تصبغات", "اكزيما",
        "أكزيما", "التهاب جلد"
    ],

    # 7️⃣ أنف وأذن وحنجرة
    "ENT Specialist": [
        "ودن", "أذن", "حلق", "زور", "أنف","مناخيري","مناخيرى","زورى","زوري",
        "رشح", "كحة ناشفة", "بحة", "صوت",
        "انسداد أنف", "اللوز"
    ],

    # 8️⃣ مسالك بولية
    "Urologist": [
        "بول", "تبول", "حرقان بول",
        "كلى", "كلي", "حصوة",
        "ألم جنب", "مجرى البول"
    ],

    # 9️⃣ أسنان
    "Dentist": [
        "سن", "ضرس", "لثه", "لثة","ضرسي","ضرسى","سني","سنى","اسنان","وجع اسنان","سناني",
        "وجع سن", "ألم ضرس", "نزيف لثة"
    ],

    # 🔟 حساسية ومناعة
    "Allergist / Immunologist": [
        "حساسية", "عطس", "كحة حساسية",
        "حكة حساسية", "طفح تحسسي",
        "تورم مفاجئ", "حساسية صدر"
    ],

    # 1️⃣1️⃣ غدد صماء
    "Endocrinologist": [
        "سكر", "سكري", "السكر",
        "غدة", "غدة درقية", "درقية",
        "هرمونات", "زيادة وزن",
        "نقص وزن بدون سبب"
    ],

    # 1️⃣2️⃣ جراح عام
    "General Surgeon": [
        "عملية", "جراحة", "فتق",
        "خراج", "ورم", "ألم شديد",
        "نزيف", "جرح"
    ],

    # 1️⃣3️⃣ أمراض دم
    "Hematologist": [
        "أنيميا", "فقر دم", "نزيف متكرر",
        "كدمات", "سيولة", "جلطات",
        "نقص صفائح"
    ],

    # 1️⃣4️⃣ أمراض معدية
    "Infectious Disease Specialist": [
        "سخونية", "حرارة", "عدوى",
        "التهاب", "فيروس", "بكتيريا",
        "كورونا", "انفلونزا"
    ],

    # 1️⃣5️⃣ كلى
    "Nephrologist": [
        "كلى", "كلي", "فشل كلوي",
        "تورم", "احتباس سوائل",
        "تحاليل كلى"
    ],

    # 1️⃣6️⃣ نساء وتوليد
    "Obstetrician-Gynecologist (OB-GYN)": [
        "دورة", "حيض", "تأخر دورة", "ولادة",
        "نزيف مهبلي", "ألم اسفل البطن",
        "حمل", "إفرازات", "مبيض", "رحم"
    ],

    # 1️⃣7️⃣ أورام
    "Oncologist": [
        "ورم", "سرطان", "كتلة",
        "نزول وزن شديد", "علاج كيماوي",
        "اشعاع"
    ],

    # 1️⃣8️⃣ روماتيزم
    "Rheumatologist": [
        "روماتيزم", "تيبس","وجع في مفصل"
        "ألم مفاصل صباحي","تورم في المفصل",
        "التهاب مفاصل",
        "تورم مفصل"
    ],

    # 1️⃣9️⃣ عيون
    "Ophthalmologist": [
        "عين", "زغللة", "تشويش",
        "احمرار العين", "ألم عين",
        "دموع", "ضعف نظر"
    ]
}


SPECIALTY_PRIORITY = {
    "Cardiologist": 5, "Neurologist": 5, "Pulmonologist": 5, "Oncologist": 5,
    "Gastroenterologist": 4, "Urologist": 4, "Nephrologist": 4, "Obstetrician-Gynecologist (OB-GYN)": 4,
    "Infectious Disease Specialist": 4, "Orthopedic": 3, "Rheumatologist": 3, "General Surgeon": 3,
    "ENT Specialist": 2, "Dermatologist": 2, "Ophthalmologist": 2, "Dentist": 2, "Endocrinologist": 2,
    "Hematologist": 2, "Allergist / Immunologist": 2
}

# ============================
# 4. Cleaning & Logic Functions
# ============================
def super_clean_arabic(text):
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ+", "", text)
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def arabic_rule_based_decision(text):
    text = text.lower()
    best_specialty, best_score, best_hits = None, 0, 0
    for specialty, keywords in ARABIC_SPECIALTY_RULES.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            score = (hits * 2) + SPECIALTY_PRIORITY.get(specialty, 1)
            if score > best_score:
                best_score, best_specialty, best_hits = score, specialty, hits
    if best_specialty is None: return None, 0
    confidence = min(0.95, 0.6 + (best_hits * 0.05) + (SPECIALTY_PRIORITY.get(best_specialty, 1) * 0.03))
    return best_specialty, round(confidence, 2)

def arabic_top3_candidates(text):
    text = text.lower()
    scores = []
    for specialty, keywords in ARABIC_SPECIALTY_RULES.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            priority = SPECIALTY_PRIORITY.get(specialty, 1)
            confidence = min(0.9, 0.4 + (hits * 0.07) + (priority * 0.04))
            scores.append((specialty, round(confidence, 2)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:3]

# ============================
# 5. Model Architecture & Setup
# ============================
class MedicalHybridSmart(nn.Module):
    def __init__(self, num_labels=19):
        super(MedicalHybridSmart, self).__init__()
        self.en_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.ar_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02")
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )
    def forward(self, input_ids_en, mask_en, input_ids_ar, mask_ar):
        out_en = self.en_model(input_ids=input_ids_en, attention_mask=mask_en)[1]
        out_ar = self.ar_model(input_ids=input_ids_ar, attention_mask=mask_ar)[1]
        combined = torch.cat((out_en, out_ar), dim=1)
        return self.classifier(combined)

from typing import Optional, List, Dict

class PredictionResponse(BaseModel):
    predicted_specialty: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    top_candidates: Optional[List[Dict]] = None





specialties_list = ['Allergist / Immunologist', 'Cardiologist', 'Dentist', 'Dermatologist', 'ENT Specialist', 'Endocrinologist', 'Gastroenterologist', 'General Surgeon', 'Hematologist', 'Infectious Disease Specialist', 'Nephrologist', 'Neurologist', 'Obstetrician-Gynecologist (OB-GYN)', 'Oncologist', 'Ophthalmologist', 'Orthopedic', 'Pulmonologist', 'Rheumatologist', 'Urologist']
specialties_list_ar = ['أخصائي حساسية ومناعة', 'طبيب قلب', 'طبيب أسنان', 'طبيب جلدية', 'طبيب أنف وأذن وحنجرة', 'أخصائي غدد صماء', 'أخصائي جهاز هضمي', 'جراح عام', 'أخصائي أمراض دم', 'أخصائي أمراض معدية', 'أخصائي أمراض كلى', 'أخصائي مخ وأعصاب', 'طبيب نساء وتوليد', 'أخصائي أورام', 'طبيب عيون', 'أخصائي عظام', 'أخصائي أمراض صدرية', 'أخصائي روماتيزم', 'أخصائي مسالك بولية']
SPECIALTY_AR_MAP = dict(zip(specialties_list, specialties_list_ar))

device = torch.device("cpu")
model = MedicalHybridSmart(num_labels=len(specialties_list))
try:
    model.load_state_dict(torch.load("medical_ai_model_final.pt", map_location=device))
    model.eval()
except:
    print("Warning: Model weights not found.")

bio_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
ara_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
translator = GoogleTranslator(source='auto', target='en')


# ============================
# ✅ Specialty Keywords (Arabic + English)
# ============================
SPECIALTY_KEYWORDS = {
    "Neurologist": [
        "صداع", "وجع راس", "رأس", "دوخة", "دوار", "إغماء",
        "تشنج", "رعشة", "تنميل", "خدر", "شلل", "فقدان وعي",
        "headache", "head pain", "dizziness", "vertigo",
        "fainting", "seizure", "tremor", "numbness",
        "tingling", "paralysis", "loss of consciousness"
    ],
    "Cardiologist": [
        "قلب", "ألم الصدر", "وجع صدر", "خفقان", "نهجان",
        "ضغط", "ضغط عالي", "تسارع ضربات القلب",
        "heart", "chest pain", "palpitation",
        "shortness of breath", "high blood pressure",
        "rapid heartbeat"
    ],
    "Pulmonologist": [
        "كحة", "سعال", "بلغم", "ضيق تنفس", "حساسية صدر",
        "صفير", "التهاب رئة",
        "cough", "sputum", "phlegm",
        "breathing difficulty", "wheezing",
        "asthma", "pneumonia", "lung pain"
    ],
    "Orthopedic": [
        "عظم", "عظام", "مفصل", "ركبة", "كتف",
        "ضهر", "ظهر", "عمود فقري", "غضروف",
        "انزلاق غضروفي", "خشونة", "تورم مفصل",
        "bone", "bones", "joint", "knee",
        "shoulder", "back pain", "spine",
        "disc", "slipped disc", "arthritis",
        "joint swelling"
    ],
    "Ophthalmologist": [
        "عين", "وجع عين", "زغللة", "تشويش",
        "ضعف نظر", "احمرار العين", "دموع",
        "eye", "eye pain", "blurred vision",
        "vision loss", "red eye", "tearing"
    ],
    "ENT Specialist": [
        "ودن", "أذن", "وجع ودن", "طنين",
        "أنف", "رشح", "انسداد أنف",
        "حلق", "زور", "صعوبة بلع", "بحة صوت",
        "ear", "ear pain", "tinnitus",
        "nose", "runny nose", "nasal congestion",
        "sore throat", "throat pain",
        "difficulty swallowing", "hoarseness"
    ],
    "Dermatologist": [
        "جلد", "حكة", "هرش", "طفح",
        "حساسية", "حبوب", "بثور",
        "بقع", "تصبغات", "أكزيما",
        "skin", "itching", "rash",
        "allergy", "acne", "pimples",
        "spots", "pigmentation", "eczema"
    ],
    "Gastroenterologist": [
        "بطن", "معدة", "وجع بطن", "مغص",
        "قيء", "ترجيع", "إسهال", "إمساك",
        "انتفاخ", "حموضة",
        "abdomen", "stomach pain",
        "abdominal pain", "vomiting",
        "nausea", "diarrhea", "constipation",
        "bloating", "acidity", "heartburn"
    ],
    "Urologist": [
        "بول", "حرقان بول", "تبول",
        "كثرة تبول", "ألم جنب",
        "كلى", "حصوة",
        "urine", "burning urination",
        "frequent urination", "kidney pain",
        "flank pain", "kidney stone"
    ],
    "Gynecologist": [
        "دورة", "حيض", "تأخر دورة",
        "نزيف", "ألم أسفل البطن",
        "إفرازات", "حمل", "مبيض", "رحم",
        "menstrual cycle", "period",
        "delayed period", "bleeding",
        "pelvic pain", "vaginal discharge",
        "pregnancy", "ovary", "uterus"
    ],
    "Pediatrist": [
        "طفل", "رضيع", "بيبي",
        "حرارة طفل", "قيء طفل",
        "إسهال طفل", "بكاء شديد",
        "child", "baby", "infant",
        "fever in child", "vomiting child",
        "diarrhea child"
    ],
    "Psychiatrist": [
        "قلق", "توتر", "اكتئاب",
        "أرق", "خوف", "هلع",
        "وسواس", "ضغط نفسي",
        "anxiety", "stress", "depression",
        "insomnia", "panic attack",
        "fear", "obsessive thoughts"
    ],
    "Infectious Disease Specialist": [
        "تعب عام", "إرهاق", "سخونية",
        "دوخة عامة", "ضعف", "فقدان وزن",
        "fatigue", "general weakness",
        "fever", "weight loss",
        "dizziness", "tiredness"
    ]
}






# ============================
# 7. Endpoints
# ============================
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/predict_specialty/", response_model=PredictionResponse)
@limiter.limit("5/minute") # Rate Limiting: 5 requests per minute
async def predict(request: Request, input_data: MedicalInput):
    try:
        original_text = input_data.text
        cleaned_text = super_clean_arabic(original_text)
        is_arabic_user = bool(re.search(r'[\u0600-\u06FF]', original_text))

        # --- Phase 1: Arabic Rule Based ---
        if is_arabic_user:
            rule_specialty, rule_conf = arabic_rule_based_decision(cleaned_text)
            
            if rule_conf < 0.5:
                top3 = arabic_top3_candidates(cleaned_text)
                if not top3:
                    return error_response("الوصف غير مفهوم", 400)

                top3_ar = [{"specialty": SPECIALTY_AR_MAP.get(s, s), "confidence": conf} for s, conf in top3]
                return PredictionResponse(
                    predicted_specialty="أكثر من احتمال",
                    confidence=min(rule_conf, 0.49),
                    message="الأعراض متداخلة، نعرض أقرب احتمالات حسب وصفك",
                    top_candidates=top3_ar
                )
            
            if rule_specialty:
                return PredictionResponse(
                    predicted_specialty=SPECIALTY_AR_MAP.get(rule_specialty, rule_specialty),
                    confidence=rule_conf
                )

        # --- Phase 2: Hybrid AI Model ---
        final_text_for_trans = cleaned_text if cleaned_text else original_text
        translated = translator.translate(final_text_for_trans)
        
        inputs_en = bio_tokenizer(translated, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
        inputs_ar = ara_tokenizer(final_text_for_trans, return_tensors="pt", padding='max_length', truncation=True, max_length=64)


        model.eval()
        with torch.no_grad():
            outputs = model(inputs_en['input_ids'], inputs_en['attention_mask'], inputs_ar['input_ids'], inputs_ar['attention_mask'])
            probs = F.softmax(outputs, dim=1)


         # ============================
        # ✅ Keyword Boosting (addition only)
        # ============================
        text = cleaned_text.lower()
        for idx, specialty in enumerate(specialties_list):
            keywords = SPECIALTY_KEYWORDS.get(specialty, [])
            for kw in keywords:
                if kw.lower() in text:
                    probs[0][idx] += 0.05

    

        top_prob, top_idx = torch.max(probs, dim=1)
        res_idx = top_idx.item()
        confidence = round(top_prob.item(), 2)

        predicted_specialty = specialties_list_ar[res_idx] if is_arabic_user else specialties_list[res_idx]
        msg = "تم التوقع بنجاح" if is_arabic_user else "Success"
        
        return PredictionResponse(
            predicted_specialty=predicted_specialty,
            confidence=confidence,
            message=msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
