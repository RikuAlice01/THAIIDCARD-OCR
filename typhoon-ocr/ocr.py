from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from io import BytesIO
from typing import  Optional
import logging
from pydantic import BaseModel
from utils.parse_thai_id_card import parse_thai_id_card
import asyncio
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil

class ThaiIDCardData(BaseModel):
    clean_text: Optional[str]
    citizen_id: Optional[str]
    prefix_th: Optional[str]
    name_th: Optional[str]
    lastname_th: Optional[str]
    prefix_en: Optional[str]
    name_en: Optional[str]
    lastname_en: Optional[str]
    dob: Optional[str]
    religion: Optional[str]
    address: Optional[str]
    alley: Optional[str]
    village: Optional[str]
    subdistrict: Optional[str]
    district: Optional[str]
    province: Optional[str]
    issued_date: Optional[str]
    expired_date: Optional[str]

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Typhoon OCR API (Optimized)",
    description="High-performance OCR API using Typhoon OCR 7B model with optimizations",
    version="2.0.0"
)

# Global variables for model and processor
model = None
processor = None
device = None
thread_pool = None

# Cache สำหรับ prompts ที่ใช้บ่อย
@lru_cache(maxsize=10)
def get_cached_prompt(prompt_type: str = "id_card") -> str:
    """Cache prompts ที่ใช้บ่อย"""
    if prompt_type == "id_card":
        return """You are given extracted text from a Thai National ID card. Your task is to extract the following fields exactly as they appear on the card. If a field is missing or unreadable, return "none" for that field. Output the result in the following JSON format:

{
  "citizen_id": "",
  "prefix_th": "",
  "name_th": "",
  "lastname_th": "",
  "prefix_en": "",
  "name_en": "",
  "lastname_en": "",
  "dob": "",
  "religion": "", 
  "address": "",
  "alley": "", 
  "village": "", 
  "subdistrict": "",
  "district": "",
  "province": "",
  "issued_date": "",
  "expired_date": ""
}

**Definitions:**
- "citizen_id": 13-digit Thai national ID number
- "prefix_th", "name_th", "lastname_th": Thai full name split into parts
- "prefix_en", "name_en", "lastname_en": English full name split into parts
- "dob": Date of birth (exact format as printed on the card, usually in Thai date)
- "religion": Religion listed on the card (e.g., พุทธ, อิสลาม, etc.)
- "address": Full address as a single string
- "alley", "village", "subdistrict", "district", "province": Extracted components of the address
- "issued_date": Card issue date (exact format as printed)
- "expired_date": Card expiration date (exact format as printed)

Ensure all Thai and English text is preserved exactly. Use "none" if any field cannot be determined.

Here is the OCR text input:
"""
    return prompt_type

def optimize_gpu_settings():
    """ปรับแต่งการใช้งาน GPU เพื่อประสิทธิภาพสูงสุด"""
    if torch.cuda.is_available():
        # เปิด TensorFloat-32 (TF32) สำหรับ Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # เปิด cudnn benchmark สำหรับ input sizes ที่คงที่
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # ปรับแต่ง memory management
        torch.cuda.empty_cache()
        
        # Set memory fraction หากต้องการจำกัดการใช้ memory
        # torch.cuda.set_per_process_memory_fraction(0.8)
        
        logger.info("GPU optimizations applied")

def precompile_model():
    """Pre-compile model สำหรับ inference ที่เร็วขึ้น"""
    global model
    if model is not None and torch.cuda.is_available():
        try:
            # Warm up model with dummy input
            dummy_input = torch.randint(0, 1000, (1, 100)).to(device)
            with torch.no_grad():
                _ = model(input_ids=dummy_input)
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

@app.on_event("startup")
async def startup_event():
    """โหลดโมเดลเมื่อเริ่มต้น server พร้อมการปรับแต่ง"""
    global model, processor, device, thread_pool
    
    try:
        logger.info("กำลังโหลดโมเดลพร้อมการปรับแต่ง...")
        
        # ตรวจสอบ system resources
        logger.info(f"Available CPU cores: {psutil.cpu_count()}")
        logger.info(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # ตั้งค่า thread pool สำหรับ async operations
        thread_pool = ThreadPoolExecutor(max_workers=min(4, psutil.cpu_count()))
        
        # ตรวจสอบ CUDA และปรับแต่ง
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            optimize_gpu_settings()
        logger.info(f"Using device: {device}")
        
        # โหลดโมเดลและ processor พร้อมการปรับแต่ง
        start_time = time.time()
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "scb10x/typhoon-ocr-7b", 
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()
        
        processor = AutoProcessor.from_pretrained(
            "scb10x/typhoon-ocr-7b", 
            use_fast=True,  # ใช้ fast tokenizer
            padding_side="left"  # สำหรับ batch processing
        )
        
        # Pre-compile และ warm up model
        precompile_model()
        
        load_time = time.time() - start_time
        logger.info(f"โหลดโมเดลสำเร็จ! ใช้เวลา {load_time:.2f} วินาที")
        
        # แสดงข้อมูล memory usage
        if torch.cuda.is_available():
            logger.info(f"GPU Memory used: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """ทำความสะอาดเมื่อปิด server"""
    global model, processor, thread_pool
    
    if thread_pool:
        thread_pool.shutdown(wait=True)
    
    if model is not None:
        del model
    if processor is not None:
        del processor
    
    # ทำความสะอาด memory อย่างละเอียด
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.info("ทำความสะอาด resources สำเร็จ")

def optimize_image(image: Image.Image) -> Image.Image:
    """ปรับแต่งรูปภาพเพื่อประสิทธิภาพที่ดีขึ้น"""
    # แปลงเป็น RGB ถ้าจำเป็น
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # ปรับขนาดแบบ smart - คงอัตราส่วนและไม่ให้ใหญ่เกินไป
    max_size = 620  # ลดขนาดเพื่อความเร็ว
    ratio = min(max_size / image.width, max_size / image.height)
    
    if ratio < 1:
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

async def perform_ocr_optimized(image: Image.Image, prompt: str) -> str:
    """ทำ OCR ที่ปรับแต่งแล้วเพื่อความเร็ว"""
    try:
        start_time = time.time()
        
        # ปรับแต่งรูปภาพใน thread pool
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(thread_pool, optimize_image, image)
        
        # สร้าง conversation
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # ใช้ apply_chat_template
        text_prompt = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True,
            tokenize=False
        )

        # Process inputs
        inputs = processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # จำกัด input length
        )
        
        # ย้าย inputs ไป device
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}

        # Generate with optimized settings
        with torch.no_grad():
            # ใช้ torch.compile หากรองรับ (Python 3.11+, PyTorch 2.0+)
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    compiled_generate = torch.compile(model.generate, mode="reduce-overhead")
                    output = compiled_generate(
                        **inputs,
                        max_new_tokens=800,  # ลดจำนวน tokens
                        temperature=0.1,
                        do_sample=True,
                        repetition_penalty=1.1,  # ลดค่าเล็กน้อยเพื่อความเร็ว
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                        # เพิ่มการปรับแต่งสำหรับความเร็ว
                        early_stopping=True,
                        num_beams=1,  # ใช้ greedy decoding แทน beam search
                    )
                except Exception:
                    # Fallback หาก compile ไม่สำเร็จ
                    output = model.generate(
                        **inputs,
                        max_new_tokens=800,
                        temperature=0.1,
                        do_sample=True,
                        repetition_penalty=1.1,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                        early_stopping=True,
                        num_beams=1,
                    )
            else:
                output = model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.1,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    early_stopping=True,
                    num_beams=1,
                )

        # Decode output
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_len:]
        decoded = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        processing_time = time.time() - start_time
        logger.info(f"OCR processing completed in {processing_time:.2f} seconds")

        return decoded[0].strip()

    except Exception as e:
        logger.error(f"Error in OCR processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/ocr/id-card", response_model=ThaiIDCardData)
async def ocr_id_card_optimized(file: UploadFile = File(...)):
    """
    ทำ OCR บัตรประชาชนแบบปรับแต่งแล้ว - เร็วขึ้นและประสิทธิภาพดีขึ้น
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        start_time = time.time()
        
        # อ่านไฟล์แบบ async
        contents = await file.read()
        
        # เปิดรูปภาพใน thread pool เพื่อไม่ให้ block
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(thread_pool, lambda: Image.open(BytesIO(contents)))
        
        # ใช้ cached prompt
        prompt = get_cached_prompt("id_card")
        
        # ทำ OCR
        ocr_result = await perform_ocr_optimized(image, prompt)
        print(f"OCR Result: {ocr_result}")
        # แยกข้อมูลจากผลลัพธ์ OCR
        parsed_data = parse_thai_id_card(ocr_result)
        
        total_time = time.time() - start_time
        logger.info(f"Total request processing time: {total_time:.2f} seconds")
        
        return parsed_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing ID card: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process ID card: {str(e)}")

@app.get("/")
async def root():
    """หน้าแรกของ API"""
    return {
        "message": "Typhoon OCR API (Optimized)",
        "version": "2.0.0",
        "optimizations": [
            "GPU optimizations (TF32, cudNN benchmark)",
            "Model warm-up and compilation",
            "Image preprocessing optimization",
            "Async processing with thread pools",
            "Cached prompts",
            "Reduced token generation",
            "Smart memory management"
        ],
        "endpoints": {
            "/ocr/id-card": "Thai ID card OCR - optimized for speed",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # เพิ่มการปรับแต่งสำหรับ uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,  # ใช้ 1 worker เพื่อหลีกเลี่ยงการโหลดโมเดลหลายครั้ง
        loop="uvloop",  # ใช้ uvloop สำหรับประสิทธิภาพที่ดีขึ้น
        http="httptools",  # ใช้ httptools สำหรับ HTTP parsing ที่เร็วขึ้น
        access_log=False,  # ปิด access log เพื่อประสิทธิภาพ
        # ssl_keyfile=None,
        # ssl_certfile=None,
    )