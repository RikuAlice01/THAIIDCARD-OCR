from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from rapidfuzz import process,fuzz
import easyocr
import numpy as np
import cv2
import re
import torch

app = FastAPI()

# ตรวจสอบ CUDA
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# โหลด EasyOCR Reader ครั้งเดียว
reader = easyocr.Reader(['th', 'en'], gpu=torch.cuda.is_available())

@app.post("/ocr/id-card")
async def ocr_id_card(file: UploadFile = File(...)):
    image = await file.read()
    npimg = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # อ่าน OCR
    results = reader.readtext(img, detail=0)
    print(results)

    # ดึงข้อความมา
    texts = [text for text in results]

    # (ตัวอย่างง่าย) แปลงผลลัพธ์เป็น JSON
    response = {
        # "results":results,
        "full_text": " ".join(texts),
        "fields": extract_fields(texts)
    }
    return JSONResponse(content=response)

# ดึงข้อมูลเฉพาะออกมาอย่างง่าย
def extract_fields(texts):
    if isinstance(texts, list):
        text = " ".join(texts)
    else:
        text = texts

    # Cleaning Text
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace('|', '1').replace('ฺ', '').replace(']', 'l')
    text = text.replace("aทธ", "พุทธ")
    text = re.sub(r"เก[็ด]วัน[ทีที่]", "เกิดวันที่", text)
    text = re.sub(r"(หมู่ที|หมูที่|หม่ที่|หมูที|หมที|หม่ที)", "หมู่ที่", text)
    text = re.sub(r"\bสกล\b", "สกุล", text)
    text = re.sub(r"\bน\.ส\.\b", "นางสาว", text)
    text = re.sub(r"\bว่าที\b", "ว่าที่", text)

    # คำที่ถูกต้อง
    target_keywords = [
        "ชื่อตัวและชื่อสกุล",
        "บัตรประจำตัวประชาชน",
        "thai national id card",
        "เลขประจำตัวประชาชน",
        "identification number",
        "เกิดวันที่",
        "date of birth",
        "ศาสนา",
        "ที่อยู่",
        "หมู่ที่",
        "หมู่",
        "วันออกบัตร",
        "วันบัตรหมดอายุ",
        "date of issue",
        "date of expiry",
        "เจ้าพนักงานออกบัตร"
    ]

    # สำหรับเก็บคำผิดที่พบและแทนที่
    replacements = {}

    # split ข้อความเป็นคำหรือคำกลุ่ม (ใช้ regex แยกคำรวมกับ whitespace/จุด)
    tokens = re.findall(r"[^\s]+", text)

    # ตรวจหาคำผิดจากทุก token ด้วย fuzzy matching
    for token in tokens:
        match = process.extractOne(token, target_keywords, scorer=fuzz.ratio, score_cutoff=60)
        if match:
            correct_word = match[0]
            # เก็บคำผิดและคำถูกที่จับได้
            if token != correct_word:
                replacements[token] = correct_word

    # แทนที่คำผิดด้วยคำที่ถูก
    for wrong, right in replacements.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text)

    data = {
        "clean_text": text,
        "citizen_id": None,
        "prefix_th": None,
        "name_th": None,
        "lastname_th": None,
        "prefix_en": None,
        "name_en": None,
        "lastname_en": None,
        "dob": None,
        "religion": None,
        "address": None,
        "alley": None,
        "village": None,
        "subdistrict": None,
        "district": None,
        "province": None,
        "issued_date": None,
        "expired_date": None
    }

    # 1. เลขบัตรประชาชน
    cid_match = re.search(r"\b\d\s?\d{4}\s?\d{5}\s?\d{2}\s?\d\b", data["clean_text"])
    if cid_match:
        data["citizen_id"] = cid_match.group().replace(" ", "")

    # 2. ชื่อ-นามสกุล ภาษาไทย + คำนำหน้า
    # กรณีมีวงเล็บ เช่น พระมหา(ไชยสยาม ปัญญาคโม (เสรีมาศ))
    match_special = re.search(
        r"(นาย|นางสาว|น\.ส\.|พระมหา|ว่าที่ ร\.ต\.|ว่าที ร\.ต\.|ร\.ต\.|ด\.ร\.|นาง|น\.ส\.|เด็กชาย|เด็กหญิง|ว่าที่ ร.ต.|ว่าที ร.ต.|พระ|พลทหาร|สิบเอก|จ่าสิบเอก|พลฯ|พล.ท.|ดร\.?)\s*\(*([ก-๙]+)\s+([ก-๙().\s]+?)\)*\s+(?:name|last name)",
        data["clean_text"]
    )

    # กรณีทั่วไป
    match_normal = re.search(
        r"(นาย|นางสาว|น\.ส\.|พระมหา|ว่าที่ ร\.ต\.|ว่าที ร\.ต\.|ร\.ต\.|ด\.ร\.|นาง|น\.ส\.|เด็กชาย|เด็กหญิง|ว่าที่ ร.ต.|ว่าที ร.ต.|พระ|พลทหาร|สิบเอก|จ่าสิบเอก|พลฯ|พล.ท.|ดร\.?)\s+([ก-๙]+)\s+([ก-๙().\s]+?)\s+(?:name|last name)",
        data["clean_text"]
    )

    if match_special:
        data["prefix_th"] = match_special.group(1).strip()
        data["name_th"] = match_special.group(2).strip()
        data["lastname_th"] = match_special.group(3).strip()
    elif match_normal:
        data["prefix_th"] = match_normal.group(1).strip()
        data["name_th"] = match_normal.group(2).strip()
        data["lastname_th"] = match_normal.group(3).strip()

    # 3. ชื่อ-นามสกุล ภาษาอังกฤษ
    name_block_match = re.search(r"name\s+([a-z. ]+?)\s+last\s+name", data["clean_text"], re.IGNORECASE)
    lastname_en_match = re.search(r"last\s+name[\s:]+([a-z]+)", data["clean_text"], re.IGNORECASE)

    if name_block_match:
        name_parts = name_block_match.group(1).split()
        if len(name_parts) >= 2:
            data["prefix_en"] = " ".join(name_parts[:-1]).capitalize()
            data["name_en"] = name_parts[-1].capitalize()
        elif len(name_parts) == 1:
            data["name_en"] = name_parts[0].capitalize()

    if lastname_en_match:
        data["lastname_en"] = lastname_en_match.group(1).capitalize()

    # 4. วันเกิด
    dob_match = re.search(r"([0-9]{1,2}\s*[ก-๙a-z.]+\s*[0-9]{4}).{0,20}(เกิดวันที่|date of birth|date ot birth)", data["clean_text"])
    if dob_match:
        data["dob"] = dob_match.group(1).strip()

    # 5. ศาสนา
    religion_match = re.search(r"ศาสนา\s*([ก-๙a-z]+)", data["clean_text"])
    if religion_match:
        data["religion"] = religion_match.group(1).capitalize()

    # 6. ที่อยู่
    address_match = re.search(
        r"(\d+/\d+|\d+)\s+(?=หมู่ที่|หมู่ที|หมู่|หม่ที|ม\.|ต\.|อ\.|จ\.|ซ\.|ช\.)",
        data["clean_text"]
    )
    if address_match:
        data["address"] = address_match.group(1).strip()


    alley_match = re.search(r"(ซอย|ซ\.|ช\.)\s*([ก-๙]+)", data["clean_text"])
    if alley_match:
        data["alley"] = alley_match.group(2)

    village_match = re.search(r"(หมู่ที่|หมู่|ม\.)\s*(\d{1,5})", data["clean_text"])
    if village_match:
        data["village"] = village_match.group(2)

    # 7. ตำบล / แขวง
    subdistrict_match = re.search(r"(ตำบล| ต\.|แขวง)\s*([ก-๙]+)", data["clean_text"])
    if subdistrict_match:
        data["subdistrict"] = subdistrict_match.group(2)

    # 8. อำเภอ / เขต
    district_match = re.search(r"(อำเภอ| อ\.|เขต)\s*([ก-๙]+)", data["clean_text"])
    if district_match:
        data["district"] = district_match.group(2)

    # 9. จังหวัด
    province_match = re.search(r"(จังหวัด|จ\.)\s*([ก-๙]+)", data["clean_text"])
    if province_match:
        data["province"] = province_match.group(2)
    else:
        cleaned_text = re.sub(r"[^ก-๙a-zA-Z0-9\s]", "", data["clean_text"].lower())
        provinces = [
            "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น",
            "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่",
            "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช",
            "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
            "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก",
            "เพชรบุรี", "เพชรบูรณ์", "แพร่", "พรรคใต้", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
            "ยะลา", "ยโสธร", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน",
            "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร",
            "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์",
            "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี",
            "อุบลราชธานี"
        ]
        result = process.extractOne(cleaned_text, provinces, score_cutoff=50)
        if result:
            data["province"] = result[0]

    # 10. วันออกบัตร / วันหมดอายุ
    card_dates_match = re.search(
        r"([0-9]{1,2}\s*[ก-๙a-z.]+\s*[0-9]{4}).{0,30}วันออกบัตร.{0,30}([0-9]{1,2}\s*[ก-๙a-z.]+\s*[0-9]{4})",
        data["clean_text"]
    )
    if card_dates_match:
        data["issued_date"] = card_dates_match.group(1).strip()
        data["expired_date"] = card_dates_match.group(2).strip()

    return data
