import re
import json
from rapidfuzz import process
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def parse_thai_id_card(text: str) -> Dict[str, Any]:
    """
    Enhanced Thai ID card parser with improved accuracy and error handling.
    
    Args:
        text (str): Raw text extracted from Thai ID card
        
    Returns:
        Dict[str, Any]: Parsed data with all extracted fields
    """
    
    data = {
        "clean_text": text.strip(),
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
        "expired_date": None,
        "gender": None,
        "postal_code": None
    }

    try:
        # --- 1. Enhanced Citizen ID extraction ---
        citizen_patterns = [
            r'(?:เลขประจำตัวประชาชน|ID Card|Number|Identification Number)[:\s\-]*([0-9\s\-]{13,})',
            r'(?:^|\s)([0-9]\s+[0-9]{4}\s+[0-9]{5}\s+[0-9]{2}\s+[0-9])',  # Format: 1 1037 02071 81 1
            r'(?:^|\s)([0-9]\s[0-9]{4}\s[0-9]{5}\s[0-9]{2}\s[0-9])',      # Alternative spacing
            r'([0-9]-[0-9]{4}-[0-9]{5}-[0-9]{2}-[0-9])',                   # Dash format
        ]
        
        for pattern in citizen_patterns:
            m = re.search(pattern, data["clean_text"], re.MULTILINE)
            if m:
                digits = re.sub(r'\D', '', m.group(1))
                if len(digits) == 13:
                    data["citizen_id"] = digits
                    break

        # --- 2. Enhanced Thai name extraction ---
        # Try multiple patterns for Thai names
        thai_name_patterns = [
            r'ชื่อตัวและชื่อสกุล[:\s\-]*\n?([ก-๙A-Za-z.\s\(\)]+)',
            r'ชื่อสกุล[:\s\-]*([ก-๙A-Za-z.\s\(\)]+)',
            r'##\s*ชื่อตัวและชื่อสกุล[:\s\-]*\n?([ก-๙A-Za-z.\s\(\)]+)',
        ]
        
        for pattern in thai_name_patterns:
            m = re.search(pattern, data["clean_text"])
            if m:
                full_name = m.group(1).strip()
                # Remove parentheses content (like aliases)
                full_name = re.sub(r'\([^)]+\)', '', full_name).strip()
                parts = full_name.split()

                # Enhanced Thai prefixes
                known_prefixes_th = [
                    "เด็กชาย", "เด็กหญิง", "ด.ช.", "ด.ญ.", "นาย", "นาง", "นางสาว", "น.ส.", "ดร.", 
                    "ว่าที่ ร.ต.", "ว่าที่ ร.ท.", "ว่าที่ ร.อ.", "ร.ต.", "ร.ท.", "ร.อ.", 
                    "พ.ต.", "พ.ท.", "พ.อ.", "พระ", "พระมหา", "สมเด็จพระ", "หลวงพ่อ"
                ]

                # Check for multi-word prefixes first
                prefix_found = False
                for i in range(min(3, len(parts)), 0, -1):
                    possible_prefix = " ".join(parts[:i])
                    if possible_prefix in known_prefixes_th:
                        data["prefix_th"] = possible_prefix
                        if i < len(parts):
                            data["name_th"] = parts[i]
                        if i + 1 < len(parts):
                            data["lastname_th"] = " ".join(parts[i+1:])
                        prefix_found = True
                        break
                
                if not prefix_found and len(parts) >= 2:
                    # Fallback: assume first part is prefix if it's common
                    if parts[0] in known_prefixes_th:
                        data["prefix_th"] = parts[0]
                        data["name_th"] = parts[1] if len(parts) > 1 else ""
                        data["lastname_th"] = " ".join(parts[2:]) if len(parts) > 2 else ""
                    else:
                        # No recognized prefix, treat as name + lastname
                        data["name_th"] = parts[0]
                        data["lastname_th"] = " ".join(parts[1:])
                break

        # --- 3. Enhanced English name extraction ---
        # Handle different English name formats
        name_patterns = [
            r'Name[:\s\-]*([A-Z][a-zA-Z. ]+?)(?:\n|Last name)',
            r'##\s*Name[:\s\-]*\n?-?\s*([A-Z][a-zA-Z. ]+)',
        ]
        
        lastname_patterns = [
            r'Last Name[:\s\-]*([A-Z][a-zA-Z.\s]+)',
             r'##\s*Last Name[:\s\-]*\n?-?\s*([A-Z][a-zA-Z.\s]+)',
            r'Last name[:\s\-]*([A-Z][a-zA-Z.\s]+)',
            r'##\s*Last name[:\s\-]*\n?-?\s*([A-Z][a-zA-Z.\s]+)',
        ]

        # Extract English first name
        for pattern in name_patterns:
            name_match = re.search(pattern, data["clean_text"])
            if name_match:
                name_line = name_match.group(1).strip()
                name_parts = name_line.split()

                # Enhanced English prefixes
                known_prefixes_en = [
                    "Mr.", "Mrs.", "Ms.", "Miss", "Dr.", "Acting Sub.Lt", "Sub.Lt", 
                    "Lt.", "Capt.", "Major", "Col.", "Gen.", "Professor", "Prof."
                ]

                # Check for multi-word prefixes
                prefix_found = False
                for i in range(min(3, len(name_parts)), 0, -1):
                    possible_prefix = " ".join(name_parts[:i])
                    if possible_prefix in known_prefixes_en:
                        data["prefix_en"] = possible_prefix
                        if i < len(name_parts):
                            data["name_en"] = name_parts[i]
                        prefix_found = True
                        break
                
                if not prefix_found:
                    data["name_en"] = name_parts[0] if name_parts else name_line
                break

        # Extract English last name
        for pattern in lastname_patterns:
            lastname_match = re.search(pattern, data["clean_text"])
            if lastname_match:
                data["lastname_en"] = lastname_match.group(1).strip()
                break

        # --- 4. Enhanced Date of Birth extraction ---
        dob_patterns = [
            r'(?:เกิดวันที่|Date of Birth)[:\s\-]*(\d{1,2}[\s/.-][ก-๙A-Za-z\.]+[\s/.-]\d{4})',
            r'##\s*เกิดวันที่[:\s\-]*\n?-?\s*(\d{1,2}[\s/.-][ก-๙A-Za-z\.]+[\s/.-]\d{4})',
        ]
        for pattern in dob_patterns:
            m = re.search(pattern, data["clean_text"])
            if m:
                data["dob"] = m.group(1).strip()
                break

        # --- 5. Enhanced Religion extraction ---
        religion_patterns = [
            r'ศาสนา[:\s\-]*([ก-๙]+)', 
            r'Religion[:\s\-]*([A-Za-z]+)',
            r'##\s*ศาสนา[:\s\-]*\n?-?\s*([ก-๙]+)'
        ]
        for pattern in religion_patterns:
            m = re.search(pattern, data["clean_text"])
            if m:
                data["religion"] = m.group(1).strip()
                break

        # --- 6. Enhanced Address parsing ---
        address_patterns = [
            r'ที่อยู่[:\s]*([^#]+?)(?=##|$)',
            r'##\s*ที่อยู่[:\s]*\n?([^#]+?)(?=##|$)',
        ]
        
        for pattern in address_patterns:
            m = re.search(pattern, data["clean_text"], re.DOTALL)
            if m:
                full_address = m.group(1).strip()
                
                # House number
                house_match = re.search(r'(\d+(?:/\d+)?)', full_address)
                if house_match:
                    data["address"] = house_match.group(1)

                # Alley (ซอย)
                alley_match = re.search(r'(?:ซอย|ซ\.|ช\.)\s*([ก-๙A-Za-z0-9\-/]+)', full_address)
                if alley_match:
                    data["alley"] = alley_match.group(1)

                # Village (หมู่)
                village_patterns = [
                    r'(?:หมู่ที่|หมู่|หมูที่|หม่ที่|หมูที|หม่ที|หมที\.)\s*(\d+)',
                    r'หมู่\s*(\d+)'
                ]
                for vp in village_patterns:
                    village_match = re.search(vp, full_address)
                    if village_match:
                        data["village"] = village_match.group(1)
                        break

                # Subdistrict (ตำบล/แขวง)
                subdistrict_match = re.search(r'(?:ตำบล|ต\.|แขวง)\s*([ก-๙A-Za-z]+)', full_address)
                if subdistrict_match:
                    data["subdistrict"] = subdistrict_match.group(1)

                # District (อำเภอ/เขต)
                district_match = re.search(r'(?:อำเภอ|อ\.|เขต)\s*([ก-๙A-Za-z]+)', full_address)
                if district_match:
                    data["district"] = district_match.group(1)

                # Province (จังหวัด)
                province_match = re.search(r'(?:จ\.|จังหวัด)\s*([ก-๙A-Za-z]+)', full_address)
                if province_match:
                    data["province"] = province_match.group(1)
                else:
                    # Fuzzy matching for provinces
                    provinces = [
                        "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น",
                        "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่",
                        "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช",
                        "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
                        "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก",
                        "เพชรบุรี", "เพชรบูรณ์", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
                        "ยะลา", "ยโสธร", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน",
                        "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร",
                        "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์",
                        "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี",
                        "อุบลราชธานี"
                    ]
                    
                    # Clean text for fuzzy matching
                    cleaned_address = re.sub(r"[^ก-๙a-zA-Z0-9\s]", "", full_address.lower())
                    result = process.extractOne(cleaned_address, provinces, score_cutoff=60)
                    if result:
                        data["province"] = result[0]

                # Postal code
                postal_match = re.search(r'(\d{5})', full_address)
                if postal_match:
                    data["postal_code"] = postal_match.group(1)
                break

        # --- 7. Enhanced Date of Issue extraction ---
        issue_patterns = [
            r'(?:วันออกบัตร|Date of Issue|Issue Date)[:\s\-]*(\d{1,2}[\s/.-][ก-๙A-Za-z\.]+[\s/.-]\d{4})',
            r'##\s*วันออกบัตร[:\s\-]*\n?-?\s*(\d{1,2}[\s/.-][ก-๙A-Za-z\.]+[\s/.-]\d{4})',
        ]
        for pattern in issue_patterns:
            m = re.search(pattern, data["clean_text"])
            if m:
                data["issued_date"] = m.group(1).strip()
                break

        # --- 8. Enhanced Expiry Date extraction ---
        expiry_patterns = [
            r'(?:วันหมดอายุ|Expiry Date|Date of Expiry)[:\s\-]*(\d{1,2}[\s/.-][ก-๙A-Za-z\.]+[\s/.-]\d{4})',
            r'##\s*วันหมดอายุ[:\s\-]*\n?-?\s*(\d{1,2}[\s/.-][ก-๙A-Za-z\.]+[\s/.-]\d{4})',
        ]
        for pattern in expiry_patterns:
            m = re.search(pattern, data["clean_text"])
            if m:
                data["expired_date"] = m.group(1).strip()
                break

        # --- 9. Gender extraction (new feature) ---
        gender_patterns = [
            r'(?:เพศ|Gender)[:\s\-]*([ก-๙A-Za-z]+)',
            r'(?:ผู้ชาย|ผู้หญิง|Male|Female)'
        ]
        for pattern in gender_patterns:
            m = re.search(pattern, data["clean_text"])
            if m:
                gender_text = m.group(0).lower()
                if any(word in gender_text for word in ['ผู้ชาย', 'male', 'ชาย']):
                    data["gender"] = "ชาย"
                elif any(word in gender_text for word in ['ผู้หญิง', 'female', 'หญิง']):
                    data["gender"] = "หญิง"
                break

    except Exception as e:
        logger.error(f"Error parsing Thai ID card: {str(e)}")
        # Continue execution even if there's an error, return partial data

    return data


def validate_citizen_id(citizen_id: str) -> bool:
    """
    Validate Thai citizen ID using the checksum algorithm.
    
    Args:
        citizen_id (str): 13-digit citizen ID
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not citizen_id or len(citizen_id) != 13 or not citizen_id.isdigit():
        return False
    
    # Thai ID validation algorithm
    total = 0
    for i in range(12):
        total += int(citizen_id[i]) * (13 - i)
    
    remainder = total % 11
    check_digit = (11 - remainder) % 10
    
    return int(citizen_id[12]) == check_digit


def format_output(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format and clean the parsed output.
    
    Args:
        parsed_data (Dict[str, Any]): Raw parsed data
        
    Returns:
        Dict[str, Any]: Cleaned and formatted data
    """
    # Remove clean_text from output
    output = {k: v for k, v in parsed_data.items() if v is not None}
    
    # Add validation status for citizen ID
    if output.get("citizen_id"):
        output["citizen_id_valid"] = validate_citizen_id(output["citizen_id"])

    return format_output(output)


# Test with your sample data
if __name__ == "__main__":
    json_text = [
        "{\"natural_text\": \"# บัตรประจำตัวประชาชน Thai National ID Card\\n\\n**เลขประจำตัวประชาชน:** 1 037 02071 81 1\\n\\n**ชื่อตัวและชื่อสกุล (Thai):** น.ส. ณัฐธีรา ยางสวาย\\n\\n**Name (English):** Miss Nattarika\\n\\n**Last Name (English):** Yangsuai\\n\\n**เกิดวันที่:** 25 มิ.ย. 2539\\n\\n**Date of Birth:** 25 Jun. 1996\\n\\n**สถานะ:** ผู้เสียภาษี\\n\\n**ที่อยู่:** 111/17 หมู่ที่ 2 ต.ลาดหญ้า อ.เมืองกาญจนบุรี จ.กาญจนบุรี 24 ก.ค. 2563\\n\\n**วันออกบัตร:** 24 มิ.ย. 2572\\n\\n**Date of Issue:** 24 Jun. 2019\\n\\n**วันหมดอายุ:** 24 มิ.ย. 2581\\n\\n**Date of Expiry:** 24 Jun. 2028\\n\\n**ที่อยู่ (English):** 111/17 หมู่ที่ 2 ต.ลาดหญ้า อ.เมืองกาญจนบุรี จ.กาญจนบุรี 24 ก.ค. 2563\\n\\n**ชื่อเจ้าหน้าที่ผู้ออกบัตร:** นายธนาคม จริงจิระ\"}",
        "{\"text\": \"# บัตรประจำตัวประชาชน Thai National ID Card\\n\\n## เลขประจำตัวประชาชน\\n- 3 4001 00212 47 7\\n\\n## ชื่อตัวและชื่อสกุล\\n- พระมหา ไชยสยาม ปัญญาโคโน่ (เสรีมาศ)\\n\\n## Name\\n- Mr. Chaisayam\\n\\n## Last name\\n- Serimat\\n\\n## เกิดวันที่\\n- 16 พ.ย. 2515\\n\\n## Date of Birth\\n- 16 Nov. 1972\\n\\n## สถานะการสมัคร\\n- ผู้ชาย\\n\\n## ที่อยู่\\n- 1 หมู่ที่ 10 ต.หนองไม้แก่น อ.แปลงยาว จ.ฉะเชิงเทรา\\n\\n## วันออกบัตร\\n- 8 ก.ค. 2554\\n\\n## Date of Issue\\n- 08 Jul 2011\\n\\n## วันหมดอายุ\\n- 15 พ.ย. 2563\\n\\n## Expiry Date\\n- 15 Nov. 2020\\n\\n## รหัสประจำตัวประชาชน\\n- 150-150-140-140\\n\\n## ภาพประกอบ\\n![](attachment://path_to_image)\\n\\n## หมายเหตุ\\n- บัตรนี้มีวงกลมสีแดงที่หมายเลข 14 และ 15 ของเลขประจำตัวประชาชน\"}",
        "{\"Text\": \"# บัตรประจำตัวประชาชน Thai National ID Card\\n\\n## เลขประจำตัวประชาชน\\n- 8 9031 15238 54 2 1\\n\\n## ชื่อตัวและชื่อสกุล\\n- ว่าที่ ร.ต. ณัฏฐาวีรกร ตาซื่อ ณ อยุธยา\\n\\n## Name\\n- Acting Sub.Lt Nutthaweerakorn\\n\\n## Last name\\n- Tasue\\n\\n## เกิดวันที่\\n- 16 พ.ค. 2516\\n\\n## Date of Birth\\n- 16 May 2013\\n\\n## สถานที่เกิด\\n- โรงพยาบาลเด็กสมิติเวช กรุงเทพมหานคร\\n\\n## รหัสประจำตัวประชาชน\\n- 110 170 160 160 150 150 140 140 130 130\\n\\n## วันออกบัตร\\n- 18 เม.ย. 2556\\n\\n## Issue Date\\n- 18 Apr 2013\\n\\n## วันหมดอายุ\\n- 18 เม.ย. 2565\\n\\n## Expiry Date\\n- 18 Apr 2022\\n\\n## ที่อยู่:\\n- 99/1 หมู่ 5 ถนนเลี่ยงเมืองปากเกร็ด ตำบลบางตลาด อำเภอปากเกร็ด นนทบุรี 11120\\n\\n## รหัสไปรษณีย์: 11120\\n\\n## วันที่ออกบัตร: 18 เมษายน 2556\\n\\n## วันที่หมดอายุ: 18 เมษายน 2565\"}"
    ]

    # Test all samples
    for i, json_str in enumerate(json_text):
        print(f"\n=== Testing Sample {i+1} ===")
        try:
            data = json.loads(json_str)
            text_key = next((k for k in ["Text", "text", "natural_text"] if k in data), None)
            
            if text_key:
                formatted = parse_thai_id_card(data[text_key])
                print(json.dumps(formatted, ensure_ascii=False, indent=2))
            else:
                print("No valid text key found.")
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")