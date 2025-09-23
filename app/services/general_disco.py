import os
import time
import shutil
import base64
import pandas as pd
import re
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF for better PDF text extraction


class BillOCRParser:
    def __init__(self):
        # Configure Tesseract for better number recognition
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-:/ ()'

    def pdf_to_images(self, pdf_path, dpi=300):
        """Convert PDF to high-quality images"""
        return convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=3)

    def extract_text_from_pdf(self, pdf_path):
        """Try to extract text directly from PDF first (better for digital PDFs)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text if text.strip() else None
        except:
            return None

    def advanced_preprocess_image(self, img):
        """Advanced image preprocessing for better OCR"""
        # Convert PIL to OpenCV format
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Convert back to PIL
        return Image.fromarray(cleaned)

    def extract_text_with_multiple_methods(self, img):
        """Try multiple OCR approaches and combine results"""
        texts = []

        # Method 1: Original preprocessing
        preprocessed1 = self.preprocess_image(img)
        text1 = pytesseract.image_to_string(preprocessed1, config=self.tesseract_config)
        texts.append(text1)

        # Method 2: Advanced preprocessing
        preprocessed2 = self.advanced_preprocess_image(img)
        text2 = pytesseract.image_to_string(preprocessed2, config=self.tesseract_config)
        texts.append(text2)

        # Method 3: Enhanced contrast
        enhancer = ImageEnhance.Contrast(img)
        enhanced = enhancer.enhance(2.0)
        preprocessed3 = self.preprocess_image(enhanced)
        text3 = pytesseract.image_to_string(preprocessed3, config=self.tesseract_config)
        texts.append(text3)

        # Method 4: Different PSM mode
        text4 = pytesseract.image_to_string(preprocessed1, config=r'--oem 3 --psm 4')
        texts.append(text4)

        # Combine all texts
        combined_text = '\n'.join(texts)
        return combined_text

    def preprocess_image(self, img):
        """Original preprocessing method"""
        gray = img.convert('L')
        # More aggressive binarization
        bw = gray.point(lambda x: 0 if x < 140 else 255, '1')
        return bw

    def clean_extracted_number(self, text):
        """Clean and validate extracted numbers"""
        if not text:
            return ""

        # Remove common OCR errors
        text = text.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        text = text.replace('S', '5').replace('s', '5').replace('Z', '2').replace('z', '2')

        # Extract only numbers, dots, and commas
        cleaned = re.sub(r'[^\d.,]', '', text)
        return cleaned

    def clean_tariff_text(self, text):
        """Clean tariff text while preserving alphanumeric characters, slashes, parentheses"""
        if not text:
            return ""

        # Remove extra spaces and clean up
        text = re.sub(r'\s+', ' ', text.strip())

        # Common OCR corrections for tariff codes
        text = text.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        text = text.replace('S', '5').replace('s', '5')

        # Keep alphanumeric, slashes, parentheses, hyphens
        cleaned = re.sub(r'[^A-Za-z0-9()\-/]', '', text)
        return cleaned

    def extract_bill_month_from_date(self, date_text):
        """Extract bill month from various date formats"""
        if not date_text:
            return ""

        # Month mapping
        month_map = {
            'jan': 'JAN', 'january': 'JAN',
            'feb': 'FEB', 'february': 'FEB',
            'mar': 'MAR', 'march': 'MAR',
            'apr': 'APR', 'april': 'APR',
            'may': 'MAY',
            'jun': 'JUN', 'june': 'JUN',
            'jul': 'JUL', 'july': 'JUL',
            'aug': 'AUG', 'august': 'AUG',
            'sep': 'SEP', 'september': 'SEP',
            'oct': 'OCT', 'october': 'OCT',
            'nov': 'NOV', 'november': 'NOV',
            'dec': 'DEC', 'december': 'DEC'
        }

        date_text = date_text.lower().strip()

        # Try to extract month and year
        # Pattern 1: JUL 25, JUL-25, JUL/25
        match = re.search(r'([a-z]{3,9})[\s\-/]*(\d{2,4})', date_text)
        if match:
            month_str, year = match.groups()
            if month_str in month_map:
                if len(year) == 2:
                    year = '20' + year
                return f"{month_map[month_str]}-{year}"

        # Pattern 2: 25-JUL-2024, 25/JUL/2024
        match = re.search(r'(\d{1,2})[\s\-/]*([a-z]{3,9})[\s\-/]*(\d{2,4})', date_text)
        if match:
            day, month_str, year = match.groups()
            if month_str in month_map:
                if len(year) == 2:
                    year = '20' + year
                return f"{month_map[month_str]}-{year}"

        return ""

    def read_sanc_load(self, text):
        """Dedicated function to extract SANC.LOAD (Sanctioned Load) from bill text"""

        def clean_load_value(value):
            """Clean and validate load value"""
            if not value:
                return ""

            # Remove common OCR errors
            value = value.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
            value = value.replace('S', '5').replace('s', '5').replace('Z', '2').replace('z', '2')

            # Extract only numbers
            cleaned = re.sub(r'[^\d]', '', value)
            return cleaned

        # Pattern 1: Direct SANC.LOAD field patterns
        sanc_load_patterns = [
            r'SANC\.?LOAD[:\s]*(\d+)',
            r'SANCTIONED\s*LOAD[:\s]*(\d+)',
            r'SANC\s*LOAD[:\s]*(\d+)',
            r'Sanctioned\s*Load[:\s]*(\d+)',
            r'SANC\.?\s*LD[:\s]*(\d+)',
        ]

        # Try direct patterns first
        for pattern in sanc_load_patterns:
            match = re.search(pattern, text, re.I | re.M)
            if match:
                value = clean_load_value(match.group(1))
                if value:
                    return value

        # Pattern 2: CNCT LOAD patterns (Connected Load - often same as Sanctioned)
        cnct_load_patterns = [
            r'CNCT\s*LOAD[:\s=]*(\d+)',
            r'Connected\s*Load[:\s]*(\d+)',
            r'Conn[\s\.]*Load[:\s]*(\d+)',
            r'CNCT\s*LOAD\s*=\s*(\d+)',
            r'Connection\s*Load[:\s]*(\d+)',
        ]

        for pattern in cnct_load_patterns:
            match = re.search(pattern, text, re.I | re.M)
            if match:
                value = clean_load_value(match.group(1))
                if value:
                    return value

        # Additional patterns for generic load mentions
        fallback_patterns = [
            r'Load[:\s]*(\d+)',
            r'KW[:\s]*(\d+)',
            r'Capacity[:\s]*(\d+)',
            r'(?:Load|KW|Capacity).*?(\d+)',
            r'(\d+).*?(?:KW|Load)',
        ]

        for pattern in fallback_patterns:
            matches = re.findall(pattern, text, re.I)
            if matches:
                for match in matches:
                    value = clean_load_value(match)
                    if value and 1 <= int(value) <= 10000:
                        return value

        return ""

    def enhanced_regex_search(self, pattern, text, field_name=""):
        """Enhanced regex search with field-specific processing"""
        # Try original pattern
        match = re.search(pattern, text, re.I | re.M | re.S)
        if match:
            raw = match.group(1).strip()

            # Field-specific processing
            if field_name.upper() in ["TARRIF", "TARIFF"]:
                return self.clean_tariff_text(raw)
            elif field_name.upper() == "BILL MONTH":
                return self.extract_bill_month_from_date(raw)
            else:
                # For numeric fields, clean to numbers/dots/commas
                cleaned = self.clean_extracted_number(raw)
                if cleaned:
                    return cleaned

        # Try a more flexible spacing version
        flexible = pattern.replace(r'\s+', r'\s*').replace(r'\s*', r'[\s\n]*')
        match = re.search(flexible, text, re.I | re.M | re.S)
        if match:
            raw = match.group(1).strip()
            if field_name.upper() in ["TARRIF", "TARIFF"]:
                return self.clean_tariff_text(raw)
            elif field_name.upper() == "BILL MONTH":
                return self.extract_bill_month_from_date(raw)
            else:
                cleaned = self.clean_extracted_number(raw)
                if cleaned:
                    return cleaned

        return ""

    def parse_disco_bill(self, text):
        """Enhanced bill parsing with improved patterns for DISCO bills"""
        result = {
            "REFERENCE NO": "",
            "SANC.LOAD": "",
            "CNCT LOAD": "",
            "TARRIF": "",
            "BILL MONTH": "",
            "KWH METER READING UNITS CONSUMED (P)": "",
            "KWH METER READING UNITS CONSUMED (O)": "",
            "KVARH METER READING (P)": "",
            "KVARH METER READING (O)": "",
            "PAYABLE WITHIN DUE DATE": "",
            "LPF PENALTY": "",
            "OFF Peak Unit Rate": "",
            "ON Peak Unit Rate": "",
            "MDI METER READING off peak O": "",
            "MDI METER READING on Peak P": ""
        }

        # Enhanced patterns based on the actual bill structure
        patterns = {
            "REFERENCE NO": [
                r'REFERENCE\s*NO[:\s]*(\d{2}\s+\d{5}\s+\d{7}\s*[A-Z]?)',
                r'(\d{2}\s+\d{5}\s+\d{7}\s*[A-Z]?)\s*U',
                r'(\d{2}\s+\d{5}\s+\d{7})',
                r'UNIQUE\s*KEY[:\s]*(\d+)',
                r'Account\s*No[:\s]*([0-9\s]+)'
            ],

            "SANC.LOAD": [
                r'SANC\.?LOAD[:\s]*(\d+)',
                r'SANCTIONED\s*LOAD[:\s]*(\d+)',
                r'SANC\s*LOAD[:\s]*(\d+)',
                r'([A-Z]\d+[a-z]*\(\d+\)[A-Z]+)\s+(\d+)',
                r'[A-Z]\d+[a-z]*\(\d+\)[A-Z]+\s+(\d+)',
                r'B\d+[a-z]*\(\d+\)[A-Z]+\s+(\d+)',
                r'Load[:\s]*(\d+)',
                r'Sanc[:\s]*(\d+)'
            ],

            "CNCT LOAD": [
                r'CNCT\s*LOAD[:\s=]*(\d+)',
                r'Connected\s*Load[:\s]*(\d+)',
                r'Conn[\s\.]*Load[:\s]*(\d+)',
                r'CNCT\s*LOAD\s*=\s*(\d+)',
            ],

            "TARRIF": [
                r'TARIFF[:\s]*([A-Z]\s*-?\s*\d*\s*[a-z]*\s*\(\s*\d+\s*\)\s*[A-Z]+)',
                r'TARRIF[:\s]*([A-Z]\s*-?\s*\d*\s*[a-z]*\s*\(\s*\d+\s*\)\s*[A-Z]+)',
                r'([A-Z]\s*\d+\s*[a-z]*\s*\(\s*\d+\s*\)\s*[A-Z]+)',
                r'([A-Z]\s*-\s*\d*\s*[a-z]*\s*\(\s*\d+\s*\)\s*[A-Z]+)',
                r'([A-Z]-?\d*[a-z]*\(\d+\)[A-Z]+)',
                r'([A-Z]\s+\d+\s+[a-z]+\s+\(\s*\d+\s*\)\s+[A-Z]+)',
                r'(?:REFERENCE|UNIQUE).*?([A-Z]\s*-?\s*\d*\s*[a-z]*\s*\(\s*\d+\s*\)\s*[A-Z]+)',
                r'(\w+\s*-?\s*\d*\s*[a-z]*\s*\(\s*\d+\s*\)\s*\w+)',
                r'([A-Z]\s*\d+\s*[a-z]*\s*\(\s*\d+\s*\)\s*[A-Z]+)',
                r'([A-Z][\s-]*\d*[\s]*[a-z]*[\s]*\([\s]*\d+[\s]*\)[\s]*[A-Z]+)'
            ],

            "BILL MONTH": [
                r'BILL\s*MONTH[:\s]*([A-Z]{3}\s*\d{2})',
                r'Bill\s*Month[:\s]*([A-Z]{3}\s*\d{2})',
                r'BILL\s*MONTH[:\s]*([A-Z]{3}[\s\-/]*\d{2,4})',
                r'(?:CONN\.DATE|BILL\s*MONTH).*?([A-Z]{3}\s+\d{2})',
                r'([A-Z]{3}\s+\d{2})(?:\s+\d{2}\s+[A-Z]{3})',
                r'([A-Z]{3}[\s\-/]+\d{2,4})',
                r'30\s+2\s+%\s+([A-Z]{3}\s+\d{2})',
                r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s\-/]*(\d{2,4})'
            ],

            "KWH METER READING UNITS CONSUMED (P)": [
                r'UNITS\s*CONSUMED.*?\(O\)\s*\d+.*?\(P\)\s*(\d+)',
                r'KWH.*?UNITS.*?CONSUMED.*?\(P\)\s*(\d+)',
                r'\(P\)\s*(\d+)(?:\s+\(O\)|\s+Off)',
                r'Peak.*?Units.*?(\d+)',
                r'On.*?Peak.*?(\d+)'
            ],

            "KWH METER READING UNITS CONSUMED (O)": [
                r'UNITS\s*CONSUMED.*?\(O\)\s*(\d+)',
                r'KWH.*?UNITS.*?CONSUMED.*?\(O\)\s*(\d+)',
                r'\(O\)\s*(\d+)(?:\s+\(P\)|\s+Peak)',
                r'Off.*?Peak.*?Units.*?(\d+)',
                r'Off.*?Peak.*?(\d+)'
            ],

            "KVARH METER READING (O)": [
                r'UNITS\s*CONSUMED\s*\(O\)\s*\d+\s*\(P\)\s*\d+\s*\(O\)\s*(\d+)',
                r'KWH.*?CONSUMED.*?\(O\)\s*\d+.*?\(P\)\s*\d+.*?\(O\)\s*(\d+)',
                r'KVARH.*?METER.*?READING.*?\(O\)\s*(\d+)',
                r'KVARH.*?\(O\)\s*(\d+)'
            ],

            "KVARH METER READING (P)": [
                r'UNITS\s*CONSUMED\s*\(O\)\s*\d+\s*\(P\)\s*\d+\s*\(O\)\s*\d+\s*\(P\)\s*(\d+)',
                r'KWH.*?CONSUMED.*?\(O\)\s*\d+.*?\(P\)\s*\d+.*?\(O\)\s*\d+.*?\(P\)\s*(\d+)',
                r'KVARH.*?METER.*?READING.*?\(P\)\s*(\d+)',
                r'KVARH.*?\(P\)\s*(\d+)'
            ],

            "PAYABLE WITHIN DUE DATE": [
                r'PAYABLE\s*WITHIN\s*DUE\s*DATE[:\s]*(\d+)',
                r'Amount\s*Payable[:\s]*(\d+)',
                r'PAYM[:\s]*(\d+)',
                r'DUE.*?(\d+)'
            ],

            "LPF PENALTY": [
                r'LPF\s*PENALTY[:\s]*(\d+)',
                r'PF\s*Penalty[:\s]*(\d+)',
                r'Power\s*Factor\s*Penalty[:\s]*(\d+)'
            ],

            "OFF Peak Unit Rate": [
                r'GOP\s*Tariff[:\s]*X[:\s]*Units[:\s]*([0-9.]+)',
                r'(\d+\.\d+)\s*X\s*\d+',
                r'Government\s*Tariff[:\s]*([0-9.,]+)'
            ],

            "ON Peak Unit Rate": [
                r'GOP\s*Tariff[:\s]*X[:\s]*Units\s*.*?\n\s*([0-9.]+)',
                r'Government\s*Tariff[:\s]*.*?\n\s*([0-9.]+)',
                r'(\d+\.\d+)\s*X\s*\d+'
            ],

            "MDI METER READING off peak O": [
                r'UNITS\s*CONSUMED\s*\(O\)\s*\d+\s*\(P\)\s*\d+\s*\(O\)\s*\d+\s*\(P\)\s*\d+\s*\(O\)\s*(\d+)',
                r'KWH.*?CONSUMED.*?\(O\)\s*\d+.*?\(P\)\s*\d+.*?\(O\)\s*\d+.*?\(P\)\s*\d+.*?\(O\)\s*(\d+)',
                r'MDI.*?METER.*?READING.*?\(O\)\s*(\d+)',
                r'MDI.*?\(O\)\s*(\d+)'
            ],

            "MDI METER READING on Peak P": [
                r'UNITS\s*CONSUMED\s*\(O\)\s*\d+\s*\(P\)\s*\d+\s*\(O\)\s*\d+\s*\(P\)\s*\d+\s*\(O\)\s*\d+\s*\(P\)\s*(\d+)',
                r'KWH.*?CONSUMED.*?\(O\)\s*\d+.*?\(P\)\s*\d+.*?\(O\)\s*\d+.*?\(P\)\s*\d+.*?\(O\)\s*\d+.*?\(P\)\s*(\d+)',
                r'MDI.*?METER.*?READING.*?\(P\)\s*(\d+)',
                r'MDI.*?\(P\)\s*(\d+)'
            ]
        }

        # Apply enhanced regex search for each field
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                try:
                    value = self.enhanced_regex_search(pattern, text, field)
                    if value:
                        result[field] = value
                        break
                except re.error as e:
                    continue

        # Special handling for SANC.LOAD using dedicated function
        if not result["SANC.LOAD"]:
            sanc_load_value = self.read_sanc_load(text)
            if sanc_load_value:
                result["SANC.LOAD"] = sanc_load_value

        # Fallback: If CNCT LOAD not found, use SANC.LOAD
        if not result["CNCT LOAD"] and result["SANC.LOAD"]:
            result["CNCT LOAD"] = result["SANC.LOAD"]

        # Additional fallback for BILL MONTH
        if not result["BILL MONTH"]:
            month_matches = re.findall(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s-]?(\d{2,4})\b', text,
                                       re.I)
            if month_matches:
                month, year = month_matches[-1]
                if len(year) == 2:
                    year = '20' + year
                result["BILL MONTH"] = f"{month.upper()}-{year}"

        if not result["BILL MONTH"]:
            result["BILL MONTH"] = "NOT FOUND"

        return result

    def process_bill(self, pdf_path):
        """Main processing function"""
        # Try to extract text directly from PDF first
        pdf_text = self.extract_text_from_pdf(pdf_path)

        if pdf_text and 'not found' in pdf_text.lower():
            # Set ALL fields to "BILL NOT FOUND"
            result_fields = [
                "REFERENCE NO", "SANC.LOAD", "CNCT LOAD", "TARRIF", "BILL MONTH",
                "KWH METER READING UNITS CONSUMED (P)", "KWH METER READING UNITS CONSUMED (O)",
                "KVARH METER READING (P)", "KVARH METER READING (O)", "PAYABLE WITHIN DUE DATE",
                "LPF PENALTY", "OFF Peak Unit Rate", "ON Peak Unit Rate",
                "MDI METER READING off peak O", "MDI METER READING on Peak P"
            ]
            return {field: "BILL NOT FOUND" for field in result_fields}

        if pdf_text and len(pdf_text.strip()) > 100:
            return self.parse_disco_bill(pdf_text)

        # Fallback to OCR
        images = self.pdf_to_images(pdf_path)
        all_text = ""

        for i, img in enumerate(images):
            text = self.extract_text_with_multiple_methods(img)
            all_text += text + "\n"

        return self.parse_disco_bill(all_text)


class IntegratedBillProcessor:
    def __init__(self, input_excel, output_dir, chromedriver_path, extracted_data_file="extracted_bills_data.xlsx"):
        self.input_excel = input_excel
        self.output_dir = output_dir
        self.chromedriver_path = chromedriver_path
        self.extracted_data_file = extracted_data_file

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize OCR parser
        self.ocr_parser = BillOCRParser()

        # DISCO URL map
        self.disco_url_map = {
            "QESCO": "https://bill.pitc.com.pk/qescobill/",
            "LESCO": "https://bill.pitc.com.pk/lescobill",
            "MEPCO": "https://bill.pitc.com.pk/mepcobill",
            "PESCO": "https://bill.pitc.com.pk/pescobill",
            "SEPCO": "https://bill.pitc.com.pk/sepcobill",
            "GEPCO": "https://bill.pitc.com.pk/gepcobill",
            "IESCO": "https://bill.pitc.com.pk/iescobill",
            "HESCO": "https://bill.pitc.com.pk/hescobill",
            "FESCO": "https://bill.pitc.com.pk/fescobill"
        }

    def try_get_bill(self, driver, url, acc_num, retries=3, wait_seconds=10):
        """Retry logic for bill fetching"""
        for attempt in range(retries):
            try:
                driver.get(url)
                acc_input = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "searchTextBox"))
                )
                acc_input.clear()
                acc_input.send_keys(acc_num)

                view_button = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "btnSearch"))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", view_button)
                time.sleep(1)
                driver.execute_script("arguments[0].click();", view_button)

                # Wait for bill info
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//*[contains(text(), 'Reference') or contains(text(), 'ref')]"))
                )

                time.sleep(2)  # Let the full page load
                return True
            except Exception as e:
                print(f"🔁 Retry {attempt + 1}/{retries} for {acc_num} failed: {e}")
                time.sleep(wait_seconds)
        return False

    def setup_webdriver(self):
        """Setup Chrome WebDriver with optimized settings"""
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)

        service = Service(executable_path=self.chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Avoid detection
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })

        return driver

    def download_bills(self):
        """Download all bills from Excel file"""
        print("📥 Starting bill download process...")

        # Load Excel file
        df = pd.read_excel(self.input_excel)

        # Setup WebDriver
        driver = self.setup_webdriver()

        downloaded_files = []

        try:
            # Process each account
            for index, row in df.iterrows():
                acc_num = str(row['Account Number']).strip()
                disco = str(row['DISCO']).strip().upper()

                if disco not in self.disco_url_map:
                    print(f"❌ Skipping unknown DISCO: {disco}")
                    continue

                url = self.disco_url_map[disco]
                print(f"🔄 Processing {disco} - {acc_num}")

                if self.try_get_bill(driver, url, acc_num):
                    try:
                        time.sleep(3)
                        result = driver.execute_cdp_cmd("Page.printToPDF", {
                            "printBackground": True,
                            "paperWidth": 8.27,  # A4 size
                            "paperHeight": 11.69
                        })

                        pdf_data = base64.b64decode(result['data'])
                        filename = f"{disco}_{acc_num}.pdf"
                        pdf_path = os.path.join(self.output_dir, filename)

                        with open(pdf_path, "wb") as f:
                            f.write(pdf_data)
                        print(f"✅ Saved PDF: {pdf_path}")
                        downloaded_files.append(pdf_path)
                    except Exception as e:
                        print(f"❌ Error printing PDF for {acc_num}: {e}")
                else:
                    print(f"❌ Failed to fetch bill for {acc_num} - {disco}")

        finally:
            driver.quit()

        print(f"📥 Download complete! {len(downloaded_files)} bills downloaded.")
        return downloaded_files

    def load_bills_from_folder(self):
        """Load all PDF bills from the bills folder"""
        pdf_files = []
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(self.output_dir, filename)
                    pdf_files.append(pdf_path)

        print(f"📁 Found {len(pdf_files)} PDF files in {self.output_dir}")
        return pdf_files

    def process_all_bills_ocr(self, pdf_files=None):
        """Process all bills with OCR and extract data"""
        if pdf_files is None:
            pdf_files = self.load_bills_from_folder()

        if not pdf_files:
            print("❌ No PDF files found to process!")
            return []

        print("🔍 Starting OCR processing for all bills...")

        extracted_data_list = []

        for i, pdf_path in enumerate(pdf_files, 1):
            filename = os.path.basename(pdf_path)
            print(f"🔍 Processing {i}/{len(pdf_files)}: {filename}")

            try:
                # Extract data using OCR
                extracted_data = self.ocr_parser.process_bill(pdf_path)

                # Add metadata
                extracted_data['FILE_NAME'] = filename
                extracted_data['FILE_PATH'] = pdf_path

                # Extract DISCO and Account Number from filename
                if '_' in filename:
                    parts = filename.replace('.pdf', '').split('_', 1)
                    if len(parts) == 2:
                        extracted_data['DISCO'] = parts[0]
                        extracted_data['ACCOUNT_NUMBER'] = parts[1]

                extracted_data_list.append(extracted_data)
                print(f"✅ Processed: {filename}")

                # Print key extracted data for verification
                print(f"   📋 Reference: {extracted_data.get('REFERENCE NO', 'N/A')}")
                print(f"   ⚡ Sanc Load: {extracted_data.get('SANC.LOAD', 'N/A')}")
                print(f"   📅 Bill Month: {extracted_data.get('BILL MONTH', 'N/A')}")
                print(f"   💰 Payable: {extracted_data.get('PAYABLE WITHIN DUE DATE', 'N/A')}")
                print()

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                # Add error entry to maintain record
                error_data = {field: "ERROR" for field in [
                    "REFERENCE NO", "SANC.LOAD", "CNCT LOAD", "TARRIF", "BILL MONTH",
                    "KWH METER READING UNITS CONSUMED (P)", "KWH METER READING UNITS CONSUMED (O)",
                    "KVARH METER READING (P)", "KVARH METER READING (O)", "PAYABLE WITHIN DUE DATE",
                    "LPF PENALTY", "OFF Peak Unit Rate", "ON Peak Unit Rate",
                    "MDI METER READING off peak O", "MDI METER READING on Peak P"
                ]}
                error_data['FILE_NAME'] = filename
                error_data['FILE_PATH'] = pdf_path
                error_data['ERROR_MESSAGE'] = str(e)

                if '_' in filename:
                    parts = filename.replace('.pdf', '').split('_', 1)
                    if len(parts) == 2:
                        error_data['DISCO'] = parts[0]
                        error_data['ACCOUNT_NUMBER'] = parts[1]

                extracted_data_list.append(error_data)

        print(f"🔍 OCR processing complete! Processed {len(extracted_data_list)} bills.")
        return extracted_data_list

    def save_extracted_data(self, extracted_data_list):
        """Save extracted data to Excel"""
        if not extracted_data_list:
            print("❌ No data to save!")
            return

        try:
            # Convert to DataFrame
            df = pd.DataFrame(extracted_data_list)

            # Reorder columns for better readability
            column_order = [
                'FILE_NAME', 'DISCO', 'ACCOUNT_NUMBER', 'REFERENCE NO', 'SANC.LOAD',
                'CNCT LOAD', 'TARRIF', 'BILL MONTH', 'PAYABLE WITHIN DUE DATE',
                'KWH METER READING UNITS CONSUMED (P)', 'KWH METER READING UNITS CONSUMED (O)',
                'KVARH METER READING (P)', 'KVARH METER READING (O)',
                'MDI METER READING off peak O', 'MDI METER READING on Peak P',
                'LPF PENALTY', 'OFF Peak Unit Rate', 'ON Peak Unit Rate',
                'FILE_PATH'
            ]

            # Add any missing columns and reorder
            for col in column_order:
                if col not in df.columns:
                    df[col] = ""

            # Add error message column if it exists
            if 'ERROR_MESSAGE' in df.columns:
                column_order.append('ERROR_MESSAGE')

            df = df[column_order]

            # Save to Excel
            output_file = os.path.join(os.path.dirname(self.output_dir), self.extracted_data_file)
            df.to_excel(output_file, index=False)

            print(f"💾 Extracted data saved to: {output_file}")

            # Print summary statistics
            total_bills = len(df)
            successful_extractions = len(df[df['REFERENCE NO'] != 'ERROR'])
            failed_extractions = total_bills - successful_extractions

            print("\n📊 EXTRACTION SUMMARY:")
            print(f"   Total Bills Processed: {total_bills}")
            print(f"   Successful Extractions: {successful_extractions}")
            print(f"   Failed Extractions: {failed_extractions}")
            print(f"   Success Rate: {(successful_extractions / total_bills * 100):.1f}%")

            return output_file

        except Exception as e:
            print(f"❌ Error saving extracted data: {str(e)}")
            return None

    def create_zip_archive(self):
        """Create zip archive of all PDFs"""
        try:
            zip_path = os.path.join(os.path.dirname(self.output_dir), "bills.zip")
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', self.output_dir)
            print(f"📦 Created zip archive: {zip_path}")
            return zip_path
        except Exception as e:
            print(f"❌ Error creating zip archive: {str(e)}")
            return None

    def run_complete_process(self, download_bills=True, process_ocr=True):
        """Run the complete bill processing workflow"""
        print("🚀 Starting Integrated Bill Processing...")
        print("=" * 60)

        downloaded_files = []

        # Step 1: Download bills (if requested)
        if download_bills:
            print("📥 STEP 1: Downloading Bills")
            print("-" * 30)
            downloaded_files = self.download_bills()

            # Create zip archive
#            if downloaded_files:
#                self.create_zip_archive()
        else:
            print("⏭️ STEP 1: Skipping bill download")

        # Step 2: Process with OCR (if requested)
        if process_ocr:
            print("\n🔍 STEP 2: OCR Processing")
            print("-" * 30)

            # Use downloaded files or load from folder
            pdf_files = downloaded_files if downloaded_files else None
            extracted_data_list = self.process_all_bills_ocr(pdf_files)

            # Step 3: Save extracted data
            print("\n💾 STEP 3: Saving Extracted Data")
            print("-" * 30)
            output_file = self.save_extracted_data(extracted_data_list)

            print("\n✅ PROCESS COMPLETE!")
            print("=" * 60)

            return {
                'downloaded_files': downloaded_files,
                'extracted_data': extracted_data_list,
                'output_file': output_file
            }
        else:
            print("⏭️ STEP 2: Skipping OCR processing")
            return {
                'downloaded_files': downloaded_files,
                'extracted_data': [],
                'output_file': None
            }

# --- [YOUR FULL CODE HERE, UNCHANGED] ---
# (Paste your existing code exactly as you provided.)

# ... keep everything exactly as it is, including the classes and the main block ...


# --- [Helper function for modular usage] ---
def process_disco_bills(master_excel_path, download_dir="downloads"):
    """
    Helper to run the full download and OCR process for general DISCO bills and return a DataFrame.

    Args:
        master_excel_path (str): Path to the input Excel file containing DISCO/account data.
        download_dir (str): Directory where PDFs will be saved.

    Returns:
        pd.DataFrame: OCR extracted data for all processed bills.
    """
    # You must set the chromedriver path (edit as appropriate for your environment)
    from webdriver_manager.chrome import ChromeDriverManager
    chromedriver_path = ChromeDriverManager().install()
    processor = IntegratedBillProcessor(
        input_excel=master_excel_path,
        output_dir=download_dir,
        chromedriver_path=chromedriver_path,
        #extracted_data_file="extracted_bills_data.xlsx"
    )
    results = processor.run_complete_process(
        download_bills=True,  # Set to False to skip download, True to download
        process_ocr=True
    )
    # The output Excel file path
    output_file = results.get('output_file')
    if output_file and os.path.exists(output_file):
        df = pd.read_excel(output_file)
        return df
    else:
        return pd.DataFrame()

# Usage example and main execution
if __name__ == "__main__":
    # Configuration
    input_excel = "C:\\Users\\bintern1\\Downloads\\DISCO_ACCOUNT_LIST.xlsx"
    output_dir = "C:\\Users\\bintern1\\Downloads\\bills"
    chromedriver_path = "C:\\Users\\bintern1\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"

    # Initialize processor
    processor = IntegratedBillProcessor(
        input_excel=input_excel,
        output_dir=output_dir,
        chromedriver_path=chromedriver_path,
        #extracted_data_file="extracted_bills_data.xlsx"
    )

    # Run complete process
    # Set download_bills=False if you want to skip downloading and only process existing PDFs
    results = processor.run_complete_process(
        download_bills=True,  # Set to False to skip download
        process_ocr=True  # Set to False to skip OCR processing
    )

    print(f"\n🎉 Processing completed!")
    if results['output_file']:
        print(f"📋 Extracted data available at: {results['output_file']}")

    # Optional: Process only existing bills without downloading
    # processor.run_complete_process(download_bills=False, process_ocr=True)