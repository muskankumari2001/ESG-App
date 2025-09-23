#!/usr/bin/env python3
"""
K-Electric Bill Downloader and OCR Parser
Automates downloading K-Electric bills and extracts data using OCR
"""

import os
import sys
import time
import glob
import re
import cv2
import numpy as np
import requests
import pandas as pd
from pathlib import Path

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# OCR and image processing imports
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
import fitz  # PyMuPDF


class KEBillDownloader:
    """Handles downloading K-Electric bills from the website"""

    def __init__(self, download_dir="downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.driver = None

    def setup_chrome_driver(self):
        """Configure and setup Chrome WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": str(self.download_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True
        })

        try:
            # Try to use ChromeDriverManager for automatic driver management
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            print(f"ChromeDriverManager failed: {e}")
            # Fallback to system chromedriver
            self.driver = webdriver.Chrome(options=chrome_options)

    def download_bill(self, account_number):
        account_number = '0' + account_number
        """Download K-Electric bill for given account number"""
        if not self.driver:
            self.setup_chrome_driver()

        url = "https://staging.ke.com.pk:24555/"
        self.driver.get(url)

        try:
            # Fill in Account Number
            acc_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "txtAccNo"))
            )
            acc_input.clear()
            acc_input.send_keys(account_number)

            # Get text-based CAPTCHA
            captcha_text = self.driver.find_element(By.ID, "lblCaptcha").text.strip()
            print(f"CAPTCHA Text: {captcha_text}")

            # Enter CAPTCHA
            captcha_input = self.driver.find_element(By.ID, "txtimgcode")
            captcha_input.clear()
            captcha_input.send_keys(captcha_text)

            # Click "View Bill" button
            view_button = self.driver.find_element(By.ID, "btnViewBill")
            view_button.click()

            # Wait for table to load
            table = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.XPATH, '//table[contains(@class, "table")]'))
            )
            print("📄 Table loaded...")

            # Wait for and click download button
            download_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//input[@type="button" and @value="Download"]'))
            )
            self.driver.execute_script("arguments[0].click();", download_button)
            print("✅ Download button clicked.")

            # Wait for download to complete
            print("📥 Waiting for download to complete...")
            time.sleep(20)

            return True

        except Exception as e:
            print(f"Error downloading bill: {e}")
            return False

    def get_latest_pdf(self):
        """Return the most recently modified PDF in download directory"""
        pdfs = list(self.download_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {self.download_dir}")
        return max(pdfs, key=lambda p: p.stat().st_mtime)



    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()


class BillOCRParser:
    def __init__(self):
        # Configure Tesseract for better number recognition
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-:/ '

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

    def enhanced_regex_search(self, pattern, text, field_name=""):
        """Enhanced regex search with multiple attempts, but preserve letters for TARIFF."""
        # Try original pattern
        match = re.search(pattern, text, re.I | re.M)
        if match:
            raw = match.group(1).strip()
            # If it's the TARIFF field, return the raw capture (with letters/slashes)
            if field_name.upper() == "TARIFF":
                return raw
            # Otherwise clean to numbers/dots/commas
            cleaned = self.clean_extracted_number(raw)
            if cleaned:
                return cleaned

        # Try a more flexible‑spacing version
        flexible = pattern.replace(r'\s+', r'\s*').replace(r'\s*', r'[\s\n]*')
        match = re.search(flexible, text, re.I | re.M)
        if match:
            raw = match.group(1).strip()
            if field_name.upper() == "TARIFF":
                return raw
            cleaned = self.clean_extracted_number(raw)
            if cleaned:
                return cleaned

        return ""

    def parse_pesco_bill(self, text):
        """Enhanced bill parsing for new KE bill format"""
        result = {
            "REFERENCE NO": "",
            "SANC. LOAD": "",
            "CNTC LOAD": "",
            "TARIFF": "",
            "BILL MONTH": "",
            "ACTIVE UNITS ON PEAK": "",
            "ACTIVE UNITS OFF PEAK": "",
            "REACTIVE UNITS ON PEAK": "",
            "REACTIVE UNITS OFF PEAK": "",
            "MDI ON PEAK": "",
            "MDI OFF PEAK": "",
            "BILL AMOUNT": "",
            "PF PENALTY": "",
            "OFF PEAK UNIT RATE OLD": "",
            "OFF PEAK UNIT RATE NEW": "",
            "ON PEAK UNIT RATE OLD": "",
            "ON PEAK UNIT RATE NEW": ""
        }

        # Updated patterns based on the new KE bill format
        patterns = {
            "REFERENCE NO": [
                r'Account\s*No[:\s\.]*([0-9]{10,15})',
                r'Consumer\s*No[:\s]*([0-9A-Z]{6,15})',
                r'Invoice\s*No[:\s\.]*([0-9]{10,15})'
            ],
            "SANC. LOAD": [
                r'Sanc\s*Load[:\s]*([0-9]+(?:\.[0-9]+)?)',
                r'Sanctioned\s*Load[:\s]*([0-9]+(?:\.[0-9]+)?)'
            ],
            "CNTC LOAD": [
                r'Conn\s*Load[:\s]*([0-9]+(?:\.[0-9]+)?)',
                r'Connected\s*Load[:\s]*([0-9]+(?:\.[0-9]+)?)'
            ],
            "TARIFF": [
                r'Tariff[:\s]*([A-Za-z0-9\-]+)',
                r'TARIFF[:\s]*([A-Z0-9\-]+)'
            ],
            # Updated patterns for unit consumption from meter reading table
            "ACTIVE UNITS ON PEAK": [
                # Match the "Energy - Peak" row and capture the Units (KWh) column (3rd column)
                r'Energy\s*-\s*Peak\s+[0-9,\.]+\s+[0-9,\.]+\s+([0-9,]+)',
                r'Peak\s+Energy\s+[0-9,\.]+\s+[0-9,\.]+\s+([0-9,]+)',
                # Alternative patterns for different formats
                r'Energy\s*-?\s*Peak[^\d]*(?:[0-9,\.]+\s+){2}([0-9,]+)',
                r'(?:Energy.*?Peak|Peak.*?Energy).*?(\d{3,})\s+\d{1,2}(?:\s|$)',
            ],
            "ACTIVE UNITS OFF PEAK": [
                # Match the "Energy - Off Peak" row and capture the Units (KWh) column (3rd column)
                r'Energy\s*-\s*Off\s*Peak\s+[0-9,\.]+\s+[0-9,\.]+\s+([0-9,]+)',
                r'Off\s*Peak\s*Energy\s+[0-9,\.]+\s+[0-9,\.]+\s+([0-9,]+)',
                # Alternative patterns for different formats
                r'Energy\s*-?\s*Off\s*Peak[^\d]*(?:[0-9,\.]+\s+){2}([0-9,]+)',
                r'(?:Energy.*?Off.*?Peak|Off.*?Peak.*?Energy).*?(\d{4,})\s+\d{1,2}(?:\s|$)',
            ],
            "REACTIVE UNITS ON PEAK": [
                # Match the "Reactive Energy On Peak" row and capture the Units column (3rd column)
                r'Reactive\s*Energy\s*On\s*Peak\s+[0-9,\.]+\s+[0-9,\.]+\s+([0-9,]+)',
                r'KVARH\s*On\s*Peak\s+[0-9,\.]+\s+[0-9,\.]+\s+([0-9,]+)',
                # Alternative patterns
                r'Reactive.*?Energy.*?On.*?Peak[^\d]*(?:[0-9,\.]+\s+){2}([0-9,]+)',
                r'(?:Reactive.*?On.*?Peak|On.*?Peak.*?Reactive).*?(\d{3,})(?:\s|$)',
            ],
            "REACTIVE UNITS OFF PEAK": [
                # Match the "Reactive Energy Off Peak" row and capture the Units column (3rd column)
                r'Reactive\s*Energy\s*Off\s*Peak\s+[0-9,\.]+\s+[0-9,\.]+\s+([0-9,]+)',
                r'KVARH\s*Off\s*Peak\s+[0-9,\.]+\s+[0-9,\.]+\s+([0-9,]+)',
                # Alternative patterns
                r'Reactive.*?Energy.*?Off.*?Peak[^\d]*(?:[0-9,\.]+\s+){2}([0-9,]+)',
                r'(?:Reactive.*?Off.*?Peak|Off.*?Peak.*?Reactive).*?(\d{4,})(?:\s|$)',
            ],
            "MDI OFF PEAK": [
                # match the "Energy – Off Peak" line, skip 3 numbers, capture the 4th
                r'Energy[\s\-–]*Off\s*Peak[:\s]*[0-9\.,]+\s+[0-9\.,]+\s+[0-9\.,]+\s+([0-9\.,]+)',
                r'Off\s*Peak\s*Energy[:\s]*[0-9\.,]+\s+[0-9\.,]+\s+[0-9\.,]+\s+([0-9\.,]+)',
                r'Off\s*Peak[:\s]*[0-9\.,]+\s+[0-9\.,]+\s+[0-9\.,]+\s+([0-9\.,]+)'
            ],
            "MDI ON PEAK": [
                # match the "Energy – Peak" line, skip 3 numbers, capture the 4th
                r'Energy[\s\-–]*Peak[:\s]*[0-9\.,]+\s+[0-9\.,]+\s+[0-9\.,]+\s+([0-9\.,]+)',
                r'Peak\s*Energy[:\s]*[0-9\.,]+\s+[0-9\.,]+\s+[0-9\.,]+\s+([0-9\.,]+)',
                r'Peak[:\s]*[0-9\.,]+\s+[0-9\.,]+\s+[0-9\.,]+\s+([0-9\.,]+)'
            ],

            "BILL AMOUNT": [
                r'Your\s*Electricity\s*Charges\s*for\s*the\s*Period[:\s]*([0-9,]+\.[0-9]{2})',
                r'Amount\s*Payable[:\s]*([0-9,]+)',
                r'Total\s*Amount[:\s]*([0-9,]+)',
                r'Outstanding\s*Balance[:\s]*([0-9,]+\.[0-9]{2})',
                r'Amount\s*Payable\s*within\s*Due\s*Date[:\s]*([0-9,]+)',
            ],
            "AMOUNT PAYABLE WITHIN DUE DATE": [
                r'Amount\s*Payable\s*within\s*Due\s*Date[:\s]*([0-9,]+)',
                r'Till[:\s]*[0-9\-A-Za-z]+\s*Rs\.([0-9,]+)'
            ],
            "PF PENALTY": [
                r'PF\s*Penalty[:\s]*([0-9.,]+)',
                r'Power\s*Factor\s*Penalty[:\s]*([0-9.,]+)'
            ],
            # Updated patterns for unit rates from Variable Charges section
            "OFF PEAK UNIT RATE OLD": [
                # Match "Variable Off Peak" line and extract the rate (2nd column after units)
                r'Variable\s*Off\s*Peak\s+[0-9,\.]+\s+([0-9\.]+)',
                r'Off\s*Peak\s*\(Old\)\s+[0-9\.,]+\s+([0-9\.]+)',
                r'Off\s*Peak\s*\(\s*Old\s*\)\s+[0-9\.,]+\s+([0-9\.]+)',
                # Alternative patterns
                r'Variable.*?Off.*?Peak.*?([0-9]{1,2}\.[0-9]{4})',
            ],
            "OFF PEAK UNIT RATE NEW": [
                # Match "Off Peak (New Rates)" line and extract the rate
                r'Off\s*Peak\s*\(New\s*Rates\)\s+[0-9\.,]+\s+([0-9\.]+)',
                r'Off\s*Peak\s*\(\s*New\s*Rates\s*\)\s+[0-9\.,]+\s+([0-9\.]+)',
                # Alternative patterns
                r'Variable Charges.*?Off\s*Peak\s*\(New\s*Rates\).*?([0-9\.]+)',
                r'Off\s*Peak.*?New.*?Rates.*?([0-9\.]+)'
            ],
            "ON PEAK UNIT RATE OLD": [
                # Match "Variable Peak" line and extract the rate (2nd column after units)
                r'Variable\s*Peak\s+[0-9,\.]+\s+([0-9\.]+)',
                r'Peak\s*\(Old\)\s+[0-9\.,]+\s+([0-9\.]+)',
                r'Peak\s*\(\s*Old\s*\)\s+[0-9\.,]+\s+([0-9\.]+)',
                # Alternative patterns
                r'Variable.*?Peak.*?([0-9]{1,2}\.[0-9]{4})',
            ],
            "ON PEAK UNIT RATE NEW": [
                # Match "Peak (New Rates)" line and extract the rate
                r'Peak\s*\(New\s*Rates\)\s+[0-9\.,]+\s+([0-9\.]+)',
                r'Peak\s*\(\s*New\s*Rates\s*\)\s+[0-9\.,]+\s+([0-9\.]+)',
                # Alternative patterns
                r'Variable Charges.*?Peak\s*\(New\s*Rates\).*?([0-9\.]+)',
                r'Peak.*?New.*?Rates.*?([0-9\.]+)'
            ],
        }

        # Special handling for BILL MONTH from Reading Date
        reading_date_patterns = [
            r"Reading\s*Date[:\s]*(\d{1,2})\-([A-Za-z]+)\-(\d{2,4})",
            r"Date[:\s]*(\d{1,2})\-([A-Za-z]+)\-(\d{2,4})",
            r"Issue\s*Date[:\s]*(\d{1,2})\-([A-Za-z]+)\-(\d{2,4})"
        ]

        for pattern in reading_date_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                day, month, year = match.groups()
                if len(year) == 2:
                    year = '20' + year
                result["BILL MONTH"] = f"{month[:3].capitalize()}-{year}"
                break

        if not result["BILL MONTH"]:
            # Try alternative date format like "Jul-25"
            date_match = re.search(r': ([A-Za-z]{3}\-\d{2}) :', text)
            if date_match:
                result["BILL MONTH"] = date_match.group(1)
            else:
                result["BILL MONTH"] = "NOT FOUND"

        # Apply enhanced regex search for each field
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                value = self.enhanced_regex_search(pattern, text, field)
                if value:
                    result[field] = value
                    break

        # Post-processing for specific fields based on the document structure
        # Remove commas from numeric values for consistency
        for field in ["ACTIVE UNITS ON PEAK", "ACTIVE UNITS OFF PEAK", "REACTIVE UNITS ON PEAK", "REACTIVE UNITS OFF PEAK", "MDI ON PEAK", "MDI OFF PEAK"]:
            if result[field]:
                result[field] = result[field].replace(",", "")

        return result

    def process_bill(self, pdf_path):
        """Main processing function"""
        print(f"Processing bill: {pdf_path}")

        # Try to extract text directly from PDF first
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if pdf_text and len(pdf_text.strip()) > 100:
            print("Using direct PDF text extraction")
            return self.parse_pesco_bill(pdf_text)

        # Fallback to OCR
        print("Using OCR extraction")
        images = self.pdf_to_images(pdf_path)
        all_text = ""

        for i, img in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            text = self.extract_text_with_multiple_methods(img)
            all_text += text + "\n"

        return self.parse_pesco_bill(all_text)



def save_to_excel(result, output_path="extracted_bill_data.xlsx"):
    """Save extracted data to Excel with proper formatting"""
    # Build a pandas DataFrame with a two-row header
    cols = pd.MultiIndex.from_tuples([
        ("Sanctioned Load (KW)", ""),
        ("Connected Load (KW)", ""),
        ("Tariff", ""),
        ("Bill Month", ""),
        ("Active Units (KWH)", "On Peak"),
        ("Active Units (KWH)", "Off Peak"),
        ("Reactive Units (KVARH)", "On Peak"),
        ("Reactive Units (KVARH)", "Off Peak"),
        ("MDI (KW)", "On Peak"),
        ("MDI (KW)", "Off Peak"),
        ("Bill Amount (Rs)", ""),
        ("PF Penalty (Rs)", ""),
        ("Off Peak Unit Rate Old (Rs)", ""),
        ("On Peak Unit Rate Old (Rs)", ""),
        ("Off Peak Unit Rate New (Rs)", ""),
        ("On Peak Unit Rate New (Rs)", ""),
    ])

    # Map parser keys into that structure
    row = {
        ("Sanctioned Load (KW)", ""): result["SANC. LOAD"],
        ("Connected Load (KW)", ""): result["CNTC LOAD"],
        ("Tariff", ""): result["TARIFF"],
        ("Bill Month", ""): result["BILL MONTH"],
        ("Active Units (KWH)", "On Peak"): result["ACTIVE UNITS ON PEAK"],
        ("Active Units (KWH)", "Off Peak"): result["ACTIVE UNITS OFF PEAK"],
        ("Reactive Units (KVARH)", "On Peak"): result["REACTIVE UNITS ON PEAK"],
        ("Reactive Units (KVARH)", "Off Peak"): result["REACTIVE UNITS OFF PEAK"],
        ("MDI (KW)", "On Peak"): result.get("MDI ON PEAK", ""),
        ("MDI (KW)", "Off Peak"): result.get("MDI OFF PEAK", ""),
        ("Bill Amount (Rs)", ""): result["BILL AMOUNT"],
        ("PF Penalty (Rs)", ""): result["PF PENALTY"],
        ("Off Peak Unit Rate Old (Rs)", ""): result["OFF PEAK UNIT RATE OLD"],
        ("On Peak Unit Rate Old (Rs)", ""): result["ON PEAK UNIT RATE OLD"],
        ("Off Peak Unit Rate New (Rs)", ""): result["OFF PEAK UNIT RATE NEW"],
        ("On Peak Unit Rate New (Rs)", ""): result["ON PEAK UNIT RATE NEW"],
    }

    df = pd.DataFrame([row], columns=cols)
    df.to_excel(output_path)
    print(f"Saved extracted data to {output_path}")


def process_k_electric_bills(master_excel_path, download_dir="downloads"):
    """
    Process K-Electric bills by reading account numbers from Excel and extracting bill data.

    Args:
        master_excel_path (str): Path to Excel file containing account numbers
        download_dir (str): Directory to save downloaded bills (default: 'downloads')

    Returns:
        tuple: (success_count, total_count, results_df, error_log)
            - success_count: Number of successfully processed bills
            - total_count: Total number of account numbers processed
            - results_df: DataFrame containing all extracted bill data
            - error_log: List of errors encountered during processing

    Expected Excel format:
        The Excel file should have a column named 'Account Number' or 'account_number'
        containing K-Electric account numbers to process.
    """

    # Initialize components
    downloader = KEBillDownloader(download_dir)
    parser_ocr = BillOCRParser()

    # Track results and errors
    all_results = []
    error_log = []
    success_count = 0

    try:
        # Read account numbers from Excel
        try:
            df = pd.read_excel(master_excel_path)
            print(f"Loaded Excel file: {master_excel_path}")
            print(f"Available columns: {list(df.columns)}")

            # Find account number column (flexible naming)
            account_col = None
            for col in df.columns:
                if 'account' in col.lower() and 'number' in col.lower():
                    account_col = col
                    break

            if account_col is None:
                # Try common variations
                for col in df.columns:
                    if col.lower() in ['account_number', 'account', 'acc_no', 'consumer_no']:
                        account_col = col
                        break

            if account_col is None:
                raise ValueError(
                    "Could not find account number column. Expected columns like 'Account Number', 'account_number', etc.")

            account_numbers = df[account_col].dropna().astype(str).tolist()
            print(f"Found {len(account_numbers)} account numbers to process")

        except Exception as e:
            error_msg = f"Error reading Excel file: {str(e)}"
            print(error_msg)
            error_log.append(error_msg)
            return pd.DataFrame()

        # Process each account number
        for i, account_number in enumerate(account_numbers, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing {i}/{len(account_numbers)}: Account {account_number}")
            print('=' * 60)

            try:
                # Download bill
                print(f"Downloading bill for account: {account_number}")
                success = downloader.download_bill(account_number)

                if not success:
                    error_msg = f"Failed to download bill for account {account_number}"
                    print(error_msg)
                    error_log.append(error_msg)
                    continue

                # Get the latest PDF
                try:
                    latest_pdf = downloader.get_latest_pdf()
                    print(f"Processing PDF: {latest_pdf}")
                except FileNotFoundError as e:
                    error_msg = f"No PDF found for account {account_number}: {str(e)}"
                    print(error_msg)
                    error_log.append(error_msg)
                    continue

                # Parse the bill
                result = parser_ocr.process_bill(str(latest_pdf))

                # Add account number to result
                result["ACCOUNT_NUMBER"] = account_number
                result["PDF_PATH"] = str(latest_pdf)
                result["PROCESSING_STATUS"] = "SUCCESS"

                all_results.append(result)
                success_count += 1

                # Print extracted data
                print("\nExtracted data:")
                for key, value in result.items():
                    if key not in ["PDF_PATH", "PROCESSING_STATUS"]:
                        print(f"  {key:25}: {value}")

                print(f"✅ Successfully processed account {account_number}")

            except Exception as e:
                error_msg = f"Error processing account {account_number}: {str(e)}"
                print(error_msg)
                error_log.append(error_msg)

                # Add failed result
                failed_result = {
                    "ACCOUNT_NUMBER": account_number,
                    "PROCESSING_STATUS": "FAILED",
                    "ERROR": str(e)
                }
                all_results.append(failed_result)

        # Create results DataFrame
        if all_results:
            results_df = pd.DataFrame(all_results)

            # Save consolidated results
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"k_electric_bills_processed.xlsx"

            # Create a properly formatted output
            try:
                # Separate successful and failed results
                successful_results = [r for r in all_results if r.get("PROCESSING_STATUS") == "SUCCESS"]
                failed_results = [r for r in all_results if r.get("PROCESSING_STATUS") == "FAILED"]

                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Write successful results with proper formatting
                    if successful_results:
                        formatted_data = []
                        for result in successful_results:
                            formatted_row = {
                                "Account Number": result.get("REFERENCE NO", ""),
                                "Sanctioned Load (KW)": result.get("SANC. LOAD", ""),
                                "Connected Load (KW)": result.get("CNTC LOAD", ""),
                                "Tariff": result.get("TARIFF", ""),
                                "Bill Month": result.get("BILL MONTH", ""),
                                "Active Units On Peak (KWH)": result.get("ACTIVE UNITS ON PEAK", ""),
                                "Active Units Off Peak (KWH)": result.get("ACTIVE UNITS OFF PEAK", ""),
                                "Reactive Units On Peak (KVARH)": result.get("REACTIVE UNITS ON PEAK", ""),
                                "Reactive Units Off Peak (KVARH)": result.get("REACTIVE UNITS OFF PEAK", ""),
                                "MDI On Peak (KW)": result.get("MDI ON PEAK", ""),
                                "MDI Off Peak (KW)": result.get("MDI OFF PEAK", ""),
                                "Bill Amount (Rs)": result.get("BILL AMOUNT", ""),
                                "PF Penalty (Rs)": result.get("PF PENALTY", ""),
  				                "Off Peak Unit Rate Old (Rs)": result.get("OFF PEAK UNIT RATE OLD", ""),
				                "On Peak Unit Rate Old (Rs)": result.get("ON PEAK UNIT RATE OLD", ""),
				                "Off Peak Unit Rate New (Rs)": result.get("OFF PEAK UNIT RATE NEW", ""),
				                "On Peak Unit Rate New (Rs)": result.get("ON PEAK UNIT RATE NEW", ""),
                                "PDF Path": result.get("PDF_PATH", "")
                            }
                            formatted_data.append(formatted_row)

                        success_df = pd.DataFrame(formatted_data)
                        success_df.to_excel(writer, sheet_name='Successful_Extractions', index=False)

                    # Write failed results
                    if failed_results:
                        failed_df = pd.DataFrame(failed_results)
                        failed_df.to_excel(writer, sheet_name='Failed_Extractions', index=False)

                    # Write summary
                    summary_data = {
                        "Metric": ["Total Accounts", "Successful Extractions", "Failed Extractions", "Success Rate"],
                        "Value": [len(account_numbers), success_count, len(account_numbers) - success_count,
                                  f"{(success_count / len(account_numbers) * 100):.1f}%"]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

                print(f"\n📊 Results saved to: {output_path}")

            except Exception as e:
                print(f"Warning: Could not create formatted Excel output: {e}")
                # Fallback to simple CSV
                results_df.to_csv(f"k_electric_bills_processed_{timestamp}.csv", index=False)
                print(f"Results saved to CSV instead: k_electric_bills_processed_{timestamp}.csv")

        else:
            results_df = pd.DataFrame()

        # Print summary
        print(f"\n{'=' * 60}")
        print("PROCESSING SUMMARY")
        print('=' * 60)
        print(f"Total accounts processed: {len(account_numbers)}")
        print(f"Successful extractions: {success_count}")
        print(f"Failed extractions: {len(account_numbers) - success_count}")
        print(f"Success rate: {(success_count / len(account_numbers) * 100):.1f}%")

        if error_log:
            print(f"\nErrors encountered: {len(error_log)}")
            for error in error_log:
                print(f"  - {error}")

        return results_df

    finally:
        # Clean up
        downloader.close()


def main():
    """Main function to run the K-Electric bill processing"""
    import argparse

    parser = argparse.ArgumentParser(description='K-Electric Bill Downloader and Parser')
    parser.add_argument('account_number', help='K-Electric account number')
    parser.add_argument('--download-dir', default='downloads', help='Directory to save downloaded bills')
    parser.add_argument('--output', default='extracted_bill_data.xlsx', help='Output Excel file')
    parser.add_argument('--skip-download', action='store_true', help='Skip download and process existing PDF')

    args = parser.parse_args()

    # Initialize components
    downloader = KEBillDownloader(args.download_dir)
    parser_ocr = BillOCRParser()

    try:
        # Download bill if not skipping
        if not args.skip_download:
            print(f"Downloading bill for account: {args.account_number}")
            success = downloader.download_bill(args.account_number)
            if not success:
                print("Failed to download bill")
                return

        # Get the latest PDF
        try:
            latest_pdf = downloader.get_latest_pdf()
            print(f"Processing latest PDF: {latest_pdf}")
        except FileNotFoundError as e:
            print(e)
            return

        # Parse the bill
        result = parser_ocr.process_bill(str(latest_pdf))

        # Print results
        print("\n" + "=" * 50)
        print("EXTRACTED BILL DATA")
        print("=" * 50)

        for key, value in result.items():
            print(f"{key:25}: {value}")

        # Save to Excel
        save_to_excel(result, args.output)

    finally:
        # Clean up
        downloader.close()


if __name__ == "__main__":
    main()