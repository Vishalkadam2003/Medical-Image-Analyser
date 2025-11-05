import os
import sqlite3
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import numpy as np
import cv2
import uuid
import logging
import pandas as pd
import ast  # for safe string-to-dict conversion

def ensure_dirs():
    for folder in ["xray_images", "xray_reports", "xray_feedback", "reports"]:
        os.makedirs(folder, exist_ok=True)

def create_database():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patient_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        contact TEXT,
        image_path TEXT,
        pneumonia TEXT,
        timestamp TEXT,
        top_disease TEXT,
        probabilities TEXT,
        feedback TEXT DEFAULT NULL,
        notes TEXT DEFAULT NULL
    )
    """)
    conn.commit()
    conn.close()

def save_xray_record(name, age, gender, contact, image, pneumonia_result, top_disease, prob_dict, notes):
    safe_name = name.replace(' ', '_')
    relative_path = f"xray_images/{safe_name}_{contact}.png"
    full_path = os.path.join("app", "static", relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    image.save(full_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO patient_data (name, age, gender, contact, image_path, pneumonia, timestamp, top_disease, probabilities, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (name, age, gender, contact, relative_path, pneumonia_result, timestamp, top_disease, str(prob_dict), notes)
    )
    conn.commit()
    conn.close()

    logging.info(f"Saved X-ray record for {name} at {timestamp}")
    return relative_path

def save_mri_record(name, age, gender, contact, image, top_prediction, prob_dict):
    safe_name = name.replace(' ', '_')
    relative_path = f"mri_images/mri_{safe_name}_{contact}.png"
    full_path = os.path.join("app", "static", relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    image.save(full_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mri_patient_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            contact TEXT,
            image_path TEXT,
            top_prediction TEXT,
            probabilities TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()

    cursor.execute(
        "INSERT INTO mri_patient_data (name, age, gender, contact, image_path, top_prediction, probabilities, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (name, age, gender, contact, relative_path, top_prediction, str(prob_dict), timestamp)
    )
    conn.commit()
    conn.close()

    logging.info(f"Saved MRI record for {name} at {timestamp}")
    return relative_path

def get_patient_list(table="patient_data"):
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    query = f"SELECT id, name, timestamp FROM {table} ORDER BY id DESC"
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "timestamp": r[2]} for r in results]

def load_patient_details(patient_id):
    # patient_id is int directly, no split
    if not patient_id:
        return "", None, None, None, None
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, age, gender, contact, image_path, pneumonia, timestamp, top_disease, probabilities FROM patient_data WHERE id=?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        markdown = f"""### {row[0]}
**Age:** {row[1]}  
**Gender:** {row[2]}  
**Contact:** {row[3]}  
**Pneumonia Detected:** {row[5]}  
**Diagnosis Time:** {row[6]}  
**Most Probable Disease:** {row[7]}"""
        probabilities_dict = ast.literal_eval(row[8])
        return markdown, row[4], probabilities_dict, row[0], row[4]
    else:
        return "No data found", None, None, None, None

def load_mri_patient_details(patient_id):
    # patient_id is int directly
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, age, gender, contact, image_path, top_prediction, probabilities, timestamp 
        FROM mri_patient_data WHERE id=?
    """, (patient_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        markdown = f"""### {row[0]}
**Age:** {row[1]}  
**Gender:** {row[2]}  
**Contact:** {row[3]}  
**Most Probable Tumor:** {row[5]}  
**Report Date:** {row[7]}"""
        probabilities_dict = ast.literal_eval(row[6])  # Safe eval
        return markdown, row[4], probabilities_dict, row[0], row[4]
    return "No MRI patient data found", None, None, None, None

# (Other utility functions remain unchanged)


def generate_pdf(probabilities, image_path, patient_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Chest X-ray Report", ln=True, align="C")
    pdf.ln(10)

    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, age, gender, contact, feedback, notes FROM patient_data WHERE image_path=?", (image_path,))
    info = cursor.fetchone()
    conn.close()

    if info:
        pdf.multi_cell(0, 10, txt=f"Name: {info[0]}\nAge: {info[1]}\nGender: {info[2]}\nContact: {info[3]}")
        pdf.ln(5)

    for k, v in ast.literal_eval(probabilities).items():  # safer
        pdf.cell(0, 10, txt=f"{k}: {v:.2f}", ln=True)

    if info and info[4]:
        pdf.ln(5)
        pdf.multi_cell(0, 10, txt=f"Feedback: {info[4]}")
    if info and info[5]:
        pdf.ln(5)
        pdf.multi_cell(0, 10, txt=f"Doctor Notes: {info[5]}")

    if os.path.exists(image_path):
        pdf.add_page()
        pdf.image(image_path, x=10, y=30, w=180)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"xray_reports/{patient_name.replace(' ', '_')}_{timestamp}_report.pdf"
    pdf.output(file_path)
    return file_path

def generate_mri_pdf(name, age, gender, contact, probs, img, top):
    import tempfile

    os.makedirs("reports", exist_ok=True)

    def sanitize(text):
        return ''.join(c for c in text if ord(c) < 256)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="MRI Brain Tumor Diagnostic Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Name: {sanitize(name)}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {sanitize(str(age))}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {sanitize(gender)}", ln=True)
    pdf.cell(200, 10, txt=f"Contact: {sanitize(contact)}", ln=True)
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Most Probable Tumor Type: {sanitize(top)}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Tumor Probabilities:", ln=True)
    for k, v in probs.items():
        pdf.cell(200, 10, txt=f"{sanitize(k)}: {v:.2f}", ln=True)

    if isinstance(img, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            pdf.image(tmp.name, x=10, y=pdf.get_y() + 10, w=100)

    output_path = os.path.join("reports", f"{sanitize(name)}_mri_report.pdf")
    pdf.output(output_path)
    return output_path

def save_feedback(patient_name, feedback):
    with open(f"xray_feedback/{patient_name}_feedback.txt", "w") as f:
        f.write(feedback)

    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    existing_columns = [r[1] for r in cursor.execute("PRAGMA table_info(patient_data)").fetchall()]
    if "notes" not in existing_columns:
        cursor.execute("ALTER TABLE patient_data ADD COLUMN notes TEXT")
    if "feedback" not in existing_columns:
        cursor.execute("ALTER TABLE patient_data ADD COLUMN feedback TEXT")
    conn.commit()
    cursor.execute("UPDATE patient_data SET feedback=? WHERE name=?", (feedback, patient_name))
    conn.commit()
    conn.close()
    logging.info(f"Feedback saved for {patient_name}")
    return "Thank you for your feedback!"

def export_csv():
    conn = sqlite3.connect("patients.db")
    df = pd.read_sql_query("SELECT * FROM patient_data", conn)
    conn.close()
    filepath = "xray_reports/patient_data_export.csv"
    df.to_csv(filepath, index=False)
    return filepath

from fpdf import FPDF
import os

def generate_xray_pdf(patient_id, patient_data, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="X-Ray Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Name: {patient_data['name']}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {patient_data['age']}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {patient_data['gender']}", ln=True)
    pdf.cell(200, 10, txt=f"Contact: {patient_data['contact']}", ln=True)
    pdf.ln(5)

    pdf.multi_cell(200, 10, txt=f"Prediction: {patient_data['top_disease']}")
    pdf.multi_cell(200, 10, txt=f"Pneumonia Detected: {patient_data['pneumonia']}")

    pdf.ln(5)
    pdf.cell(200, 10, txt="Probabilities:", ln=True)
    for disease, prob in patient_data['probabilities'].items():
        pdf.cell(200, 10, txt=f"{disease}: {prob:.2f}", ln=True)

    pdf.output(output_path)

def generate_mri_pdf(patient_id, patient_data, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="MRI Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Name: {patient_data['name']}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {patient_data['age']}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {patient_data['gender']}", ln=True)
    pdf.cell(200, 10, txt=f"Contact: {patient_data['contact']}", ln=True)
    pdf.ln(5)

    pdf.multi_cell(200, 10, txt=f"Top Prediction: {patient_data['top_prediction']}")
    pdf.ln(5)
    pdf.cell(200, 10, txt="Probabilities:", ln=True)
    for tumor, prob in patient_data['tumor_probabilities'].items():
        pdf.cell(200, 10, txt=f"{tumor}: {prob:.2f}", ln=True)

    pdf.output(output_path)


