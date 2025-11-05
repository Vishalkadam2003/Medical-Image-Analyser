from flask import current_app as app
from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from .models import analyze_xray_model, analyze_mri_model

import io
import base64
from PIL import Image

from .utils import (
    ensure_dirs,
    create_database,
    save_xray_record,
    save_mri_record,    
    get_patient_list,
    load_patient_details,
    load_mri_patient_details,
    generate_pdf,
    generate_mri_pdf,
    save_feedback,
    export_csv
)

# Ensure required folders and DB exist before first request
ensure_dirs()
create_database()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_xray', methods=['GET', 'POST'])
def analyze_xray_route():
    if request.method == 'POST':
        # Get form fields
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        contact = request.form.get('contact')
        notes = request.form.get('notes', '')

        # Validate form inputs
        if not all([name, age, gender, contact]):
            flash("Please fill in all required fields.")
            return redirect(request.url)
        try:
            age = int(age)
        except ValueError:
            flash("Age must be a number.")
            return redirect(request.url)

        # Get uploaded file
        file = request.files.get('xray_image')
        if not file or file.filename == '':
            flash("No X-ray file uploaded")
            return redirect(request.url)

        # Open image with PIL
        try:
            image = Image.open(file.stream)
        except Exception as e:
            flash(f"Invalid image file: {e}")
            return redirect(request.url)

        # Call AI model function
        fig, prob_dict, pneumonia_result, top_disease = analyze_xray_model(image)

        # Save record in database or storage
        image_path = save_xray_record(name, age, gender, contact, image, pneumonia_result, top_disease, prob_dict, notes)

        # Convert Matplotlib figure to base64 string for HTML embedding
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Convert uploaded image to base64 string for HTML embedding
        buf_img = io.BytesIO()
        image.save(buf_img, format='PNG')
        buf_img.seek(0)
        uploaded_img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

        return render_template('xray_result.html',
                               name=name,
                               age=age,
                               gender=gender,
                               contact=contact,
                               pneumonia=pneumonia_result,
                               probabilities=prob_dict,
                               top_disease=top_disease,
                               image_path=image_path,
                               plot_img=plot_img_b64,
                               uploaded_img=uploaded_img_b64)

    return render_template('xray_form.html')


@app.route('/analyze_mri', methods=['GET', 'POST'])
def analyze_mri_route():
    if request.method == 'POST':
        # Get form fields
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        contact = request.form.get('contact')

        # Validate inputs
        if not all([name, age, gender, contact]):
            flash("Please fill in all required fields.")
            return redirect(request.url)
        try:
            age = int(age)
        except ValueError:
            flash("Age must be a number.")
            return redirect(request.url)

        # Get uploaded MRI file
        file = request.files.get('mri_image')
        if not file or file.filename == '':
            flash("No MRI file uploaded")
            return redirect(request.url)

        # Open image
        try:
            image = Image.open(file.stream)
        except Exception as e:
            flash(f"Invalid image file: {e}")
            return redirect(request.url)

        # Call AI model function
        fig, probs, top_prediction = analyze_mri_model(image)

        # Save record
        image_path = save_mri_record(name, age, gender, contact, image, top_prediction, probs)

        # Convert figure and image to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        buf_img = io.BytesIO()
        image.save(buf_img, format='PNG')
        buf_img.seek(0)
        uploaded_img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

        return render_template('mri_result.html',
                               name=name,
                               age=age,
                               gender=gender,
                               contact=contact,
                               tumor_probabilities=probs,
                               top_prediction=top_prediction,
                               plot_img=plot_img_b64,
                               uploaded_img=uploaded_img_b64,
                               image_path=image_path)

    return render_template('mri_form.html')


@app.route('/patients')
def patients():
    xray_patients = get_patient_list(table="patient_data")
    mri_patients = get_patient_list(table="mri_patient_data")
    return render_template('patients.html', xray_patients=xray_patients, mri_patients=mri_patients)


import markdown2

@app.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    markdown_text, image_path, probabilities, name, img_path = load_patient_details(patient_id)
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(markdown_text)
    image_path = '/static/new_background.jpg'  # pass this to template

    return render_template('patient_detail.html', 
                           markdown=html_content,  # now HTML
                           image_path=image_path,
                           probabilities=probabilities,
                           name=name,
                           patient_id=patient_id)

@app.route('/mri_patient/<int:patient_id>')
def mri_patient_detail(patient_id):
    markdown, image_path, probabilities, name, img_path = load_mri_patient_details(patient_id)
    return render_template('mri_patient_detail.html', markdown=markdown, image_path=image_path, probabilities=probabilities, name=name)





@app.route('/download_xray_pdf/<int:patient_id>')
def download_xray_pdf(patient_id):
    markdown, image_path, probabilities, name, img_path = load_patient_details(patient_id)
    patient_data = {
        'name': name,
        'age': 45,
        'gender': 'Male',
        'contact': '1234567890',
        'top_disease': 'Cardiomegaly',
        'pneumonia': 'No',
        'probabilities': probabilities
    }
    output_path = f'static/reports/xray_report_{patient_id}.pdf'
    
    # Generate PDF file (use the correct function from utils)
    generate_pdf(patient_id, patient_data, output_path)  # make sure function signature matches
    
    # Check if file created
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    else:
        abort(404, description="PDF report not found.")


@app.route('/download_mri_pdf/<int:patient_id>')
def download_mri_pdf(patient_id):
    markdown, image_path, probabilities, name, img_path = load_mri_patient_details(patient_id)
    patient_data = {
        'name': name,
        'age': 45,
        'gender': 'Male',
        'contact': '1234567890',
        'top_prediction': 'Glioma',
        'tumor_probabilities': probabilities
    }
    output_path = f'static/reports/mri_report_{patient_id}.pdf'
    
    generate_mri_pdf(patient_id, patient_data, output_path)
    
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    else:
        abort(404, description="PDF report not found.")

import pandas as pd
from flask import make_response

@app.route('/export_xray_csv')
def export_xray_csv():
    xray_patients = get_patient_list(table="patient_data")
    df = pd.DataFrame(xray_patients)
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=xray_patients.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

@app.route('/export_mri_csv')
def export_mri_csv():
    mri_patients = get_patient_list(table="mri_patient_data")
    df = pd.DataFrame(mri_patients)
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=mri_patients.csv"
    response.headers["Content-Type"] = "text/csv"
    return response
