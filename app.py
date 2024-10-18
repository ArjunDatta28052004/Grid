import os
from flask import Flask, request, render_template, redirect, flash, url_for, session
from werkzeug.utils import secure_filename
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import easyocr
import re
import cv2

app = Flask(__name__)
app.secret_key = 'kpd'  # Change this to a random secret key
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load models and data
class_names_path = r'C:\Users\Arjun Datta PC\OneDrive\Desktop\GRID\class_names.pkl'  # Change this path accordingly
with open(class_names_path, 'rb') as f:
    class_names = pickle.load(f)

model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft.load_state_dict(torch.load(r'C:\Users\Arjun Datta PC\OneDrive\Desktop\GRID\resnet50_fruits_model.pth', map_location=torch.device('cpu')))
model_ft.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ft = model_ft.to(device)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model_ft(img_tensor)
        _, preds = torch.max(outputs, 1)
    
    return class_names[preds.item()]

def process_image(file_path, predict_type):
    product_details = {}
    if predict_type == 'expiry':
        text = extract_text_from_image(file_path)
        mfg_date, exp_date = extract_dates(text)
        if exp_date:
            priority = classify_priority(exp_date)
        else:
            priority = 'No Expiry Date Found'
        
        product_details = {
            'image': os.path.basename(file_path),
            'mfg_date': mfg_date,
            'exp_date': exp_date,
            'priority': priority,
        }
    elif predict_type == 'freshness':
        predicted_class = predict_image(file_path)
        product_details = {
            'image': os.path.basename(file_path),
            'prediction': predicted_class
        }
    elif predict_type == 'brand_detection':
        data = extract_data_from_image(file_path)
        product_details = {
            'image': os.path.basename(file_path),
            'MRP': data['MRP'],
            'Brand': data['Brand'],
            'Dimensions': data['Dimensions'],
            'Product Type': data['Product Type']
        }
    return product_details

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']  # Just an example
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        predict_type = request.form.get('predict_type')
        files = request.files.getlist('files[]')
        product_details = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                details = process_image(file_path, predict_type)
                product_details.append(details)

        return render_template('predictions.html', product_details=product_details)

    return render_template('upload.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")
    
    results = reader.readtext(image)
    extracted_text = " ".join([result[1] for result in results])
    return extracted_text

import re
from datetime import datetime

def extract_dates(text):
    # Define regular expressions for different date formats
    date_patterns = [
        r'\b\d{2}[-/]\d{2}[-/]\d{4}\b',            # DD-MM-YYYY, MM-DD-YYYY, or DD/MM/YYYY
        r'\b\d{4}[-/]\d{2}[-/]\d{2}\b',            # YYYY-MM-DD or YYYY/MM/DD
        r'\b\d{2}[-/]\d{2}[-/]\d{2}\b',            # DD-MM-YY or MM-DD-YY
        r'\b\d{1,2} [A-Za-z]{3} \d{4}\b',          # DD-MMM-YYYY
        r'\b[A-Za-z]{3} \d{1,2}, \d{4}\b',         # MMM DD, YYYY
        r'\b\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b',  # DD Month YYYY
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b',  # Month DD, YYYY
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)-\d{4}\b',  # Month-YYYY
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b',  # Month YYYY
    ]
    def find_dates(text, patterns):
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return dates
    combined_pattern = '|'.join(date_patterns)
    # Keywords to identify manufacturing and expiry dates
    manufacture_keywords = re.search(r'(manufacture|mfg|manufactured|mfd|packed on|produced on|production date|packing date|mfg date|MFG|Mfg|MFD|MFG|DATE OF MANUFACTURE|DATE OF ISSUE|Mfg|Ready|Made|Date|date).*?(' + combined_pattern + ')', text, re.IGNORECASE)
    expiry_keywords = re.search(r'(expiry|exp|expires|use by|best before|bb|exp date|expiring on|sell by|EXP|Exp|EXP|EXPIRY DATE|DATE OF EXPIRY|EXP DATE|Exp|Discard|#).*?(' + combined_pattern + ')', text, re.IGNORECASE)

    
    manufacturing_date = manufacture_keywords.group(2) if manufacture_keywords else None
    expiry_date = expiry_keywords.group(2) if expiry_keywords else None


    return manufacturing_date, expiry_date

def convert_to_date(date_str):
    # Try parsing the date in various formats
    date_formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d %b %Y', '%d%b%Y']
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def classify_priority(exp_date_str):
    # Convert the extracted expiry date string to a datetime object
    exp_date = convert_to_date(exp_date_str)

    if not exp_date:
        return 'Invalid Date'

    current_date = datetime.now()

    # Classify based on the expiry date
    if exp_date < current_date:
        return 'Expired'
    elif (exp_date - current_date).days <= 30:
        return 'Near Expiry'
    else:
        return 'Safe'

def process_images(image_paths):
    product_details = []

    for image_path in image_paths:
        # Extract text from the image
        text = extract_text_from_image(image_path)

        if text:
            # Extract manufacturing and expiry dates
            mfg_date, exp_date = extract_dates(text)

            # Classify the expiry date priority
            if exp_date:
                priority = classify_priority(exp_date)
                exp_date_obj = convert_to_date(exp_date)
            else:
                priority = 'No Expiry Date Found'
                exp_date_obj = None

            # Store the product details
            product_details.append({
                'image': image_path,
                'mfg_date': mfg_date,
                'exp_date': exp_date,
                'priority': priority,
                'expiry_date_obj': exp_date_obj
            })

    # Sort the products based on priority and expiry date (if available)
    product_details.sort(key=lambda x: (x['priority'], x['expiry_date_obj'] or datetime.max))

    return product_details

def extract_data_from_image(image_path):
    # Use EasyOCR to read text from the image
    text = extract_text_from_image(image_path)
    mrp_pattern = re.compile(r'(?:MRP|Maximum Retail Price|Retail Price|MRP Price|MRP|Price)[^\d]*(?:₹|Rs|INR|USD|€)?[\s]*([\d\s,.]+)', re.IGNORECASE)
    brand_pattern = re.compile(r'(?i)(?:Brand|Manufacturer|Product)\s*[:\-]?\s*([A-Za-z0-9\s&]+)', re.IGNORECASE)
    dimensions_pattern = re.compile(r'(?i)(?:Dimensions|Size|Dimension)\s*[:\-]?\s*([0-9]+x[0-9]+)', re.IGNORECASE)
    product_type_pattern = re.compile(r'(?i)(?:Product Type|Category|Type)\s*[:\-]?\s*([A-Za-z0-9\s]+)', re.IGNORECASE)

    mrp = None
    brand = None
    dimensions = None
    product_type = None

    # Search for MRP in the text
    mrp_match = mrp_pattern.search(text)
    if mrp_match:
        mrp = mrp_match.group(1).replace(',', '').replace(' ', '')

    # Search for Brand in the text
    brand_match = brand_pattern.search(text)
    if brand_match:
        brand = brand_match.group(1).strip()

    # Search for Dimensions in the text
    dimensions_match = dimensions_pattern.search(text)
    if dimensions_match:
        dimensions = dimensions_match.group(1).strip()

    # Search for Product Type in the text
    product_type_match = product_type_pattern.search(text)
    if product_type_match:
        product_type = product_type_match.group(1).strip()

    return {
        "MRP": mrp,
        "Brand": brand,
        "Dimensions": dimensions,
        "Product Type": product_type
    }


if __name__ == '__main__':
    app.run(debug=True)
