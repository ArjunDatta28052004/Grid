import os
from flask import Flask, request, render_template, redirect, flash
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
app.secret_key = '8'
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load models and data
class_names_path = r'C:\Users\Arjun Datta PC\OneDrive\Desktop\GRID\class_names.pkl'
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

# Example known brand list (add or modify as needed)
known_brands = [
    "Coca-Cola", "Pepsi", "Nestle", "Amul", "Dove", "Parle", "Britannia",
    "Samsung", "LG", "Arrow", "Pepe Jeans", "Wrangler", "Hindustan Unilever",
    "Arvind Fashions", "Smart Clothing"  # Added for specific case
]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    return denoised

def extract_information(text):
    mrp = None
    brand = None
    dimensions = None
    product_type = None

    mrp_pattern = r'(?:MRP|Maximum Retail Price|Retail Price|MRP Price|MRP|Price)[^\d]*(?:₹|Rs|INR|USD|€)?[\s]*([\d\s,.]+)'
    dimension_pattern_waist = r'Waist Size[\s]*([\d]+)\s?cm'
    dimension_pattern_inseam = r'Inseam Length[\s]*([\d]+)\s?cm'

    mrp_match = re.search(mrp_pattern, text, re.IGNORECASE)
    if mrp_match:
        mrp = mrp_match.group(1).replace(',', '').replace(' ', '')

    waist_match = re.search(dimension_pattern_waist, text, re.IGNORECASE)
    inseam_match = re.search(dimension_pattern_inseam, text, re.IGNORECASE)

    if waist_match and inseam_match:
        waist_size = waist_match.group(1)
        inseam_length = inseam_match.group(1)
        dimensions = f'Waist: {waist_size} cm, Inseam: {inseam_length} cm'

    brand, product_type = extract_brand_and_product_type(text)

    return {
        "MRP": mrp,
        "Brand": brand,
        "Dimensions": dimensions,
        "Product Type": product_type
    }

def extract_brand_and_product_type(text):
    brand = None
    product_type = None
    url_pattern = r'(https?://[^\s]+)'
    product_type_keywords = ['trouser', 'pants', 'jeans', 'shorts', 'skirt', 'dress']

    normalized_text = text.lower()
    url_match = re.search(url_pattern, text)
    if url_match:
        url = url_match.group(1)
        parsed_url = urlparse(url)
        brand = parsed_url.netloc.split('.')[0]

    if not brand:
        for known_brand in known_brands:
            if known_brand.lower() in normalized_text:
                brand = known_brand
                break

    for keyword in product_type_keywords:
        if keyword in normalized_text:
            product_type = keyword.capitalize()
            break

    return brand, product_type

def extract_data_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    results = reader.readtext(preprocessed_image, detail=0)
    extracted_text = " ".join(results)
    print(f"Extracted Text: {extracted_text}")  # Debugging line
    data = extract_information(extracted_text)
    print(f"Extracted Data: {data}")  # Debugging line
    return data


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path, model, class_names, device):
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
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
    
    predicted_class = class_names[preds.item()]
    return predicted_class

def extract_text_from_image(image_path):
    result = reader.readtext(image_path, detail=0)
    return ' '.join(result)

def extract_dates(text):
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',
        r'\b\d{2}/\d{2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{4}/\d{2}/\d{2}\b',
        r'\b\d{2}[A-Za-z]{3}\d{4}\b',
        r'\b\d{2} [A-Za-z]{3} \d{4}\b'
    ]
    mfg_match = re.search(r'(MFD|MFG|DATE OF MANUFACTURE|Mfg|Made|Date).*?(' + '|'.join(date_patterns) + ')', text, re.IGNORECASE)
    exp_match = re.search(r'(EXP|EXPIRY DATE|DATE OF EXPIRY|Exp).*?(' + '|'.join(date_patterns) + ')', text, re.IGNORECASE)
    
    mfg_date = mfg_match.group(2) if mfg_match else None
    exp_date = exp_match.group(2) if exp_match else None
    return mfg_date, exp_date

def convert_to_date(date_str):
    date_formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d %b %Y', '%d%b%Y']
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def classify_priority(exp_date_str):
    exp_date = convert_to_date(exp_date_str)
    if not exp_date:
        return 'Invalid Date'
    
    current_date = datetime.now()
    if exp_date < current_date:
        return 'Expired'
    elif (exp_date - current_date).days <= 30:
        return 'Near Expiry'
    else:
        return 'Safe'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        predict_type = request.form.get('predict_type')
        product_details = []
        valid_images = True

        if not files:
            flash('No files uploaded')
            return redirect(request.url)

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                if predict_type == 'expiry':
                    text = extract_text_from_image(file_path)
                    mfg_date, exp_date = extract_dates(text)
                    if exp_date:
                        priority = classify_priority(exp_date)
                        exp_date_obj = convert_to_date(exp_date)
                    else:
                        priority = 'No Expiry Date Found'
                        exp_date_obj = None

                    product_details.append({
                        'image': filename,
                        'mfg_date': mfg_date,
                        'exp_date': exp_date,
                        'priority': priority,
                        'expiry_date_obj': exp_date_obj
                    })

                    if exp_date is None:
                        valid_images = False

                elif predict_type == 'freshness':
                    predicted_class = predict_image(file_path, model_ft, class_names, device)
                    product_details.append({
                        'image': filename,
                        'prediction': predicted_class
                    })

                elif predict_type == 'brand_detection':
                    data = extract_data_from_image(file_path)
                    product_details.append({
                        'image': filename,
                        'MRP': data['MRP'],
                        'Brand': data['Brand'],
                        'Dimensions': data['Dimensions'],
                        'Product Type': data['Product Type']
                    })

        return render_template('result.html', product_details=product_details, valid_images=valid_images)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
                 
