from flask import Flask, render_template, request, send_from_directory, jsonify
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import traceback
import json
import pickle
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained models
try:
    brain_tumor_model = load_model('model.h5')
    print("Brain tumor model loaded successfully!")
    print("Model input shape:", brain_tumor_model.input_shape)
    print("Model output shape:", brain_tumor_model.output_shape)
except Exception as e:
    print("Error loading brain tumor model:", str(e))
    print(traceback.format_exc())
    brain_tumor_model = None

# Function to create a new diabetes model if needed
def create_new_model():
    df = pd.read_csv('diabetes-copy/diabetes.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    # Save the model
    with open('diabetes-copy/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model

# Load or create diabetes model
try:
    if os.path.exists('diabetes-copy/model.pkl'):
        try:
            with open('diabetes-copy/model.pkl', 'rb') as file:
                diabetes_model = pickle.load(file)
            # Verify it's the correct type of model
            if not isinstance(diabetes_model, DecisionTreeClassifier):
                print("Saved model is not a DecisionTreeClassifier. Creating new model...")
                diabetes_model = create_new_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}. Creating new model...")
            diabetes_model = create_new_model()
    else:
        print("Model file not found. Creating new model...")
        diabetes_model = create_new_model()
except Exception as e:
    print(f"Error during model initialization: {str(e)}. Creating new model...")
    diabetes_model = create_new_model()

# Class labels
class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Medical knowledge base
MEDICAL_KNOWLEDGE = {
    "fever": {
        "disease": "Viral Fever",
        "description": "A common viral infection causing elevated body temperature and general malaise.",
        "medications": [
            "Acetaminophen (Tylenol)",
            "Ibuprofen (Advil)",
            "Plenty of fluids"
        ],
        "precautions": [
            "Rest and stay hydrated",
            "Monitor temperature",
            "Avoid close contact with others",
            "Use a face mask if going out"
        ],
        "diet": [
            "Clear broths and soups",
            "Herbal teas",
            "Fresh fruits",
            "Light, easily digestible foods"
        ]
    },
    "headache": {
        "disease": "Tension Headache",
        "description": "A common type of headache characterized by a dull, aching pain and tightness around the head.",
        "medications": [
            "Over-the-counter pain relievers",
            "Aspirin",
            "Ibuprofen",
            "Acetaminophen"
        ],
        "precautions": [
            "Rest in a quiet, dark room",
            "Stay hydrated",
            "Practice stress management",
            "Maintain good posture"
        ],
        "diet": [
            "Stay hydrated with water",
            "Avoid caffeine and alcohol",
            "Eat regular meals",
            "Include magnesium-rich foods"
        ]
    },
    "cough": {
        "disease": "Upper Respiratory Infection",
        "description": "A viral infection affecting the upper respiratory tract, causing cough and cold symptoms.",
        "medications": [
            "Cough suppressants",
            "Expectorants",
            "Antihistamines",
            "Steam inhalation"
        ],
        "precautions": [
            "Cover mouth when coughing",
            "Use disposable tissues",
            "Wash hands frequently",
            "Stay home if possible"
        ],
        "diet": [
            "Warm honey and lemon water",
            "Ginger tea",
            "Chicken soup",
            "Vitamin C rich foods"
        ]
    },
    "sore throat": {
        "disease": "Pharyngitis",
        "description": "Inflammation of the pharynx causing pain and difficulty swallowing.",
        "medications": [
            "Throat lozenges",
            "Saltwater gargles",
            "Pain relievers",
            "Antibiotics (if bacterial)"
        ],
        "precautions": [
            "Rest voice",
            "Avoid irritants",
            "Stay hydrated",
            "Practice good hygiene"
        ],
        "diet": [
            "Warm soups",
            "Honey",
            "Cold foods like ice cream",
            "Soft, non-irritating foods"
        ]
    },
    "nausea": {
        "disease": "Gastritis",
        "description": "Inflammation of the stomach lining causing nausea and discomfort.",
        "medications": [
            "Antacids",
            "Anti-nausea medication",
            "H2 blockers",
            "Proton pump inhibitors"
        ],
        "precautions": [
            "Eat smaller meals",
            "Avoid trigger foods",
            "Stay upright after eating",
            "Avoid alcohol"
        ],
        "diet": [
            "Bland foods",
            "Clear liquids",
            "Bananas",
            "Rice and toast"
        ]
    },
    "joint pain": {
        "disease": "Arthritis",
        "description": "Inflammation of joints causing pain and stiffness.",
        "medications": [
            "NSAIDs",
            "Acetaminophen",
            "Topical pain relievers",
            "Prescribed anti-inflammatory drugs"
        ],
        "precautions": [
            "Regular exercise",
            "Maintain healthy weight",
            "Use assistive devices",
            "Protect joints"
        ],
        "diet": [
            "Anti-inflammatory foods",
            "Omega-3 rich foods",
            "Fruits and vegetables",
            "Avoid processed foods"
        ]
    },
    "dizziness": {
        "disease": "Vertigo",
        "description": "A sensation of spinning or movement when stationary.",
        "medications": [
            "Anti-vertigo medications",
            "Motion sickness pills",
            "Balance medications",
            "Antihistamines"
        ],
        "precautions": [
            "Avoid sudden movements",
            "Use handrails",
            "Sleep with head elevated",
            "Avoid triggers"
        ],
        "diet": [
            "Stay hydrated",
            "Low-salt diet",
            "Avoid caffeine",
            "Regular meal times"
        ]
    },
    "rash": {
        "disease": "Contact Dermatitis",
        "description": "Skin inflammation caused by contact with irritants or allergens.",
        "medications": [
            "Topical corticosteroids",
            "Antihistamines",
            "Calamine lotion",
            "Moisturizers"
        ],
        "precautions": [
            "Identify and avoid triggers",
            "Wear protective clothing",
            "Keep skin clean",
            "Avoid scratching"
        ],
        "diet": [
            "Anti-inflammatory foods",
            "Omega-3 rich foods",
            "Avoid known food allergens",
            "Stay hydrated"
        ]
    },
    "fatigue": {
        "disease": "Chronic Fatigue Syndrome",
        "description": "Persistent fatigue that interferes with daily activities.",
        "medications": [
            "Pain relievers",
            "Sleep medications",
            "Antidepressants",
            "Supplements"
        ],
        "precautions": [
            "Pace activities",
            "Regular sleep schedule",
            "Stress management",
            "Gentle exercise"
        ],
        "diet": [
            "Energy-rich foods",
            "Regular small meals",
            "Complex carbohydrates",
            "Avoid processed foods"
        ]
    },
    "chest pain": {
        "disease": "Angina",
        "description": "Chest pain caused by reduced blood flow to the heart.",
        "medications": [
            "Nitroglycerin",
            "Beta blockers",
            "Blood thinners",
            "Anti-anginal medications"
        ],
        "precautions": [
            "Regular check-ups",
            "Monitor blood pressure",
            "Avoid overexertion",
            "Emergency plan"
        ],
        "diet": [
            "Low-fat foods",
            "Heart-healthy diet",
            "Reduce salt intake",
            "Avoid large meals"
        ]
    },
    "shortness of breath": {
        "disease": "Asthma",
        "description": "Chronic condition causing airway inflammation and breathing difficulty.",
        "medications": [
            "Inhaled corticosteroids",
            "Bronchodilators",
            "Rescue inhalers",
            "Anti-inflammatory medications"
        ],
        "precautions": [
            "Avoid triggers",
            "Regular peak flow monitoring",
            "Follow action plan",
            "Keep rescue inhaler handy"
        ],
        "diet": [
            "Anti-inflammatory foods",
            "Vitamin D rich foods",
            "Avoid sulfites",
            "Stay hydrated"
        ]
    },
    "stomach pain": {
        "disease": "Gastroenteritis",
        "description": "Inflammation of the stomach and intestines causing pain and discomfort.",
        "medications": [
            "Anti-diarrheal medication",
            "Pain relievers",
            "Probiotics",
            "Electrolyte solutions"
        ],
        "precautions": [
            "Hand hygiene",
            "Food safety",
            "Rest",
            "Hydration"
        ],
        "diet": [
            "BRAT diet",
            "Clear liquids",
            "Avoid dairy",
            "Small frequent meals"
        ]
    },
    "back pain": {
        "disease": "Lumbar Strain",
        "description": "Pain in the lower back caused by muscle or ligament strain.",
        "medications": [
            "NSAIDs",
            "Muscle relaxants",
            "Topical pain relievers",
            "Heat/cold therapy"
        ],
        "precautions": [
            "Proper posture",
            "Ergonomic workspace",
            "Avoid heavy lifting",
            "Regular stretching"
        ],
        "diet": [
            "Anti-inflammatory foods",
            "Calcium-rich foods",
            "Vitamin D sources",
            "Maintain healthy weight"
        ]
    },
    "anxiety": {
        "disease": "Generalized Anxiety Disorder",
        "description": "Persistent and excessive worry about various aspects of life.",
        "medications": [
            "Anti-anxiety medications",
            "SSRIs",
            "Beta blockers",
            "Natural supplements"
        ],
        "precautions": [
            "Regular exercise",
            "Stress management",
            "Sleep hygiene",
            "Avoid triggers"
        ],
        "diet": [
            "Complex carbohydrates",
            "Omega-3 rich foods",
            "Limit caffeine",
            "Avoid alcohol"
        ]
    },
    "insomnia": {
        "disease": "Sleep Disorder",
        "description": "Difficulty falling asleep or staying asleep.",
        "medications": [
            "Sleep medications",
            "Melatonin",
            "Herbal supplements",
            "Anti-anxiety medications"
        ],
        "precautions": [
            "Regular sleep schedule",
            "Dark, quiet room",
            "Limit screen time",
            "Relaxation techniques"
        ],
        "diet": [
            "Avoid caffeine",
            "Light evening meals",
            "Calming teas",
            "Tryptophan-rich foods"
        ]
    },
    "muscle weakness": {
        "disease": "Myasthenia Gravis",
        "description": "Autoimmune disorder causing muscle weakness and fatigue.",
        "medications": [
            "Cholinesterase inhibitors",
            "Immunosuppressants",
            "Steroids",
            "Plasma exchange"
        ],
        "precautions": [
            "Regular rest periods",
            "Avoid overexertion",
            "Temperature control",
            "Regular monitoring"
        ],
        "diet": [
            "High-protein foods",
            "Easy to chew foods",
            "Small frequent meals",
            "Adequate hydration"
        ]
    },
    "vision problems": {
        "disease": "Glaucoma",
        "description": "Eye condition causing increased pressure and potential vision loss.",
        "medications": [
            "Eye drops",
            "Beta blockers",
            "Prostaglandins",
            "Carbonic anhydrase inhibitors"
        ],
        "precautions": [
            "Regular eye exams",
            "Protect eyes",
            "Monitor pressure",
            "Early treatment"
        ],
        "diet": [
            "Leafy greens",
            "Omega-3 rich foods",
            "Vitamin A foods",
            "Antioxidant-rich foods"
        ]
    },
    "memory loss": {
        "disease": "Early Dementia",
        "description": "Progressive decline in cognitive function and memory.",
        "medications": [
            "Cholinesterase inhibitors",
            "Memantine",
            "Antidepressants",
            "Anxiety medications"
        ],
        "precautions": [
            "Mental stimulation",
            "Regular routines",
            "Safety measures",
            "Social engagement"
        ],
        "diet": [
            "Mediterranean diet",
            "Omega-3 rich foods",
            "Antioxidants",
            "B-vitamin rich foods"
        ]
    },
    "weight loss": {
        "disease": "Hyperthyroidism",
        "description": "Overactive thyroid causing increased metabolism and weight loss.",
        "medications": [
            "Anti-thyroid medications",
            "Beta blockers",
            "Radioactive iodine",
            "Supplements"
        ],
        "precautions": [
            "Regular monitoring",
            "Avoid iodine-rich foods",
            "Stress management",
            "Temperature control"
        ],
        "diet": [
            "High-calorie foods",
            "Protein-rich foods",
            "Regular meals",
            "Calcium-rich foods"
        ]
    },
    "swollen joints": {
        "disease": "Rheumatoid Arthritis",
        "description": "Autoimmune disorder causing joint inflammation and pain.",
        "medications": [
            "DMARDs",
            "NSAIDs",
            "Corticosteroids",
            "Biologics"
        ],
        "precautions": [
            "Joint protection",
            "Regular exercise",
            "Rest when needed",
            "Occupational therapy"
        ],
        "diet": [
            "Anti-inflammatory foods",
            "Omega-3 rich foods",
            "Mediterranean diet",
            "Avoid processed foods"
        ]
    },
    "frequent urination": {
        "disease": "Urinary Tract Infection",
        "description": "Infection in any part of the urinary system.",
        "medications": [
            "Antibiotics",
            "Pain relievers",
            "Urinary pain relief",
            "Probiotics"
        ],
        "precautions": [
            "Proper hygiene",
            "Stay hydrated",
            "Empty bladder regularly",
            "Avoid irritants"
        ],
        "diet": [
            "Cranberry juice",
            "Probiotic foods",
            "Vitamin C rich foods",
            "Avoid caffeine"
        ]
    },
    "skin itching": {
        "disease": "Eczema",
        "description": "Chronic skin condition causing inflammation and itching.",
        "medications": [
            "Topical corticosteroids",
            "Antihistamines",
            "Moisturizers",
            "Immunosuppressants"
        ],
        "precautions": [
            "Avoid triggers",
            "Gentle skin care",
            "Humidity control",
            "Avoid scratching"
        ],
        "diet": [
            "Anti-inflammatory foods",
            "Omega-3 rich foods",
            "Probiotics",
            "Avoid trigger foods"
        ]
    }
}

# Helper function to predict tumor type
def predict_tumor(image_path):
    if brain_tumor_model is None:
        return "Error: Model not loaded", 0.0
    
    try:
        print(f"Processing image: {image_path}")
        
        # Load and preprocess the image
        IMAGE_SIZE = (128, 128)  # Updated to match model's expected input size
        print(f"Loading image with target size: {IMAGE_SIZE}")
        img = load_img(image_path, target_size=IMAGE_SIZE)
        
        print("Converting image to array")
        img_array = img_to_array(img)
        print(f"Image array shape after conversion: {img_array.shape}")
        
        print("Expanding dimensions")
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Image array shape after expansion: {img_array.shape}")
        
        print("Normalizing image")
        img_array = img_array / 255.0
        
        print("Making prediction")
        predictions = brain_tumor_model.predict(img_array)
        print(f"Raw predictions: {predictions}")
        
        predicted_class_index = np.argmax(predictions[0])
        confidence_score = predictions[0][predicted_class_index]
        
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Confidence score: {confidence_score}")

        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]
        print(f"Predicted class: {predicted_class}")

        return predicted_class, confidence_score

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("Detailed error:")
        print(traceback.format_exc())
        return f"Error during prediction: {str(e)}", 0.0

def get_available_symptoms():
    return list(MEDICAL_KNOWLEDGE.keys())

@app.route('/get-symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({
        "symptoms": get_available_symptoms()
    })

def analyze_symptoms(symptoms):
    # Convert symptoms to lowercase and trim for matching
    symptoms = [s.lower().strip() for s in symptoms if s]
    print("Processing symptoms:", symptoms)  # Debug log
    
    # Get available symptoms
    available_symptoms = get_available_symptoms()
    print("Available symptoms:", available_symptoms)  # Debug log
    
    # Find matching conditions
    matching_conditions = []
    for symptom in symptoms:
        print(f"Checking symptom: '{symptom}'")  # Debug log
        
        # Try exact match first
        if symptom in MEDICAL_KNOWLEDGE:
            print(f"Found exact match for symptom: '{symptom}'")  # Debug log
            matching_conditions.append(MEDICAL_KNOWLEDGE[symptom])
            continue
            
        # Try partial matches
        matched = False
        for known_symptom in available_symptoms:
            # Convert both to lowercase for comparison
            symptom_lower = symptom.lower()
            known_lower = known_symptom.lower()
            
            # Check if the symptom is part of a known symptom or vice versa
            if symptom_lower in known_lower or known_lower in symptom_lower:
                print(f"Found partial match: '{known_symptom}' for '{symptom}'")  # Debug log
                matching_conditions.append(MEDICAL_KNOWLEDGE[known_symptom])
                matched = True
                break
        
        if not matched:
            print(f"No match found for symptom: '{symptom}'")  # Debug log
    
    if not matching_conditions:
        print("No matching conditions found for any symptoms")  # Debug log
        return {
            "disease": "Unknown Condition",
            "description": "The symptoms provided don't match any known conditions in our database. Please consult a healthcare professional.",
            "medications": ["Consult a doctor for proper medication"],
            "precautions": ["Seek medical attention if symptoms worsen"],
            "diet": ["Maintain a balanced diet", "Stay hydrated"]
        }
    
    # Combine information from all matching conditions
    combined_info = {
        "disease": ", ".join(condition["disease"] for condition in matching_conditions),
        "description": " ".join(condition["description"] for condition in matching_conditions),
        "medications": list(set(med for condition in matching_conditions for med in condition["medications"])),
        "precautions": list(set(prec for condition in matching_conditions for prec in condition["precautions"])),
        "diet": list(set(diet for condition in matching_conditions for diet in condition["diet"]))
    }
    
    print("Found matching conditions:", [info["disease"] for info in matching_conditions])  # Debug log
    return combined_info

def predict_diabetes(data):
    if diabetes_model is None:
        return {"error": "Model not loaded"}, 500
    
    try:
        # Create feature array in the correct order
        input_data = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['blood_pressure']),
            float(data['skin_thickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetes_pedigree']),
            float(data['age'])
        ]
        
        # Make prediction
        input_array = np.array(input_data).reshape(1, -1)
        prediction = int(diabetes_model.predict(input_array)[0])
        probability = float(diabetes_model.predict_proba(input_array)[0][1])
        
        print(f"Features: {input_array}")
        print(f"Prediction: {prediction}")
        print(f"Probability: {probability}")
        
        return {
            "prediction": bool(prediction),
            "confidence": f"{probability * 100:.1f}"
        }
    
    except Exception as e:
        print(f"Error during diabetes prediction: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}, 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        if file:
            try:
                # Save the file
                file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_location)

                # Predict the tumor
                result, confidence = predict_tumor(file_location)

                # Return result along with image path for display
                return render_template('index.html', 
                                    result=result, 
                                    confidence=f"{confidence*100:.2f}%", 
                                    file_path=f'/uploads/{file.filename}')

            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {str(e)}")

    return render_template('index.html')

@app.route('/ai-doctor')
def ai_doctor():
    return render_template('ai_doctor.html')

@app.route('/diabetes-prediction')
def diabetes_prediction():
    return render_template('diabetes_prediction.html')

@app.route('/predict-diabetes', methods=['POST'])
def predict_diabetes_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Validate required fields
        required_fields = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                         'insulin', 'bmi', 'diabetes_pedigree', 'age']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        result = predict_diabetes(data)
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict_diabetes_route: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms_route():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug log
        symptoms = data.get('symptoms', [])
        print("Extracted symptoms:", symptoms)  # Debug log
        
        if not symptoms:
            print("No symptoms found in request")  # Debug log
            return jsonify({
                "error": "No symptoms provided"
            }), 400
        
        result = analyze_symptoms(symptoms)
        print("Analysis result:", result)  # Debug log
        return jsonify(result)
    
    except Exception as e:
        print("Error in analyze_symptoms_route:", str(e))  # Debug log
        print(traceback.format_exc())  # Print full traceback
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
