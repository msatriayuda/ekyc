import os
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Image, Part, SafetySetting

from config import GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GEMINI_MODEL

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize Vertex AI
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)

def analyze_image_with_gemini(image_bytes):
    """Analyze image using Gemini Flash 1.5"""
    try:
        # Create Gemini model
        model = GenerativeModel(GEMINI_MODEL)
        
        # Convert image bytes to Gemini Image
        image = Image.from_bytes(image_bytes)
        
        # Detailed prompt for ID extraction
        prompt = """
        Carefully extract all readable information from this ID document.
        Provide details in a structured format including:
        - Full Name
        - ID Number
        - Date of Birth
        - Address
        - Issue Date
        - Expiration Date
        
        If any information is not visible or readable, state 'Not Available'.
        """
        
        # Generate response
        response = model.generate_content([prompt, image])
        
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def analyze_liveness_with_gemini(video_bytes):
    """Perform liveness detection using Gemini Flash 1.5"""
    try:
        # Create Gemini model
        model = GenerativeModel(GEMINI_MODEL)
        
        # Convert video bytes to Gemini Image
        video_image = Image.from_bytes(video_bytes)
        
        # Detailed prompt for liveness detection
        prompt = """
        Analyze this video to determine if it represents a live human presence.
        Look for key indicators of liveness:
        - Natural head movement
        - Blinking
        - Facial expression changes
        - No signs of static image or pre-recorded video
        
        Provide a clear assessment:
        - Is this a live person?
        - Any suspicious elements detected?
        """
        
        # Generate response
        response = model.generate_content([prompt, video_image])
        
        return response.text
    except Exception as e:
        return f"Liveness Check Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/id_extractor', methods=['GET', 'POST'])
def id_extractor():
    if request.method == 'POST':
        if 'id_image' not in request.files:
            return jsonify({"error": "No file uploaded"})
        
        file = request.files['id_image']
        image_bytes = file.read()
        
        # Analyze image
        extraction_result = analyze_image_with_gemini(image_bytes)
        
        return jsonify({
            "result": extraction_result,
            "image": base64.b64encode(image_bytes).decode('utf-8')
        })
    
    return render_template('id_extractor.html')

@app.route('/liveness', methods=['GET', 'POST'])
def liveness():
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({"error": "No video uploaded"})
        
        file = request.files['video']
        video_bytes = file.read()
        
        # Perform liveness detection
        liveness_result = analyze_liveness_with_gemini(video_bytes)
        
        return jsonify({
            "result": liveness_result,
            "video": base64.b64encode(video_bytes).decode('utf-8')
        })
    
    return render_template('liveness.html')

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)