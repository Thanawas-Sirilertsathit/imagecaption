from flask import Flask, request, render_template, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and processor
processor = None
model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the BLIP model and processor"""
    global processor, model
    try:
        logger.info("Loading BLIP model...")
        
        # Load processor
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        if processor is None:
            raise Exception("Failed to load processor")
        logger.info("Processor loaded successfully")
        
        # Load model
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        if model is None:
            raise Exception("Failed to load model")
        logger.info("Model loaded successfully")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Verify model is properly loaded
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Processor type: {type(processor)}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Set globals to None on failure
        processor = None
        model = None
        raise

def generate_caption(image_path):
    """Generate caption for an image"""
    global model, processor
    
    try:
        # Check if model and processor are loaded
        if model is None or processor is None:
            logger.error("Model or processor is None")
            load_model()  # Try to reload
            if model is None or processor is None:
                return "Error: Model not loaded properly"
        
        # Check if image file exists
        if not os.path.exists(image_path):
            return "Error: Image file not found"
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Image loaded successfully: {image.size}")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return f"Error loading image: {str(e)}"
        
        # Process image with error handling
        try:
            inputs = processor(image, return_tensors="pt")
            logger.info("Image processed by processor")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return f"Error processing image: {str(e)}"
        
        # Move inputs to same device as model
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info(f"Inputs moved to device: {device}")
        except Exception as e:
            logger.error(f"Error moving inputs to device: {str(e)}")
            return f"Error moving inputs to device: {str(e)}"
        
        # Generate caption
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50, num_beams=5)
            logger.info("Caption generated successfully")
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return f"Error during generation: {str(e)}"
        
        # Decode caption
        try:
            caption = processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Caption decoded: {caption}")
            return caption if caption else "No caption generated"
        except Exception as e:
            logger.error(f"Error decoding caption: {str(e)}")
            return f"Error decoding caption: {str(e)}"
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_caption: {str(e)}")
        return f"Unexpected error: {str(e)}"

def image_to_base64(image_path):
    """Convert image to base64 for display in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Get image format
            with Image.open(image_path) as img:
                img_format = img.format.lower()
                
            return f"data:image/{img_format};base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle multiple file uploads and generate captions"""
    logger.info("Received request to /upload endpoint")

    if 'files[]' not in request.files:
        logger.error("No files[] key in request.files")
        return jsonify({'error': 'No files selected. Ensure files are uploaded with the key `files[]`.'}), 400

    files = request.files.getlist('files[]')
    if not files:
        logger.error("No files found in the files[] list")
        return jsonify({'error': 'No files selected'}), 400

    results = []

    for file in files:
        if file and allowed_file(file.filename):
            try:
                logger.info(f"Processing file: {file.filename}")

                # Save uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Generate caption
                caption = generate_caption(filepath)

                # Convert image to base64 for display
                image_base64 = image_to_base64(filepath)

                # Append result
                results.append({
                    'filename': filename,
                    'caption': caption,
                    'image': image_base64,
                    'success': True
                })

                # Clean up uploaded file
                os.remove(filepath)

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'error': f'Error processing image: {str(e)}',
                    'success': False
                })
        else:
            logger.warning(f"Invalid file type or empty file: {file.filename}")
            results.append({
                'filename': file.filename,
                'error': 'Invalid file type. Please upload an image file.',
                'success': False
            })

    logger.info("Completed processing all files")
    return jsonify(results)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'processor_loaded': processor is not None,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    })

# HTML Template (embedded for simplicity)
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Caption Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #4facfe;
            background: #f8fbff;
        }
        
        .upload-area.dragover {
            border-color: #4facfe;
            background: #f0f8ff;
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 3em;
            color: #ddd;
            margin-bottom: 15px;
        }
        
        .upload-text {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 15px;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(79, 172, 254, 0.3);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4facfe;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result {
            display: none;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 30px;
            margin-top: 30px;
        }
        
        .result-image {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .caption {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4facfe;
            font-size: 1.1em;
            line-height: 1.6;
            color: #333;
        }
        
        .error {
            background: #ffe6e6;
            color: #d8000c;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid #d8000c;
        }
        
        .file-info {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Image Caption Generator</h1>
            <p>Upload images and let AI describe what it sees</p>
        </div>
        
        <div class="content">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">+</div>
                    <div class="upload-text">Click to select images or drag and drop</div>
                    <input type="file" id="fileInput" class="file-input" accept="image/*" multiple>
                    <button type="button" class="btn" onclick="document.getElementById('fileInput').click()">
                        Choose Images
                    </button>
                </div>
                
                <div id="fileInfo" class="file-info" style="display: none;"></div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Generating captions... This may take a few moments.</p>
                </div>
                
                <div class="result" id="result"></div>
                <div id="error" class="error" style="display: none;"></div>
            </form>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const error = document.getElementById('error');
        
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files);
            }
        });
        
        function handleFileSelect(files) {
            const validFiles = [];
            fileInfo.innerHTML = '';

            for (const file of files) {
                if (!file.type.startsWith('image/')) {
                    showError(`Invalid file type: ${file.name}`);
                    continue;
                }

                if (file.size > 16 * 1024 * 1024) {
                    showError(`File too large: ${file.name}`);
                    continue;
                }

                validFiles.push(file);
                fileInfo.innerHTML += `<strong>${file.name}</strong> (${(file.size / 1024 / 1024).toFixed(2)} MB)<br>`;
            }

            if (validFiles.length > 0) {
                fileInfo.style.display = 'block';
                uploadFiles(validFiles);
            }
        }

        function uploadFiles(files) {
            const formData = new FormData();
            for (const file of files) {
                formData.append('files[]', file);
            }

            loading.style.display = 'block';
            result.style.display = 'none';
            error.style.display = 'none';

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';

                if (Array.isArray(data)) {
                    result.innerHTML = data.map(item => `
                        <div>
                            <img src="${item.image}" class="result-image" alt="Uploaded image">
                            <div class="caption">${item.caption}</div>
                        </div>
                    `).join('');
                    result.style.display = 'block';
                } else {
                    showError(data.error || 'An error occurred while processing the images.');
                }
            })
            .catch(err => {
                loading.style.display = 'none';
                showError('Network error. Please try again.');
                console.error('Error:', err);
            });
        }

        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
            result.style.display = 'none';
        }
    </script>
</body>
</html>
'''

# Create templates directory and save template
templates_dir = 'templates'
os.makedirs(templates_dir, exist_ok=True)

# Write template with explicit UTF-8 encoding to handle any Unicode characters
try:
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_template)
except UnicodeEncodeError:
    # Fallback: write without problematic characters
    clean_template = html_template.encode('ascii', 'ignore').decode('ascii')
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(clean_template)

if __name__ == '__main__':
    # Load model on startup
    try:
        load_model()
        print("🚀 Starting Flask application...")
        print("📝 Model loaded successfully!")
        print("🌐 Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        print("Make sure you have the required dependencies installed:")
        print("pip install flask torch transformers pillow")