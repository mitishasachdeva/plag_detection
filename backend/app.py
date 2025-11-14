import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'project_1')

MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'siamese_model.pth')
CONFIG_PATH = os.path.join(MODEL_DIR, 'inference_config.json') 
TOKENIZER_NAME = "microsoft/codebert-base"

warnings.filterwarnings("ignore", category=UserWarning)

class SiameseNetwork(nn.Module):
    def __init__(self, model_name=TOKENIZER_NAME, hidden_size=768):
        super(SiameseNetwork, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        out1 = self.encoder(input_ids=input_ids1, attention_mask=attention_mask1).pooler_output
        out2 = self.encoder(input_ids=input_ids2, attention_mask=attention_mask2).pooler_output
        out1 = self.fc(out1)
        out2 = self.fc(out2)
        return out1, out2

def load_plagiarism_model(model_path, device):
    """ Loads the trained SiameseNetwork model. """
    print("Loading SiameseNetwork model architecture...")
    model = SiameseNetwork().to(device)
    print(f"Loading trained weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def check_plagiarism(code1, code2, model, tokenizer, device, threshold):
    """
    Checks two code snippets for plagiarism using the loaded model and threshold.
    Returns: (verdict_string, similarity_score_float)
    """
    model.eval() 
    
    inputs1 = tokenizer(code1, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    inputs2 = tokenizer(code2, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        output1, output2 = model(inputs1.input_ids, inputs1.attention_mask, 
                                 inputs2.input_ids, inputs2.attention_mask)
        
        similarity_score = nn.CosineSimilarity(dim=1)(output1, output2).item()

    if similarity_score > threshold:
        verdict = "Plagiarized"
    else:
        verdict = "Not Plagiarized"
    
    return verdict, similarity_score

def preprocess_code(code):
    """Basic code preprocessing"""
    if not code or not isinstance(code, str):
        return ""
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\s+', ' ', code).strip()
    return code

# --- 4. Flask App Setup & Model Loading ---
app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load models and config at startup
try:
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    print(f"Loading model weights from: {MODEL_WEIGHTS_PATH}")
    plagiarism_model = load_plagiarism_model(MODEL_WEIGHTS_PATH, device)
    
    print(f"Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    BEST_THRESHOLD = config['best_threshold']
    
    print("✅ All models and configs loaded successfully.")
    print(f"✅ Using threshold: {BEST_THRESHOLD}")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load models or config: {e}")
    plagiarism_model = None

# --- 5. API Routes ---
@app.route('/compare_code', methods=['POST'])
def compare_code_route():
    if not plagiarism_model:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    try:
        data = request.json
        code1 = data.get('code1', '')
        code2 = data.get('code2', '')
        
        if not code1 or not code2:
            return jsonify({"error": "Both 'code1' and 'code2' are required"}), 400
        
        clean_code1 = preprocess_code(code1)
        clean_code2 = preprocess_code(code2)
        
        print("Comparing code snippets...")
        verdict, score = check_plagiarism(
            clean_code1, 
            clean_code2,
            plagiarism_model,
            tokenizer,
            device,
            BEST_THRESHOLD
        )
        
        # This response matches your frontend
        return jsonify({
            'similarity_score': score,
            'prediction': verdict,
            'confidence': f'{min(100, int(score * 100))}%',
        })
    
    except Exception as e:
        print(f"❌ Error in /compare_code: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "model": "Siamese CodeBERT",
        "plagiarism_model_loaded": plagiarism_model is not None,
        "threshold": BEST_THRESHOLD if plagiarism_model else "N/A"
    })

# --- 6. Run the App ---
if __name__ == '__main__':
    port = 7860
    print(f"🚀 Starting Flask server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)