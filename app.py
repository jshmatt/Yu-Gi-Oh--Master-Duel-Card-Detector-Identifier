import base64
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# import custom models
from YGOmodels.CardDetector import DetectCards
from YGOmodels.CardIdentifier import IdentifyCard

# the app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB

# helper functions

def pil_to_b64(img: Image.Image) -> str:
  buf = io.BytesIO()
  img.save(buf, format='PNG')
  return base64.b64encode(buf.getvalue()).decode()

def ndarray_to_b64(arr: np.ndarray) -> str:
  return pil_to_b64(Image.fromarray(arr.astype(np.uint8)))

# decode b64 string
def b64_to_ndarray(b64: str) -> np.ndarray:
  if ',' in b64:
    b64 = b64.split(',', 1)[1]
  data = base64.b64decode(b64)
  img  = Image.open(io.BytesIO(data)).convert('RGB')
  return np.array(img)

# routes
# home route
@app.route('/')
def index():
  return render_template('index.html')

# detection route
@app.route('/detect', methods=['POST'])
def detect():
  if 'image' not in request.files:
    return jsonify({'error': 'No image field in request.'}), 400

  file = request.files['image']
  if file.filename == '':
    return jsonify({'error': 'Empty filename.'}), 400

  try:
    img_pil = Image.open(file.stream).convert('RGB')
  except Exception as e:
    return jsonify({'error': f'Cannot read image: {e}'}), 400

  
  detector   = DetectCards()
  detections = detector.predict(img_pil)

  objects = [
        {'image_b64': ndarray_to_b64(d['image']), 'type': d['type']}
        for d in detections
  ]
  return jsonify({'objects': objects})

# retrieval route
@app.route('/retrieve', methods=['POST'])
def retrieve():
  body = request.get_json(silent=True)
  if not body or 'image_b64' not in body:
    return jsonify({'error': 'Missing image_b64 in JSON body.'}), 400

  try:
    img_np   = b64_to_ndarray(body['image_b64'])
    obj_type = body.get('type', 'unknown')
  except Exception as e:
    return jsonify({'error': f'Cannot decode image: {e}'}), 400

  identifier = IdentifyCard()
  result = identifier.single_card_identify([img_np, obj_type])
  result_ = {'candidates':result['candidates'],
            'plot_b64':result['plot_b64']}

  return jsonify(result_)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)