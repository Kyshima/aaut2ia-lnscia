import io

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_language, predict_crop

app = Flask(__name__)
CORS(app)

@app.route('/language_predict', methods=['POST'])
def get_language_prediction():
    data = request.get_json()

    text = data.get('text')
    disease, treatment = predict_language(text)

    return jsonify({'disease': disease, 'treatment': treatment})

@app.route('/image-crop-predict', methods=['POST'])
def get_image_prediction():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        type_pred = request.form.get('type')
        crop = predict_crop(image, type_pred)

        return jsonify({'crop': crop})
    return jsonify({'error': 'ta errado'})
if __name__ == '__main__':
    print('API is running on port 5000')
    app.run(host='localhost', port=5000)