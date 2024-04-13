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
    data = request.get_json()

    image = data.get('image')
    type = 1
    crop = predict_crop(image, type)

    return jsonify({'crop': crop})

if __name__ == '__main__':
    print('API is running on port 5000')
    app.run(host='localhost', port=5000)