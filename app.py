from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Load the pre-trained model, tokenizer, and label encoder
model = load_model('was_model.h5')

with open('was_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('was_label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Define the prediction function
def predict_sentiment(text_input):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text_input])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded_sequence)
    

    predicted_label_index = np.argmax(prediction, axis=1)[0]
    
    # Define your own mapping
    label_mapping = {'mild': 0, 'moderate': 1, 'severe': 2, 'non-depressed': 3}
    predicted_label = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_label_index)]

    return predicted_label

# Define a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
  data = request.json
  text_input = data['text']

  predicted_sentiment = predict_sentiment(text_input)
  response = {
        'predicted_sentiment': predicted_sentiment
    }
  return jsonify(response)
  
  
# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
