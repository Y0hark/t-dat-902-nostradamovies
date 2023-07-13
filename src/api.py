from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import io
from PIL import Image

# Charger votre modèle entraîné
model = load_model(os.path.abspath('./models/Model_6c.h5'))

col_names =  ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History',
                  'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance',
                  'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if request.method == "POST":
        if request.files.get("image"):
            # lire l'image en couleur (3 canaux)
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = image.resize((150, 200))
            image = np.array(image)
            image = np.expand_dims(image, axis=0)

            # scale les images de [0, 255] à [0, 1]
            image = image / 255.0

            # faire la prédiction
            prediction = model.predict(image)
            
            # obtenir les indices des 3 classes les plus probables
            top3_indices = prediction[0].argsort()[-3:][::-1]
            
            # obtenir les prédictions correspondantes
            top3_predictions = prediction[0][top3_indices]

            # Résultat
            data["predictions"] = top3_predictions.tolist()
            data["labels"] = [col_names[i] for i in top3_indices]

            # indiquer que la requête a été un succès
            data["success"] = True

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
