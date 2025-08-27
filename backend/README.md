Mobile Detector Backend
This is the backend for the Mobile Detector project. It's a simple Flask API that serves a machine learning model to predict the brand and model of a mobile phone from an image.

Setup
Install Python Dependencies:

pip install -r requirements.txt

Get the Model:
You must first train the model by running the train.py script in the model_training/ directory. This will generate mobile_detector_model.h5 and class_labels.txt.

Place the Model and Labels:
Copy the mobile_detector_model.h5 and class_labels.txt files into this backend/ directory.

Running the Server
To start the server, simply run the app.py script:

python app.py

The server will run on http://localhost:5000. It exposes a single endpoint: /predict.