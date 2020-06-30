# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-desease-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('Welcome_index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['Gender'])
        cp = int(request.form['Chest Pain'])
        trestbps = int(request.form['Resting Blood pressure'])
        chol = int(request.form['Cholestrol'])
        fbs = int(request.form['Blood Sugar'])
        restecg = int(request.form['restecg'])
        talach = int(request.form['Max Heart rate'])
        exang = int(request.form['Exang'])
        oldpeak = float(request.form['Old peak'])
        slope = int(request.form['slope'])
        ca = int(request.form['Ca'])
        thal = int(request.form['thal'])

        
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, talach, exang, oldpeak, slope, ca, thal]])
        my_prediction = classifier.predict(data)
        
        return render_template('Results.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)