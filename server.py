#importing libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

#load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
	#get the data from the post request
    data = request.get_json(force=True)

    #Make prediction using model loaded from disk as per the data
    prediction = model.predict([[np.array(dataframe[['Beat','Crime_Date_Day','Crime_month','Year','DayOT']])]])
    
    #taking the first value of prediction
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)  