from flask import Flask, request, Response, json
import numpy as numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

#load data
df = pd.read_csv("./StrokeData/stroke.csv", header = None, names=['age', 'heart_disease', 'average_glucose_level'])
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


#create onehotencoder and transform data
onehotencoder = OneHotEncoder()
ohc = onehotencoder.fit(X)
X = ohc.transform(X)

#spliting data into train-test split

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size= 0.25,random_state=555)

#train model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


#create flask instance
app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST'])
def predict(): 
    #get the data from request
    data = request.get_json(force=True)
    requestData = numpy.array([data["age"], data["heart_disease"], data["doors"], data["average_glucose_level"]])
    requestData = numpy.reshape(requestData, (1, -1))
    
    #get onehotencoding for input_data
    requestData = ohc.transform(requestData) 
    
    #Make prediction using model
    prediction = rfc.predict(requestData)
    Accuracy= rfc.score(X_test, y_test)
    # print (Accuracy)
    output = {
        "Prediction": prediction[0],
        "accuracy": Accuracy
        }
    print('output',json.dumps(output))
    return Response(json.dumps(output))

if __name__ == '__main__':
    app.run()
