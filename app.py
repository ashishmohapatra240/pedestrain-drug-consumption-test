from flask import Flask, request, jsonify
import pickle
import numpy as np
import json

app = Flask(__name__)

@app.route('/check_drug_formation', methods=['POST'])
def check_drug_formation():
    data = json.loads(request.data)
    loaded_model = pickle.load(open('finalized_model_drugC.pkl', 'rb'))
    dp = np.array(list(data.values())).reshape(1,-1)
    result = loaded_model.predict(dp)
    # result=1
    res = ""
    if result==1:
        res = "yes"
    else:
        res = "no"
    return res

if __name__ == '__main__':
    app.run(debug=True)