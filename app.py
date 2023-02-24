from flask import Flask, request, Response
import pickle
import pandas as pd
from insurance_all.insurance_all import InsuranceAll
import json
model_pipeline = pickle.load(open('models/lgbm_tuned_pipe.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predictions', methods=['POST'])

def predict():

        json_d = request.get_json()

        if json_d:

            if isinstance(json_d, dict):
                data = pd.DataFrame(json_d, index=[0])

            else:
                data = pd.DataFrame(json_d, columns=json_d[0].keys())

            pipe = InsuranceAll()

            df1 = pipe.feature_engineering(data)

            predictions = model_pipeline.predict_proba(df1)

            df_preds = pd.concat([df1.reset_index(drop=True), pd.Series(predictions[:, 1])], axis=1)
            df_preds = df_preds.rename(columns={0:'proba_predictions'})

            return json.dumps(df_preds.to_dict(orient='records'))
        else:
            return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':

    PORT = 80
    app.run(host='0.0.0.0', port=PORT, debug=True)
