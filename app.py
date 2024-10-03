from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import numpy as np
import pandas as pd

Heart_model_path = r"Heart_disease_model"
diabetes_model_path = r"diabetes_disease_model"
Cancer_model_path = r"Cancer_disease_model"
Brain_stroke_model_path = r"Brain_stroke_model"
Kidney_model_path = r"Kidney_disease_model"

Heart_disease_model = load(Heart_model_path)
Cancer_disease_model = load(Cancer_model_path)
Brain_stroke_model = load(Brain_stroke_model_path)
diabetes_disease_model = load(diabetes_model_path)
Kidney_disease_model = load(Kidney_model_path)

app = Flask(__name__)
CORS(app)


@app.route('/Heart_disease_model', methods=['GET', 'POST'])
def get_heart_disease_model_columns():
    if request.method == 'POST':
        data = request.json

        def Binary_data_helper(loaded_binary_data, key):
            feature = list(loaded_binary_data[key])
            new_feature = []
            for value in feature:
                new_feature.append(str(key+'_'+str(value)))
            return new_feature
        loaded_column_order = Heart_disease_model['Columns_order']
        loaded_encoded_dict = Heart_disease_model['encoded_dict']
        loaded_binary_data = Heart_disease_model['Binary_column']

        for key, value in dict(data).items():
            if key in loaded_encoded_dict:
                data[key] = loaded_encoded_dict[key][value]

        new_feature_data = {}

        for key, value in dict(data).items():
            if key in loaded_binary_data:
                for i in Binary_data_helper(loaded_binary_data, key):
                    if str(i) == str(key+'_'+str(float(value))) or str(i) == str(key+'_'+str(int(value))):
                        new_feature_data[i] = 1
                    else:
                        new_feature_data[i] = 0
                data.pop(key)
        data.update(new_feature_data)
        Final_data = {key: data[key] for key in loaded_column_order}
        df = pd.DataFrame([Final_data])
        result = Heart_disease_model['model'].predict(df)
        if result == 0:
            return jsonify({"result": "You have a low risk of heart disease"}), 200
        else:
            return jsonify({"result": "You have a higher risk of heart disease"}), 200

    Binary_column = Heart_disease_model.get('Binary_column', {})
    for key, value in Binary_column.items():
        Binary_column[key] = [int(v) if isinstance(
            v, np.int64) else v for v in value]

    encoded_dict = Heart_disease_model['encoded_dict']
    Columns_order = Heart_disease_model['Columns_order']

    Binary_column_updated = {}
    for key, value in Binary_column.items():
        if key not in encoded_dict:
            Binary_column_updated[key] = sorted(list(value))
    Columns_updated = []
    for i in Heart_disease_model['Columns']:
        if i not in Binary_column_updated.keys() and i not in encoded_dict.keys():
            Columns_updated.append(i)
    Updated_encoded_dict = {}
    for key, value in encoded_dict.items():
        Updated_encoded_dict[key] = list(value.keys())
    return jsonify({"columns": Columns_updated, "Binary_column": Binary_column_updated, "categorical_columns": Updated_encoded_dict})


@app.route('/Cancer_disease_model', methods=['GET', 'POST'])
def get_cancer_disease_model_columns():
    if request.method == 'POST':
        # Get data from the frontend
        data = request.json
        encoded_dict = {'Gender': {'Male': 1, 'Female': 2}}
        for key, value in dict(data).items():
            if key in encoded_dict:
                data[key] = encoded_dict[key][value]
        for key, value in dict(data).items():
            data[key] = int(value)

        loaded_column_order = Cancer_disease_model['Columns_order']
        Final_data = {key: data[key] for key in loaded_column_order}
        df = pd.DataFrame([Final_data])
        result = Cancer_disease_model['model'].predict(df)
        print(result)
        if result == 'Low':
            return jsonify({"result": "You have a low risk of Cancer"}), 200
        elif result == 'Medium':
            return jsonify({"result": "You have a medium risk of Cancer"}), 200
        else:
            return jsonify({"result": "You have a higher risk of Cancer"}), 200

    Columns = {}
    for key, value in Cancer_disease_model['Columns'].items():
        if key != 'Age' and key != 'Gender':
            Columns[key] = sorted(
                [int(v) if isinstance(v, np.int64) else v for v in value])
    try:
        Columns.pop('Gender')
    except Exception:
        pass
    encoded_dict = {'Gender': {'Male': 1, 'Female': 2}}
    Updated_encoded_dict = {}
    for key, value in encoded_dict.items():
        Updated_encoded_dict[key] = list(value.keys())
    return jsonify({"Binary_column": Columns, "columns": ['Age'], 'categorical_columns': Updated_encoded_dict})


@app.route('/Brain_stroke_model', methods=['GET', 'POST'])
def get_brain_stroke_model_columns():
    if request.method == 'POST':
        # Get data from the frontend
        data = request.json

        def Binary_data_helper(loaded_binary_data, key):
            feature = list(loaded_binary_data[key])
            new_feature = []
            for value in feature:
                new_feature.append(str(key+'_'+str(value)))
            return new_feature
        loaded_column_order = Brain_stroke_model['Columns_order']
        loaded_encoded_dict = Brain_stroke_model['encoded_dict']
        loaded_binary_data = Brain_stroke_model['Binary_column']

        for key, value in dict(data).items():
            if key in loaded_encoded_dict:
                data[key] = loaded_encoded_dict[key][value]

        new_feature_data = {}
        for key, value in dict(data).items():
            if key in loaded_binary_data:
                for i in Binary_data_helper(loaded_binary_data, key):
                    if str(i) == str(key+'_'+str(float(value))) or str(i) == str(key+'_'+str(int(value))):
                        new_feature_data[i] = 1
                    else:
                        new_feature_data[i] = 0
                data.pop(key)
        data.update(new_feature_data)
        Final_data = {key: data[key] for key in loaded_column_order}
        df = pd.DataFrame([Final_data])
        result = Brain_stroke_model['model'].predict(df)
        if result == 0:
            return jsonify({"result": "You have a low risk of Brain Stroke"}), 200
        else:
            return jsonify({"result": "You have a higher risk of Brain_stroke"}), 200

    Binary_column = Brain_stroke_model['Binary_column']
    encoded_dict = Brain_stroke_model['encoded_dict']
    Columns_order = Brain_stroke_model['Columns_order']

    Binary_column_updated = {}
    for key, value in Binary_column.items():
        if key not in encoded_dict:
            Binary_column_updated[key] = value
    Columns_updated = []
    for i in Brain_stroke_model['Columns']:
        if i not in Binary_column_updated.keys() and i not in encoded_dict.keys():
            Columns_updated.append(i)
    Updated_encoded_dict = {}
    for key, value in encoded_dict.items():
        Updated_encoded_dict[key] = list(value.keys())
    return jsonify({"columns": Columns_updated, "Binary_column": Binary_column_updated, "categorical_columns": Updated_encoded_dict})


@app.route('/diabetes_disease_model', methods=['GET', 'POST'])
def get_diabetes_disease_model_columns():
    if request.method == 'POST':
        data = request.json

        def Binary_data_helper(loaded_binary_data, key):
            feature = list(loaded_binary_data[key])
            new_feature = []
            for value in feature:
                new_feature.append(str(key+'_'+str(value)))
            return new_feature
        loaded_column_order = diabetes_disease_model['Columns_order']
        loaded_encoded_dict = {
            "Age":
            {'18-24': 1,
             '25-29': 2,
             '30-34': 3,
             '35-39': 4,
             '40-44': 5,
             '45-49': 6,
             '50-54': 7,
             '55-59': 8,
             '60-64': 9,
             '65-69': 10,
             '70-74': 11,
             '75-79': 12,
             '80 or older': 13},
            'Education': {
                "Early Childhood Education": 1,
                "Primary Education": 2,
                "Lower Secondary Education": 3,
                "Upper Secondary Education": 4,
                "Tertiary Education": 5,
                "Adult Education": 6
            }
        }
        loaded_binary_data = diabetes_disease_model['Binary_column']

        for key, value in dict(data).items():
            if key in loaded_encoded_dict:
                data[key] = loaded_encoded_dict[key][value]

        Final_data = {key: data[key] for key in loaded_column_order}
        df = pd.DataFrame([Final_data])
        result = diabetes_disease_model['model'].predict(df)
        if result == 0:
            return jsonify({"result": "You have a low risk of diabeties"}), 200
        else:
            return jsonify({"result": "You have a higher risk of diabeties"}), 200

    Binary_column = diabetes_disease_model.get('Binary_column', {})
    for key, value in Binary_column.items():
        Binary_column[key] = [int(v) if isinstance(
            v, np.int64) else v for v in value]

    Columns_order = diabetes_disease_model['Columns_order']
    encoded_dict = {
        "Age":
            {'18-24': 1,
             '25-29': 2,
             '30-34': 3,
             '35-39': 4,
             '40-44': 5,
             '45-49': 6,
             '50-54': 7,
             '55-59': 8,
             '60-64': 9,
             '65-69': 10,
             '70-74': 11,
             '75-79': 12,
             '80 or older': 13},
        'Education': {
                "Early Childhood Education": 1,
                "Primary Education": 2,
                "Lower Secondary Education": 3,
                "Upper Secondary Education": 4,
                "Tertiary Education": 5,
                "Adult Education": 6
            }
    }
    try:
        Binary_column.pop('Age')
        Binary_column.pop('Education')
    except Exception:
        pass
    Binary_column_updated = {}
    for key, value in Binary_column.items():
        if key not in encoded_dict:
            Binary_column_updated[key] = value
    Columns_updated = []
    for i in diabetes_disease_model['Columns']:
        if i not in Binary_column_updated.keys() and i not in encoded_dict.keys():
            Columns_updated.append(i)
    Updated_encoded_dict = {}
    for key, value in encoded_dict.items():
        Updated_encoded_dict[key] = list(value.keys())

    return jsonify({"columns": Columns_updated, "Binary_column": Binary_column_updated, "categorical_columns": Updated_encoded_dict})


@app.route('/Kidney_disease_model', methods=['GET', 'POST'])
def get_Kidney_disease_model_columns():
    if request.method == 'POST':
        data = request.json

        def Binary_data_helper(loaded_binary_data, key):
            feature = list(loaded_binary_data[key])
            new_feature = []
            for value in feature:
                new_feature.append(str(key+'_'+str(value)))
            return new_feature
        loaded_column_order = Kidney_disease_model['Columns_order']
        loaded_encoded_dict = Kidney_disease_model['encoded_dict']
        loaded_binary_data = Kidney_disease_model['Binary_column']

        for key, value in dict(data).items():
            if key in loaded_encoded_dict:
                data[key] = loaded_encoded_dict[key][value]

        new_feature_data = {}

        for key, value in dict(data).items():
            if key in loaded_binary_data:
                print(value, type(value))
                for i in Binary_data_helper(loaded_binary_data, key):
                    try:
                        if str(i) == str(key+'_'+str(float(value))):
                            new_feature_data[i] = 1
                        else:
                            new_feature_data[i] = 0
                    except ValueError:
                        if str(i) == str(key+'_'+str(float(value))):
                            new_feature_data[i] = 1
                        else:
                            new_feature_data[i] = 0

                data.pop(key)
        data.update(new_feature_data)
        Final_data = {key: data[key] for key in loaded_column_order}
        df = pd.DataFrame([Final_data])
        result = Kidney_disease_model['model'].predict(df)
        print(result)
        if result == 0:
            return jsonify({"result": "You have a low risk of Kidney disease"}), 200
        else:
            return jsonify({"result": "You have a higher risk of Kidney disease"}), 200

    Binary_column = Kidney_disease_model.get('Binary_column', {})

    for key, value in Binary_column.items():
        Binary_column[key] = [int(v) if isinstance(
            v, np.int64) else v for v in value]

    encoded_dict = Kidney_disease_model['encoded_dict']

    Columns_order = Kidney_disease_model['Columns_order']
    try:
        encoded_dict.pop("classification")
        Binary_column.pop('Age')
    except Exception:
        pass
    Binary_column_updated = {}
    for key, value in Binary_column.items():
        if key not in encoded_dict:
            Binary_column_updated[key] = value
    Columns_updated = []
    for i in Kidney_disease_model['Columns']:
        if i not in Binary_column_updated.keys() and i not in encoded_dict.keys():
            Columns_updated.append(i)
    Updated_encoded_dict = {}
    for key, value in encoded_dict.items():
        Updated_encoded_dict[key] = list(value.keys())
    return jsonify({"columns": Columns_updated, "Binary_column": Binary_column_updated, "categorical_columns": Updated_encoded_dict})


if __name__ == '__main__':
    app.run(debug=True)
