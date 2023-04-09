from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = int(request.form['outlet_size']) # corrected data type
    outlet_location_type = int(request.form['outlet_location_type']) # corrected data type
    outlet_type = int(request.form['outlet_type']) # corrected data type

    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    scaler_path= r'H:\Personal 2\Mini proj 222\BigMart-Sales-Prediction-With-Deployment-main\models\sc.sav'
    sc = joblib.load(scaler_path)
    X_std = sc.transform(X)

    model_path=r'H:\Personal 2\Mini proj 222\BigMart-Sales-Prediction-With-Deployment-main\models\lr.sav'
    model = joblib.load(model_path)
    Y_pred = model.predict(X_std)

    # Get the user-selected options
    item_fat_content_options = {1: 'Low Fat', 2: 'Regular',3:'High Fat'}
    item_fat_content_text = item_fat_content_options.get(int(item_fat_content), 'Unknown')

    item_type_options = {1: 'Baking Goods', 2: 'Breads', 3: 'Breakfast', 4: 'Canned', 5: 'Dairy', 6: 'Frozen Foods',
                         7: 'Fruits and Vegetables', 8: 'Hard Drinks', 9: 'Health and Hygiene', 10: 'Household',
                         11: 'Meat', 12: 'Others', 13: 'Seafood', 14 :'Snack Foods',15:'Soft Drinks',16:'Starchy Foods'}
    item_type_text = item_type_options.get(int(item_type), 'Unknown')

    outlet_size_options = {1: 'High', 2: 'Medium', 3: 'Small'}
    outlet_size_text = outlet_size_options.get(outlet_size,'Unknown')

    outlet_location_type_options = {1: 'Tier 1', 2: 'Tier 2', 3: 'Tier 3'}
    outlet_location_type_text = outlet_location_type_options.get(outlet_location_type, 'Unknown')

    outlet_type_options = {1: 'Grocery Store', 2: 'Supermarket Type1', 3: 'Supermarket Type2', 4: 'Supermarket Type3'}
    outlet_type_text = outlet_type_options.get(outlet_type, 'Unknown')

    # Render the output page with the predicted value and user-selected options
    return render_template('result.html', prediction=int(Y_pred),
                           item_weight=item_weight, item_fat_content=item_fat_content_text,
                           item_visibility=item_visibility, item_type=item_type_text,
                           item_mrp=item_mrp, outlet_establishment_year=outlet_establishment_year,
                           outlet_size=outlet_size_text, outlet_location_type=outlet_location_type_text,
                           outlet_type=outlet_type_text)

if __name__ == "__main__":
    app.run(debug=True, port=9457)
