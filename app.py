from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("random_forest_needfactor_model.pkl")  # change if best_model is different

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    area = data.get("area")
    student_count = data.get("student_count")
    book_count = data.get("book_count")

    df = pd.DataFrame([{
        'area': area,
        'student_count': student_count,
        'book_count': book_count
    }])

    prediction = model.predict(df)[0]
    return jsonify({"predicted_need_factor": round(float(prediction), 2)})

if __name__ == '__main__':
    app.run(debug=True)
