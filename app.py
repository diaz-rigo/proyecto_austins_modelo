from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Configurar CORS
CORS(app, origins=["https://austins.vercel.app"])  # Reemplaza con tu dominio real en Vercel

# Cargar el modelo entrenado y el escalador
clf = joblib.load('customer_cluster_classifier.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/cluster', methods=['POST'])
def cluster():
    data = request.get_json()
    df = pd.DataFrame(data)

    # Seleccionar las columnas relevantes
    X = df[['total_orders', 'total_amount_spent']]

    # Escalar los datos
    X_scaled = scaler.transform(X)

    # Predecir los clusters
    clusters = clf.predict(X_scaled)

    # Añadir los clusters al DataFrame original
    df['cluster'] = clusters

    # Convertir el DataFrame a diccionario y devolver como JSON
    response = df.to_dict(orient='records')
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
