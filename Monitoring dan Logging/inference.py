import requests
import json

# INI SUDAH DIPERBAIKI: Alamat lengkap ke model kamu
url = "http://127.0.0.1:1234/invocations"

data = {
    "dataframe_split": {
        "columns": ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked_Q","Embarked_S"],
        "data": [[3, 0, 22, 1, 0, 7.25, 0, 1]]
    }
}

try:
    # Mengirim data ke model
    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
    
    # Menampilkan hasil jawaban dari model
    print("HASIL PREDIKSI MODEL:", response.json())
except Exception as e:
    print("Waduh, masih gagal! Pastikan terminal 'mlflow models serve' masih jalan.")
    print(f"Error detail: {e}")
