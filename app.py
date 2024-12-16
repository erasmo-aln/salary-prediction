import joblib
import pandas as pd
from flask import Flask, request, render_template


app = Flask(__name__)

model = joblib.load('model/inference_pipeline.pkl')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Age': request.form['Age'],
            'Work Type': request.form['Work Type'],
            'Education': request.form['Education'],
            'experience': request.form['experience'],
            'profession': request.form['profession'],
            'Continent': request.form['Continent'],
        }
    except KeyError as e:
        return render_template('home.html', prediction_text=f'Entrada invalida. Erro: {e}')

    if any(value == '' for value in data.values()):
        return render_template('home.html', prediction_text='Verifique se todos os campos estao preenchidos.')

    formatted_data = pd.DataFrame(data, index=[0])
    formatted_data.columns = ['age', 'work_type', 'education', 'experience_bin', 'profession', 'continent']
    output = round(model.predict(formatted_data)[0], 2)

    return render_template('home.html', prediction_text=f'${output:,} (valor anual)')


if __name__ == "__main__":
    app.run()
