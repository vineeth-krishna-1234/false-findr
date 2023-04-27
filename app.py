from flask import Flask, jsonify, request, render_template, redirect, url_for
from prediction_model import PredictionModel
import pandas as pd
from random import randrange
from forms import OriginalTextForm


app = Flask(__name__)

app.config['SECRET_KEY'] = '4c99e0361905b9f941f17729187afdb9'


@app.route("/", methods=['POST', 'GET'])
def home():
    form = OriginalTextForm()

    if form.generate.data:
        data = pd.read_csv("random_dataset.csv")
        index = randrange(0, len(data)-1, 1)
        original_text = data.loc[index].text
        form.original_text.data = str(original_text)
        return render_template('home.html', form=form, output=False)

    elif form.predict.data:
        if len(str(form.original_text.data)) > 10:
            model = PredictionModel(form.original_text.data)
            return render_template('home.html', form=form, output=model.predict())

    return render_template('home.html', form=form, output=False)


@app.route('/predict/<original_text>', methods=['POST', 'GET'])
def predict(original_text):
    model = PredictionModel(original_text)
    return jsonify(model.predict())


@app.route('/random', methods=['GET'])
def random():
    data = pd.read_csv("random_dataset.csv")
    index = randrange(0, len(data)-1, 1)
    return jsonify({'title': data.loc[index].title, 'text': data.loc[index].text, 'label': str(data.loc[index].label)})


if __name__ == '__main__':
    app.run()
