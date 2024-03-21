from flask import Flask, render_template, request
from utility import getSentiment


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        corpus = request.form['text']
        if corpus:
            sentiment = getSentiment(corpus)
            if sentiment == 1 and corpus != '':
                result = "Positive"
            else:
                result = "Negative"
            return render_template('result.html', result=result)
        else:
            return render_template('index.html', result="Please enter text for sentiment analysis.")


@app.route("/index")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True,port=5001,host='0.0.0.0')
