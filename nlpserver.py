

from flask import Flask, render_template, request

from NLP_review2_code_search_copy import* #this fill contains the search function that we are calling 


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('nlpindex.html')

@app.route('/sametext', methods=['POST'])
def hello():
    input1 = request.form['input1']
    return search(input1)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3000)