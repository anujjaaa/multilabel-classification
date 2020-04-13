from flask import Flask, request, render_template
import concat
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('page1.html')

@app.route('/page2')
def subpage1():
    return render_template('page2.html')

# @app.route('/graph1')
# def result():
#     eda.goldmedaldistribution()
#     return render_template('graph1.html',goldmedalFile='/images/goldmedal.png')

if __name__ == '__main__':
    app.run()