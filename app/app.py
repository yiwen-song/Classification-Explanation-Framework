import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, send_file, jsonify
from werkzeug.utils import secure_filename
from urllib.parse import unquote
import json
from classify import RandomForestClassifier, LogisticRegressionClassifier, SVMClassifier, MyClassifier
UPLOAD_FOLDER = './tmp'
models = ["Random Forest", "Logistic Regression", "SVM"]
model_name = None
model = None


app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r



@app.route('/')
def upload_file():
    return render_template("index.html")


@app.route('/upload_train', methods=['GET', 'POST'])
def upload_train():
    # file upload
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Please select a file!','miss_train_file')
            return redirect("/#data")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file:
            filename = "train.tsv"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect("/#model")
    return render_template("index.html")


@app.route('/model/<num>')
def model_select(num):
    num = int(num)
    print ("=================================Model \"%d\"================================="%num)
    model_name = models[num]
    global model
    if num == 0:
        model = RandomForestClassifier()
        print ("=================================random forest=================================")
    elif num == 1:
        model = LogisticRegressionClassifier()
    else:
        model = SVMClassifier()
    return "<h3>Your select Model \"%s\" </h3>"%model_name


@app.route('/train/<num>')
def train(num):
    num = int(num)
    if num == -1:
        return "<h3>Please select a model.</h3>"
    print ("=================================Training \"%d\"================================="%num)
    model.upload_train_file("./tmp/train.tsv")
    model.train_model()
    return "<h3>Training finished. </h3>"


@app.route('/upload_test', methods=['GET', 'POST'])
def upload_test():
    # file upload
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Please select a file!', 'miss_test_file')
            return redirect("/#testglobal")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file:
            filename = "test.tsv"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            model.upload_test_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect("/#testglobal")
    return render_template("index.html")


@app.route('/predict_global')
def predict_global():
    print ("=================================predict global=================================")
    groundcolorsrc = ["rgba(255, 99, 132, 0.2)", "rgba(255, 159, 64, 0.2)",
                      "rgba(255, 205, 86, 0.2)", "rgba(75, 192, 192, 0.2)", "rgba(54, 162, 235, 0.2)",
                      "rgba(153, 102, 255, 0.2)", "rgba(201, 203, 207, 0.2)"]
    bordercolorsrc = ["rgb(255, 99, 132)", "rgb(255, 159, 64)", "rgb(255, 205, 86)",
                      "rgb(75, 192, 192)", "rgb(54, 162, 235)", "rgb(153, 102, 255)", "rgb(201, 203, 207)"]
    datas,labels = model.get_top_features()
    print (datas, labels)
    model.predict_test()
    # labels = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Grey", "again"]
    # datas = [22, 33, -55, 12, 86, -23, 14, 100]
    backgroundColor = [groundcolorsrc[i % 7] for i in range(len(labels))]
    borderColor = [bordercolorsrc[i % 7] for i in range(len(labels))]
    resp =  {
                "type": "horizontalBar",
                "data": {
                  "labels": labels,
                  "datasets": [{
                    "label": "Top features",
                    "data": datas,
                    "backgroundColor": backgroundColor,
                    "borderColor": borderColor,
                    "borderWidth": 1
                  }]
                },
                "options": {
                    "scales": {
                        "xAxes": [{
                        "ticks": {
                            "beginAtZero": True
                        }
                        }]
                    }
                }
              }
    return json.dumps(resp)


@app.route('/predict_global_static')
def predict_global_():
    labels, data = model.predict_test()
    resp = "<table class=\"table\">\
                <thead>\
                    <tr>\
                    <th scope=\"col\">Label</th>\
                    <th scope=\"col\">Precision</th>\
                    <th scope=\"col\">Recall</th>\
                    <th scope=\"col\">F1 Score</th>\
                    </tr>\
                </thead>\
                <tbody>\
                    <tr>\
                    <th scope=\"row\">%s</th>\
                    <td>%.4f</td>\
                    <td>%.4f</td>\
                    <td>%.4f</td>\
                    </tr>\
                    <tr>\
                    <th scope=\"row\">%s</th>\
                    <td>%.4f</td>\
                    <td>%.4f</td>\
                    <td>%.4f</td>\
                    </tr>\
                </tbody>\
                </table>"%(labels[0], data[0][0], data[0][1], data[0][2],labels[1], data[1][0], data[1][1], data[1][2])
    return resp


@app.route('/predict_global_roc')
def predict_global_roc():
    # TODO: generate labels and data
    labels = ["January", "February", "March", "April", "May", "June", "July"]
    data = [1, 1, 1, 1, 1, 1, 1]
    resp  = {
            "type": 'line',
            "data": {
              "labels": labels,
              "datasets": [{
                  "label": "ROC curve",
                  "data": data,
                  "backgroundColor": [
                    'rgba(105, 0, 132, .2)',
                  ],
                  "borderColor": [
                    'rgba(200, 99, 132, .7)',
                  ],
                  "borderWidth": 2
                },
                {
                  "label": "Y = X",
                  "data": [i * 1.0 / (len(labels) - 1) for i in range(len(labels))],
                  "backgroundColor": [
                    'rgba(0, 137, 132, .2)',
                  ],
                  "borderColor": [
                    'rgba(0, 10, 130, .7)',
                  ],
                  "borderWidth": 2
                }
              ]
            }
          }
    return json.dumps(resp)


@app.route('/random_one')
def predict_lcoal():
    print ("=================================predict random=================================")
    len_of_test = len(model.data.test_labels)
    import random
    idx = random.randint(0,len_of_test)
    model.explain_indexed_test_sample(idx)
    return send_file("./tmp/local_res.html")


@app.route('/your_idx/<idx>')
def predict_index(idx):
    idx = int(idx)
    print ("=================================predict %d================================="%idx)
    model.explain_indexed_test_sample(idx)
    return send_file("./tmp/local_res.html")

@app.route('/your_sentence/<sentence>')
def your_sentence(sentence):
    print ("=================================predict \"%s\"================================="%sentence)
    model.explain_self_input_sample(sentence)
    return send_file("./tmp/local_res.html")



# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
    