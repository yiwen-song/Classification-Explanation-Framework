import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
from urllib.parse import unquote
UPLOAD_FOLDER = './tmp'

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


@app.route('/model1')
def model1():
    print ("=================================model1=================================")
    return redirect("/#model")



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
            return redirect("/#testglobal")
    return render_template("index.html")


@app.route('/predict_global')
def predict():
    print ("=================================predict=================================")
    return send_file("./tmp/global_res.html")


@app.route('/random_one')
def predict_lcoal():
    print ("=================================predict=================================")
    return send_file("./tmp/local_res.html")

@app.route('/your_sentence/<sentence>')
def your_sentence(sentence):
    print ("=================================predict \"%s\"================================="%sentence)
    return send_file("./tmp/local_res.html")



# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
    