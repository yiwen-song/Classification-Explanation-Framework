import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './tmp'

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def upload_file():
    return render_template("index.html")


@app.route('/upload_train', methods=['GET', 'POST'])
def upload_train():
    # file upload
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect("/#data")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # flash('No selected file')
            return redirect("/#data")
        if file:
            filename = "train.tsv"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect("/#model")
    return render_template("index.html")


@app.route('/upload_test', methods=['GET', 'POST'])
def upload_test():
    # file upload
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect("/#testglobal")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # flash('No selected file')
            return redirect("/#testglobal")
        if file:
            filename = "test.tsv"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect("/#testglobal")
    return render_template("index.html")


@app.route('/predict')
def welcome():
    print ("=================================predict=================================")
    return redirect("/#testglobal")


@app.route('/model1')
def model1():
    print ("=================================model1=================================")
    return redirect("/#model")


# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
    