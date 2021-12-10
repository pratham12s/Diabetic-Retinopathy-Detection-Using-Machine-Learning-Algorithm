from flask import Flask, render_template, url_for, redirect, flash, request
app = Flask(__name__)
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField, FileAllowed
import tensorflow_addons as tfa
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import os
import secrets
from PIL import Image
import cv2
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

import tensorflow as tf
import numpy as np


MODEL_PATH = r'C:\Users\MY NOTEBOOK\Documents\Artificial Intelligence\AI PROJECT\DR_Detection\models\trained_model\Standard_CNN_Updated1.h5'
model = tf.keras.models.load_model(MODEL_PATH)

from models.gradcam import GradCAM

class UploadForm(FlaskForm):
    picture = FileField('', validators = [FileAllowed(['jpg','png','jpeg'])])
    submit = SubmitField('Upload')

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/pics', picture_fn)
    output_size = (224, 224)
    i = Image.open(form_picture)
    i = i.resize(output_size, Image.ANTIALIAS)
    i.save(picture_path)
    return picture_fn
    
def model_predict(filename, model):
    image = cv2.imread('static/pics/'+filename)
    image1 = cv2.resize(image,(224,224))
    image1= tf.image.adjust_contrast(image1, 0.6)
    image1 = tfa.image.equalize(image1)
    image1 = image1.astype('float32') / 255.0
    image1 = np.expand_dims(image1, axis=0)
    print("shape:", image1.shape)
    preds = model.predict(image1)
    i = np.argmax(preds[0])
    icam = GradCAM(model, i,None,None)
    heatmap = icam.compute_heatmap(image1)
    heatmap = cv2.resize(heatmap, (224, 224))
    image = np.squeeze(image)
    image = np.asarray(image, np.uint8)
    (heatmap,output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)
    print(heatmap.shape, image.shape)
    cv2.imwrite('static/pics/heatmap_'+filename, heatmap)
    cv2.imwrite('static/pics/heatmap_overlay_'+filename, output)
    return preds
    

@app.route("/", methods=['GET','POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
        return redirect(url_for('prediction', filename = picture_file))
    filename = 'default.png'
    return render_template('home.html', filename = filename, form = form)

@app.route("/prediction")
def prediction():
    filename = request.args['filename']
    preds = model_predict(filename, model)
    pred_prob = "{:.3f}".format(np.amax(preds))
    pred_class = np.argmax(np.squeeze(preds))
    diagnosis = ["No DR", "DR"]
    result = diagnosis[pred_class]
    return render_template('prediction.html',filename = filename, result = result, prob = pred_prob)

@app.route('/display/<filename>')
def display_image(filename = 'default.png'):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='pics/' + filename), code=301)
    
if __name__ == '__main__':
    app.run(debug=True)
