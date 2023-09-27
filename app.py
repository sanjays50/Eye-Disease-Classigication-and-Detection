from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)
from tensorflow.keras.layers import Layer


model = load_model('models/final_model.h5')
# model =  load_model('enmodel_keras.h5')
target_img = os.path.join(os.getcwd() , 'static/images')

# @app.route('/')
# def index_view():
#     return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):

    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html', readImg = '0')
    if request.method == 'POST':
        file = request.files['filename']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            # img = read_image(file_path)
            # class_prediction=model.predict(img) 

            image = load_img(
              file_path, target_size = (224, 224))
            input_arr = img_to_array(image)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            prediction = model(input_arr)
            classes_x = np.argmax(prediction, axis=1)
            pred = prediction.numpy()
            pred = pred.flatten()
            np.set_printoptions(precision=4)
            pred = np.around(pred,3)
            labs = ['Cataract', 'Glaucoma', 'Dia_Ret', 'Normal']
            if classes_x == 0:
              diseases = "Catarect"
            elif classes_x == 2:
              diseases = "Glaucoma"
            elif classes_x == 1:
              diseases = "Diabetic_Reitnopathy"
            else :
              diseases = "Normal"
            return render_template('index.html', readImg = '1', diseases = diseases,prob=pred, user_image = file_path, label = labs)
        else:
            return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True,use_reloader=True, port=8000)