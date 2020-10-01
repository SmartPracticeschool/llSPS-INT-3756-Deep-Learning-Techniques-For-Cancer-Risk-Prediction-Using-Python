from  flask import Flask,request,render_template
import os
import numpy as np
from werkzeug.utils import secure_filename
app=Flask(__name__)
from keras.models import load_model
model=load_model("breastcancer.h5")
from keras.preprocessing import image
import tensorflow as tf
global graph
graph=tf.get_default_graph()

@app.route('/',methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/predict',methods=["GET","POST"])
def uplode():
    if request.method =="POST":
        f=request.files["image"]
        basepath=os.path.dirname(__file__)#to get the path of current file
        #choose the uplaads folder
        file_path=os.path.join(basepath,"uploads",secure_filename(f.filename))
        f.save(file_path)# save the image
        img=image.load_img(file_path,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        with graph.as_default():
            preds=model.predict_classes(x)
        index=["Negative","Positive"]
        text=index[preds[0]]
    return text
if __name__=="__main__":
    app.run(debug=True)