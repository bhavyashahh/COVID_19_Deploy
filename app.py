from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
#import tensorflow as tf
import os
import importlib
#from tensorflow.keras.layers import Input
#from tensorflow.keras.layers.core import Dense
#from tensorflow.keras.models import Model
#import keras
from keras.models import load_model
import cv2
import numpy
from keras import backend as K
import os
#from tensorflow.python.keras.backend import set_session
#from tensorflow.python.keras.models import load_model
#import tensorflow


#tf.compat.v1.disable_eager_execution()
	
#config = tf.ConfigProto(
   # device_count={'CPU': 1},
  #  intra_op_parallelism_threads=1,
 #   allow_soft_placement=True
#)


#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.6

#session = tf.Session(config=config)

#keras.backend.set_session(session)



#session = keras.backend.get_session()
#init = tf.global_variables_initializer()
#session.run(init)

app = Flask(__name__)
#K.clear_session()

#model= tf.keras.models.load_model(r'/home/bhavyashah/Desktop/corona_deploy/acc_65')

@app.route('/')
def home():
	return render_template("corona_form.html")

@app.route('/out', methods=['POST'])
#global graph
#with graph.as_default():
def about():
	#K.clear_session()	
	model= load_model(r'acc_65')
	fname=request.form['fname']
	lname=request.form['lname']
	age=request.form['age']
	gender=request.form['gen']
	fever=request.form['fever']
	cough=request.form['cough']
	breathe=request.form['breathe']
	pain=request.form['pain']
	#img=request.form['img']
	#filestr = request.files['file'].read()
	fi=request.files['img']	
	path = 'static/images/'
	fi.save(os.path.join(path, fi.filename))
	img=fi.filename	
	f=request.files['img'].read()	
	npimg = numpy.fromstring(f, numpy.uint8)
	image = cv2.imread("static/images/"+fi.filename)	
	#image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
	#image = cv2.imread(f)
	image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
	img_orig = image
	image = np.expand_dims(image,axis=0)
	image = image/255.0
	#print(fever)
	#print(gender)
	#print(f)
	#print(type(image))
	prediction = model.predict(image) 
	target_class = np.argmax(prediction[0])
	target_class0 = round(prediction[0][0]*100,2)
	target_class1 = round(prediction[0][1]*100,2)
	target_class2 = round(prediction[0][2]*100,2)
			
	#if target_class==0:
	#	output="Normal"
	#elif target_class==1:
	#	output="Pneumonia"
	#elif target_class==2:
	#	output="COVID-19"
	
	last_conv = model.get_layer('conv2d_6')
	grads = K.gradients(model.output[:,target_class],last_conv.output)[0]
	pooled_grads = K.mean(grads,axis=(0,1,2))
	iterate = K.function([model.input],[pooled_grads,last_conv.output[0]])
	pooled_grads_value,conv_layer_output = iterate([image])
	for i in range(512):
		conv_layer_output[:,:,i] *= pooled_grads_value[i]
    
	heatmap = np.mean(conv_layer_output,axis=-1)
	for x in range(heatmap.shape[0]):
		for y in range(heatmap.shape[1]):
			heatmap[x,y] = np.max(heatmap[x,y],0)
	heatmap = np.maximum(heatmap,0)
	heatmap /= np.max(heatmap)
	img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
	upsample = cv2.resize(heatmap, (224,224))
	
	heatmap = upsample * img_gray	
	path = 'static/maps/'
	cv2.imwrite(os.path.join(path , fi.filename), heatmap)
	
	
	return render_template("out.html",fname=fname,lname=lname,age=age,gender=gender,fever=fever,cough=cough,breathe=breathe,pain=pain,img=img,predict0=target_class0,predict1=target_class1,predict2=target_class2)




if __name__ == "__main__":
	app.debug=True
	app.run()
	
