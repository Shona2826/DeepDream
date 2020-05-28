import os
from PIL import Image
from flask import Flask
from flask import request,jsonify,render_template,send_from_directory
from Data_Preprocess import *
from deep_simple import *

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def clean_static(target_path):
  for item in os.listdir(target_path):
    if item.endswith('.png'):
      os.remove(os.path.join(target_path,item))

def clean_image_folder(target_path):
  for item in os.listdir(target_path):
    os.remove(os.path.join(target_path,item))

def process_iamge(filename):
  img_location = './images/' + filename
  x = download(img_location,max_dim = 400)
  out = run_deep_dream_simple(img = x,steps = 100,step_size = 0.01 )
  target_path = APP_ROOT + '/' + 'static'
  clean_static(target_path)
  out_filename = filename.replace('.','-')
  out_filename = out_filename + '_out.png'
  if not os.path.isdir(target_path):
    os.mkdir(target_path)
  dest = '/'.join([target_path,out_filename])
  out.save(dest)
  return out_filename

@app.route('/',methods = ['GET','POST'])
def index():
  if request.method == 'POST':
    target = os.path.join(APP_ROOT,'images/')
    clean_image_folder(target)
    if not os.path.isdir(target):
      os.mkdir(target)

    f = request.files['files']
    filename = f.filename
    destination = target + filename
    f.save(destination)
    out_file = process_image(filename)
    print('Deep Dream Image; ',out_file)
    return render_template('index.html',out = out_file)
  else:
    return render_template('index.html')

if __name__ == "__main__":
  app.run(debug=True)

