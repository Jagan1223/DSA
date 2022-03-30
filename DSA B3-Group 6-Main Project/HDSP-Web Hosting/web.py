from flask import Flask,render_template,request
import pickle
import numpy as np
import matplotlib.style as stl
import seaborn as sns
import sklearn as skl
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
   a= float(request.values['aa'])
   b= float(request.values['ab'])
   c= float(request.values['ac'])
   d= float(request.values['ad'])
   e= float(request.values['ae'])
   f= e ##Data Value Alt
   g= 23.88 ##Confidence_Limit_Low
   h= 31.26 ##Confidence_Limit_High
   i= float(request.values['ai'])
   j= float(request.values['aj'])

   make_pred=[[a,b,c,d,e,f,g,h,i,j]]
   output=model.predict(make_pred)
   output=output.item()
   if (output == 1):
       output="The patient is not likely to have Cardiovascular Disease "
   else:
       output="The patient is likely to have Cardiovascular Disease "
       
   #output=round(output,2)
   return render_template ('result.html',prediction_text="{}".format(output))
if __name__=='__main__':
    app.run(port=8000)

