import numpy 
from flask import Flask, jsonify, render_template, request
import torch
from model import LeNet
from torch.autograd import Variable



# webapp
app = Flask(__name__)

@app.route('/api/mnist', methods=['POST'])
def mnist():
    
    inputs = ((255-numpy.array(request.json, dtype=numpy.uint8))/255.0).reshape(28, 28)
    inputs = torch.from_numpy(inputs).float()
    inputs = inputs.view(1, 1, 28, 28)
    
    use_gpu = torch.cuda.is_available()  
    if use_gpu:
        inputs = inputs.cuda()
    inputs = Variable(inputs)
    #input : (28,28) ndarray
    #output : list 10 double numbers
    model = torch.load('LeNet.pkl')
    
    if use_gpu:
        model = model.cuda()   
    output = model(inputs)
    output= output.data[0].tolist()
    result = []
    for i in range(0,10):
        result.append('%.3f' % (output[i]))
    print(result)
    return jsonify(results=result)
    

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="116.56.143.172", port=8006)