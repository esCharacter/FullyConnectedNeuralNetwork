import json
import datetime
import os

def modelJsonOutput(name,weight_list,basis_list,x_max,x_min,y_max,y_min):
    weights_basis = {}

    idx_w = 1
    for w in weight_list :
        lable = 'Weight' + str(idx_w)
        weights_basis[lable] = w
        idx_w+=1

    idx_b = 1
    for b in basis_list :
        lable = 'basis' + str(idx_b)
        weights_basis[lable] = b
        idx_b+=1

    weights_basis['x_max'] = x_max
    weights_basis['x_min'] = x_min
    weights_basis['y_max'] = y_max
    weights_basis['y_min'] = y_min

    rootJson = json.dumps(weights_basis)

    if os.path.exists("modelJson") == False :
        os.mkdir("modelJson")

    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S')  
    json_name = name + now_str + '.json'

    jsonWrite = open('modelJson/' + json_name, 'w')

    jsonWrite.write(rootJson)

    print("Model Output Complete !")
    print()
    #print("Model Output : ")
    #print(rootJson)
'''
inputNeuron = 3
hideNeuron = 2
outNeuron = 1

w1 = [[1,2],[3,4],[5,6]]
b1 = [[21],[22]]
w2 = [[7,8],[9,10]]
b2 = [[23],[24]]
w3 = [[11],[12]]
b3 = [[25]]
x_max = [0.1,0.2,0.3]
x_min = [-0.1,-0.2,-0.3]
y_max = [100]
y_min = [0]
wlist = []
wlist.append(w1)
wlist.append(w2)
wlist.append(w3)
blist = []
blist.append(b1)
blist.append(b2)
blist.append(b3)

modelJsonOutput(wlist,blist,x_max,x_min,y_max,y_min)
'''