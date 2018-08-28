# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SampleData as sd
import time as tm
import modelJson as mj
#import pylab as plb
import random 


###### PARAMETER ######
save_dir = 'model/graph.ckpt'
log_dir = 'tblog/'
data_file = 'trainingData_45_3_25.json'

epochs = 10000
learning_rate = 1.0e-4
batch_size = 200
layer_size = 3

inputNeuron = 460#487#412#367#15367#
hideNeuron = 20#2210#
outNeuron = 3

is_load = False 
is_save = True
is_out_model = False
is_test = False

def printVariable(data_size):
    print("input_neuron : " + str(inputNeuron))
    print("hide_neuron : " + str(hideNeuron))
    print("learning_rate : " + str(learning_rate))
    print("batch_size : " + str(batch_size))
    print("epochs : " + str(epochs))
    print("layer_size : " + str(layer_size))
    print("data_size : " + str(data_size))
    print()

###### FUNCTION ######
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      #
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      #
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      #
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      #
      tf.summary.histogram('histogram', var)

def zeroCentered(X):
    xmean = np.mean(X, axis=0).astype(np.float32)
    X -= xmean
    return X,xmean

def normalization(X):
    xstd = np.std(X, axis=0).astype(np.float32)
    X /= (xstd + 1.0e-10)
    return X,xstd

def normalizationTest(X,mean,std):
    X -= mean
    X /= (std + 1.0e-10)
    return X

def minmaxScaler(X):
    if(len(X) == 0):
        return X,[],[]
    _max = np.max(X, axis=0)
    _min = np.min(X, axis=0)
    X = 2 * (X - _min) / (_max - _min + 1.0e-10)
    X-=1.0
    return X,_max,_min

def minmaxScalerTest(X,_max,_min):
    if(len(X) == 0):
        return X
    X = 2 * (X - _min) / (_max - _min + 1.0e-10)
    X-=1.0
    return X

def Re_minmaxScalerTest(X,_max,_min):
    X += 1.0
    X = (X * (_max - _min + 1.0e-10)) / 2 + _min
    return X

def getDataByFrame(frame,data=[]):
    for sample in data:
        if sample.frameName == frame:
            return sample

def addLayer(inputData,inSize,outSize,Weights,basis,activity_function=None):
        weights_plus_b = tf.matmul(inputData,Weights) + basis
        if activity_function is None:
            ans = weights_plus_b
        else:
            ans = activity_function(weights_plus_b)
        return ans

def getTrainDataByFrame(trainData,numPar):
    x_data = []
    y_data = []
    num_adj = 45
    num_par = numPar

    if numPar == 0 :
        num_par = len(trainData.density)

    num_read = 0
    num_adj_1 = 0
    for pi in range(num_par):
        num_adj_1 = 0
        i_sampleIn = [] #
        i_sampleOut = []
        for ni in range(num_adj):
            neighbori = trainData.neighbor[pi * num_adj + ni] #i adj
            if neighbori != -1 :
                num_adj_1+=1  
                i_sampleIn.append(trainData.force[neighbori * 3 + 0])
                i_sampleIn.append(trainData.force[neighbori * 3 + 1])
                i_sampleIn.append(trainData.force[neighbori * 3 + 2])
                i_sampleIn.append(trainData.velocity[neighbori * 3 + 0])
                i_sampleIn.append(trainData.velocity[neighbori * 3 + 1])
                i_sampleIn.append(trainData.velocity[neighbori * 3 + 2])
                i_sampleIn.append(trainData.position[neighbori * 3 + 0])
                i_sampleIn.append(trainData.position[neighbori * 3 + 1])
                i_sampleIn.append(trainData.position[neighbori * 3 + 2])

                i_sampleIn.append(0.002744)
            else:
                i_sampleIn.append(0)
                i_sampleIn.append(0)
                i_sampleIn.append(0)

                i_sampleIn.append(0)
                i_sampleIn.append(0)
                i_sampleIn.append(0)

                i_sampleIn.append(0)
                i_sampleIn.append(0)
                i_sampleIn.append(0)

                i_sampleIn.append(0)

        #i
        i_sampleIn.append(trainData.force[pi * 3 + 0])
        i_sampleIn.append(trainData.force[pi * 3 + 1])
        i_sampleIn.append(trainData.force[pi * 3 + 2])
        i_sampleIn.append(trainData.velocity[pi * 3 + 0])
        i_sampleIn.append(trainData.velocity[pi * 3 + 1])
        i_sampleIn.append(trainData.velocity[pi * 3 + 2])
        i_sampleIn.append(trainData.position[pi * 3 + 0])
        i_sampleIn.append(trainData.position[pi * 3 + 1])
        i_sampleIn.append(trainData.position[pi * 3 + 2])
        i_sampleIn.append(0.002744)

        #if num_adj_1 <50 :
        #    continue

        #label
        i_sampleOut.append(trainData.correctionPressureForce[pi * 3 + 0])
        i_sampleOut.append(trainData.correctionPressureForce[pi * 3 + 1])
        i_sampleOut.append(trainData.correctionPressureForce[pi * 3 + 2])

        x_data.append(i_sampleIn)
        y_data.append(i_sampleOut)
        num_read+=1

    print("(%d)" % num_read,end=' ')

    return x_data,y_data

def getTrainDataBySample(sampleData,numFrame,numPar,direct,start=0,spacing=0):
    print("--------getTrainDataBySample    numFrame : %d    numParticle : %d" % (numFrame,numPar))
    x_data = []
    y_data = []
    if len(sampleData) == 0:
        return x_data,y_data
    idxSample = start
    iend = start + numFrame
    if direct == 1:
        print("train sample : ",end=' ')
        while idxSample < iend:   
            print(sampleData[idxSample].frameName,end=' ')    
            x_dataFrame,y_dataFrame = getTrainDataByFrame(sampleData[idxSample],numPar)
            x_data.extend(x_dataFrame)
            y_data.extend(y_dataFrame)
            idxSample+=1
            idxSample+=spacing
    else :
        print("test sample : ",end=' ')
        while idxSample < iend:
            print(sampleData[-idxSample - 1].frameName,end=' ') 
            x_dataFrame,y_dataFrame = getTrainDataByFrame(sampleData[-idxSample - 1],numPar)
            x_data.extend(x_dataFrame)
            y_data.extend(y_dataFrame)
            idxSample+=1
            idxSample+=spacing
    print()
    print("x_data size : " + str(len(x_data)))
    print("y_data size : " + str(len(y_data)))

    return x_data,y_data

def getMinibatch(X,Y,batch_size):
    minibatch_x = []
    minibatch_y = []
    clen = batch_size
    copies = len(X) / batch_size
    for i in range(int(copies)):
        minibatch_x.append(X[int(i * clen):int((i + 1) * clen)])
        minibatch_y.append(Y[int(i * clen):int((i + 1) * clen)])
        #print(len(minibatch_x[i]))
    print("copies : " + str(copies))
    print()
    return minibatch_x,minibatch_y

def initdata():
    #x_data = [[1,1,2,3,4,5,6,0,0,0,0]]
    #y_data = [[2]]
    x_data = []
    y_data = []
    points = []
    for x in range(20):    
        for y in range(25):
            points.append([x,y])
    less = 0
    for pi in points:
        xitem = []
        yitem = []
        sumd = 0
        for pj in points:
            dij = np.sqrt((pi[0] - pj[0]) * (pi[0] - pj[0]) + (pi[1] - pj[1]) * (pi[1] - pj[1]))
            if dij <= 1 and len(xitem) < 11:
                sumd+=(dij * 0.5)
                xitem.append(pj[0])
                xitem.append(pj[1])
                xitem.append(0.5)
        if len(xitem) < 15:
            less+=1
        while len(xitem) < 15:
            xitem.append(0)
            xitem.append(0) 
            xitem.append(0.5) 
        yitem.append(sumd)
        x_data.append(xitem)
        y_data.append(yitem)
    print("adj less : %d" % less)
    return x_data,y_data

####### MAIN #######
def main():
    ##get the data from json file
    sampleData = []

    sampleData = sd.getSampleData(data_file)

    #x_data =
    #[[1,1],[1,0],[0,0],[0,1],[1,1],[1,0],[0,0],[0,1],[1,1],[1,0],[0,0],[0,1],[1,1],[1,0],[0,0],[0,1],[1,1],[1,0],[0,0],[0,1]]
    #y_data =
    #[[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1]]

    #data = np.random.rand(500,3)
    #x_data = []
    #y_data = []
    #for x in data :
    #    x_data.append([x[0],x[1]])
    #    y_data.append([x[0] * 2 + x[1] * x[2] + 10 + random.randint(1,10) /
    #    100])
    #x_data,y_data = initdata()
    
    #print(y_data)

    ###### INIT DATA ######
    x_data,y_data = getTrainDataBySample(sampleData,20,10000,1)#Unchanged

    sampleData.clear()

    print()
    print("random data...")
    rand_x_data = []
    rand_y_data = []
    random.seed(16)#Unchanged
    rdatalist = random.sample(range(len(x_data)), len(x_data))
    for idx in rdatalist:
        rand_x_data.append(x_data[idx])
        rand_y_data.append(y_data[idx])

    x_data = rand_x_data
    y_data = rand_y_data
    
    t_size = 0.7
    v_size = t_size + (1 - t_size) / 2
    data_size = len(x_data)
    train_x_data = x_data[0:int(data_size * t_size)]
    train_y_data = y_data[0:int(data_size * t_size)]
    valid_x_data = x_data[int(t_size * data_size):int(data_size * v_size)]
    valid_y_data = y_data[int(t_size * data_size):int(data_size * v_size)]
    test_x_data = x_data[int(v_size * data_size):]
    test_y_data = y_data[int(v_size * data_size):]

    print("train data size : " + str(len(train_x_data)))
    print("valid data size : " + str(len(valid_x_data)))
    print("test data size : " + str(len(test_x_data)))

    #print(train_x_data)
    #print(train_y_data)
    #print(valid_x_data)
    #print(valid_y_data)
    #print(test_x_data)
    #print(test_y_data)
    print()

    ###### NORMALIZATION ######
    #train_x_data,x_mean = zeroCentered(train_x_data)
    #train_x_data,x_std = normalization(train_x_data)
    #train_y_data,y_mean = zeroCentered(train_y_data)
    #train_y_data,y_std = normalization(train_y_data)

    #valid_x_data = normalizationTest(valid_x_data,x_mean,x_std)
    #valid_y_data = normalizationTest(valid_y_data,y_mean,y_std)

    #test_x_data = normalizationTest(test_x_data,x_mean,x_std)
    #test_y_data = normalizationTest(test_y_data,y_mean,y_std)

    train_x_data,x_max,x_min = minmaxScaler(train_x_data)
    train_y_data,y_max,y_min = minmaxScaler(train_y_data)
    valid_x_data = minmaxScalerTest(valid_x_data,x_max,x_min)
    valid_y_data = minmaxScalerTest(valid_y_data,y_max,y_min)
    test_x_data = minmaxScalerTest(test_x_data,x_max,x_min)
    test_y_data = minmaxScalerTest(test_y_data,y_max,y_min)
    #print(x_max)
    #print(y_max)
    print("-----normalization train_x_data len : " + str(len(train_x_data)))
    print("-----normalization train_y_data len : " + str(len(train_y_data)))
    print()
    
    data_size = len(train_x_data)
    printVariable(data_size)

    ###### MINI BATCH ######
    if batch_size != 0 :
        train_x_data,train_y_data = getMinibatch(train_x_data,train_y_data,batch_size)

    xs = tf.placeholder(tf.float32,[None,inputNeuron]) #
    ys = tf.placeholder(tf.float32,[None,outNeuron])

    ###### WEIGHTS BASIS HIDE LOSS GRAPH ######
    Weights1 = weight_variable([inputNeuron,hideNeuron])
    basis1 = bias_variable([1,hideNeuron]) 
    Weights2 = weight_variable([hideNeuron,hideNeuron])
    basis2 = bias_variable([1,hideNeuron])
    Weights3 = weight_variable([hideNeuron,hideNeuron])
    basis3 = bias_variable([1,hideNeuron])
    Weights4 = weight_variable([hideNeuron,outNeuron])
    basis4 = bias_variable([1,outNeuron])

    '''
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, Weights1)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, Weights2)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, Weights4)
    regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / len(train_x_data))
    reg_term = tf.contrib.layers.apply_regularization(regularizer)
    '''
    hidel1 = addLayer(xs,inputNeuron,hideNeuron,Weights1,basis1,activity_function=tf.nn.relu) # relu
    hidel2 = addLayer(hidel1,hideNeuron,hideNeuron,Weights2,basis2,activity_function=tf.nn.relu)
    #hidel3 = addLayer(hidel2,hideNeuron,hideNeuron,Weights3,basis3,activity_function=tf.nn.relu)
    hidel4 = addLayer(hidel2,hideNeuron,outNeuron,Weights4,basis4,activity_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - hidel4)),reduction_indices = [1])) #+ reg_term)
    tf.summary.scalar('loss', loss)
    with tf.name_scope('ActivityHiddenLayer'):
        tf.summary.histogram('HiddenLayer1', hidel1)
        tf.summary.histogram('HiddenLayer2', hidel2)

    with tf.name_scope('Weights1'):
        variable_summaries(Weights1)
    with tf.name_scope('basis1'):
        variable_summaries(basis1)
    with tf.name_scope('Weights2'):
        variable_summaries(Weights2)
    with tf.name_scope('basis2'):
        variable_summaries(basis2)
    with tf.name_scope('Weights3'):
        variable_summaries(Weights3)
    with tf.name_scope('basis3'):
        variable_summaries(basis3)
    with tf.name_scope('Weights4'):
        variable_summaries(Weights4)
    with tf.name_scope('basis4'):
        variable_summaries(basis4)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) #

    init = tf.initialize_all_variables()

    gpuConfig = tf.ConfigProto()
    gpuConfig.gpu_options.allow_growth = True  
    sess = tf.Session(config=gpuConfig)  

    sess.run(init)

    saver = tf.train.Saver()

    ################ LOAD ##################
    if is_load == True :
        saver.restore(sess, save_dir)
        print("graph restore!")
    print()

    #print(sess.run(Weights2))
    #print(sess.run(basis2))

    if is_out_model == True :
        wList = []
        bList = []
        wList.append(sess.run(Weights1).tolist())
        wList.append(sess.run(Weights2).tolist())
        wList.append(sess.run(Weights4).tolist())
        bList.append(sess.run(basis1).tolist())
        bList.append(sess.run(basis2).tolist())
        bList.append(sess.run(basis4).tolist())
        mj.modelJsonOutput('model_',wList,bList,x_max.tolist(),x_min.tolist(),y_max.tolist(),y_min.tolist())

    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    time_start = tm.time()

    ###### TRAIN ######
    feed_data_test = {xs:valid_x_data,ys:valid_y_data}
    pre_valid_loss = 100.0
    for i in range(epochs):
        if batch_size != 0 :#mini
            scopy = int(data_size / batch_size)
            brandom = random.sample(range(scopy), scopy)
            for b in brandom:
                feed_data = {xs:train_x_data[b],ys:train_y_data[b]} #data
                sess.run(train,feed_dict=feed_data)
                if i % 200 == 0 and b % int(scopy / 3) == 0:
                    print(str(i) + " train loss : " + str(sess.run(loss,feed_dict=feed_data)))
                    summary_str = sess.run(merged_summary_op,feed_dict = feed_data)
                    summary_writer.add_summary(summary_str, i)
            if i % 200 == 0 :
                valid_loss = float(sess.run(loss,feed_dict=feed_data_test))
                print(str(i) + " ******** valid loss : " + str(valid_loss))
                if pre_valid_loss < valid_loss:
                    print("valid loss over ! pre: %f   cur: %f" % (pre_valid_loss,valid_loss))
                    break
                pre_valid_loss = valid_loss
        else :
                feed_data = {xs:train_x_data,ys:train_y_data}
                sess.run(train,feed_dict=feed_data)
                if i % 200 == 0:
                    print(str(i) + " loss : " + str(sess.run(loss,feed_dict=feed_data)))
                    summary_str = sess.run(merged_summary_op,feed_dict=feed_data)
                    summary_writer.add_summary(summary_str, i)
                    print(str(i) + " ******** valid loss : " + str(sess.run(loss,feed_dict=feed_data_test)))

    print("training elapsed time : " + str(tm.time() - time_start) + " s")

    ###### SAVE ######
    if is_save == True :
        last_chkp = saver.save(sess, save_dir)
        print("graph save!")
    
    ###### TEST ######
    print("test data : %d" % (len(test_y_data)))
    test_data_x = test_x_data
    test_data_y = test_y_data
    testdatay = sess.run(hidel4,feed_dict={xs:test_data_x,ys:test_data_y})

    #test_data_y = Re_minmaxScalerTest(test_data_y,y_max,y_min)
    #testdatay = Re_minmaxScalerTest(testdatay,y_max,y_min)

    print("right y : " + str(test_data_y[0:10]))
    print("test y : " + str(testdatay[0:10]))
    print("test loss : " + str(sess.run(loss,feed_dict={xs:test_data_x,ys:test_data_y})))
    
    test_x_data = Re_minmaxScalerTest(test_x_data,x_max,x_min)

    print("great test : ")
    num_good_loss = 0
   
    for idx in range(len(test_data_y)):
         if (np.abs(float(test_data_y[idx][0]) - float(testdatay[idx][0])) < 0.0001):
            num_good_loss+=1
            #print("*******%d good loss sample :" % num_good_loss)
            #print(test_x_data[idx])
            #print("right y: " + str(test_data_y[idx]))
            #print("test y: " + str(testdatay[idx]))
    print("good loss : %d" % num_good_loss)


    #output_data_y =
    #sess.run(hidel3,feed_dict={xs:test_data_x,ys:test_data_y})[200:500]
    #test_data_index = range(300)#len(test_data_y)
    #plot_right = plb.plot(test_data_index, test_data_y[200:500], 'r')
    #plot_test = plb.plot(test_data_index, output_data_y, 'b')
    ##plb.ylim(-1., 1.)
    #plb.legend(["plot_right", "plot_test"], loc = 'best', ncol = 2)
    #plb.show()

###### START ######
if __name__ == '__main__':
    main()

# tensorboard --logdir=F:\tlog
# ps -ef | grep python 
# setsid python3 bpnet.py > runlog.out 2>&1

    #for i in range(1000):
    #    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    #    if i % 100 == 0:
    #        print(str(i) + " loss : " + str(sess.run(loss,feed_dict = {xs:x_data,ys:y_data})))