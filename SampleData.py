# -*- coding: utf-8 -*-  
import json
import time

#############################
##
def loadJsonData(filename):
    filePath = "trainingData//" + filename
    jsonFile = open(filePath, encoding='utf-8')
    jsonData = json.load(jsonFile)
    print("-------- Json File " + filePath + " Load Complete! --------")
    return jsonData

###############################
##
class SampleData(object):
    def __init__(self,frameName):
        self.frameName = frameName
        self.velocity = []
        self.position = []
        self.force = []
        self.correctionPressureForce = []
        self.neighbor = []

    def clear(self):
        self.force.clear()
        self.position.clear()
        self.velocity.clear()
        self.correctionPressureForce.clear()
        self.neighbor.clear()
        self.frameName = ""

    def setValue(self,attribute,value):
        if(attribute == "force"):
            self.force.append(value)
        if(attribute == "velocity"):
            self.velocity.append(value)
        if(attribute == "position"):
            self.position.append(value)
        if(attribute == "correctionPressureForce"):
            self.correctionPressureForce.append(value)
        if(attribute == "neighbor"):
            self.neighbor.append(value)

###############################
##
def sortFrame(sampleData=[]):
    _sampleData = []
    slen = len(sampleData)
    fsa = sampleData[0]
    for i in range(slen):
        fsa = sampleData[0]
        for sa in sampleData:
            numfsa = int(fsa.frameName[5:])
            numsa = int(sa.frameName[5:])
            if numfsa > numsa :
               fsa = sa
        _sampleData.append(fsa)
        sampleData.remove(fsa)
    print("sortFrame :")
    for sa in _sampleData:
        print(sa.frameName, end=' ')
    print()
    return _sampleData

def getSampleData(fileName):
    time_start = time.time()
    sampleData = []
    jsonRoot = loadJsonData(fileName)
    for frame in jsonRoot: #frame
        #if frame != "frame10" and frame != "frame15":
        #    continue
        print("--------- read " + frame + " ---------")
        sd = SampleData(frame)
        sampleData.append(sd)
        for attribute in jsonRoot[frame]:   #get attribute in frame
            for data in jsonRoot[frame][attribute]:
                sampleData[-1].setValue(attribute,data)
            #print(attribute + " size: " +
            #str(len(jsonRoot[frame][attribute])))
    sampleData = sortFrame(sampleData)
    print()
    print("sampleData size: " + str(len(sampleData)))
    print("generateValue elapsed time : " + str(time.time() - time_start) + " s")
    #print("sampleData[0].density : " + str(sampleData[0].density))
    #print("sampleData[1].density : " + str(sampleData[1].density))
    print()
    return sampleData


    
    