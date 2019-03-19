import numpy as np
import cv2
import cv2.ml as ml

from MachineLearn.Classes import *

KERNEL = "SIGMOID"
FILE = "FEATURES_M1_CM8b"

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("GLCM/EXP_01/FEATURES_M1_CM8b.txt", usecols=range(24), delimiter=",")
classes = np.loadtxt("GLCM/EXP_01/FEATURES_M1_CM8b.txt", dtype=object, usecols=24, delimiter=",")
cls1 = len(classes[classes == 'Class 1'])
cls2 = len(classes[classes == 'Class 2'])
print (cls1)
print (cls2)

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
for j in range(50):
    print (j)
    oData = Data(2, 11, samples=1398)
    oData.random_training_test_by_percent([cls1, cls2], 0.8)
    svm = ml.SVM_create()
    # svm.setKernel(ml.SVM_CHI2)
    # svm.setType(ml.SVM_C_SVC)
    # svm.setDegree(0.1)
    oData.params = dict(kernel = ml.SVM_CHI2, kFold=2)
    svm.trainAuto(np.float32(oDataSet.attributes[oData.Training_indexes]), ml.ROW_SAMPLE,
                  np.int32(oDataSet.labels[oData.Training_indexes]), kFold=2)
    # svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
    #                np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
    results = []#svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
    for i in (oDataSet.attributes[oData.Testing_indexes]):
        res, cls = svm.predict(np.float32([i]))
        results.append(cls[0])
    oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
    # oData.insert_model(svm)
    oDataSet.append(oData)
oExp.add_data_set(oDataSet,
                  description="  50 execucoes SVM_{} base DroneScan arquivos em Exp01 - FEATURES_M1_CM8b.txt. ".format(
                      KERNEL))
oExp.save("Objects/EXP_01/{}_{}.gzip".format(KERNEL, FILE))

oExp = oExp.load("Objects/EXP_01/{}_{}.gzip".format(KERNEL, FILE))

print (oExp)
print (oExp.experimentResults[0].sum_confusion_matrix / 50)
