from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import classification_report
from imutils import paths
import argparse
import cv2
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import cross_val_score
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import csv
import glob
import shutil, sys                                                                                                                                                    


'''
This section is about reading Training.csv to read images from folder and pass them from Haar filter. Also the multi-labels are stored as well.
'''

Features = []
labels = []
feature_types = ['type-4','type-3-y','type-2-x','type-3-x']
count=0
categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','No Finding','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']

os.chdir("/home/safia/Desktop/MIP Project")
with open('Training.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Reading Training.csv row wise to pick name of the image alongwith its labels.
        disease=row['Finding Labels']
        count=count+1
        print(count)
        image=(row['Image Index'])
        os.chdir("/home/safia/Desktop/MIP Project/training")
        print (os.getcwd())
        
        for filename in glob.glob('*.png'):
            if(filename == image):
                print (filename)
                
                image = cv2.imread(filename)
                grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_ii = integral_image(grayScale)

                #Feature Vector
                feature = haar_like_feature(img_ii, 0, 0, 5, 5, feature_types)



                Features.append(feature)
                disease2= disease.split(',')
                labels.append(disease2)
                print (labels[-1])

labels2=[i for i in labels]

'''
This section comprises of creating and fitting Machine Learning model. Here MultiLayer Perceptron is used.
'''

#Transforming labels into Binarized Matrix form. 
mlb = MultiLabelBinarizer(classes=categories)


LabelTransformed = mlb.fit_transform(labels2)
print (LabelTransformed)

model = MLPClassifier(verbose=True,hidden_layer_sizes=(80,50,20),activation='tanh',alpha=0.0001, batch_size='auto',nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, max_iter=10000,validation_fraction=0.1)

transformation_classifier = LabelPowerset(model)
clusterer = IGraphLabelCooccurenceClusterer('fastgreedy', weighted=True, include_self_edges=True) 
classifier = LabelSpacePartitioningClassifier(transformation_classifier, clusterer)

classifier.fit(Features, LabelTransformed)


'''
The testing phase below predicts multi-label when needed. The classification report is generated at the end.
'''
FeaturesTest=[]
actualLabels=[]
pred = []
count=0
os.chdir("/home/safia/Desktop/MIP Project")
with open('testing.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        disease=row['Finding Labels']
        count=count+1
        print(count)
        image=(row['Image Index'])

        os.chdir("/home/safia/Desktop/MIP Project/test")
        print (os.getcwd())


        for filename in glob.glob('*.png'):
            if(filename == image):
                print (filename)
                print("file found")
                image = cv2.imread(filename)
                grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_ii = integral_image(grayScale)
                feature = haar_like_feature(img_ii, 0, 0, 5, 5, feature_types)
                FeaturesTest.append(feature)
                disease2= disease.split(',')
                actualLabels.append(disease2)
                print(disease2," ***")
                print (actualLabels[-1])
                print ("Done")
                print ("\n")


actualLabelsPrep=[i for i in actualLabels]
TrueLabels = mlb.fit_transform(actualLabelsPrep)
PredictLabels = classifier.predict(FeaturesTest)
print (PredictLabels)
print ("\n")

print(classification_report(TrueLabels,PredictLabels,target_names=categories))
print (accuracy_score(TrueLabels,PredictLabels))
