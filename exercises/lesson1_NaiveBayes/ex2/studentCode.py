from exercises.ex2.classify import NBAccuracy

from exercises.lesson1_NaiveBayes.ex1.prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy


print({"accuracy": submitAccuracy()})
