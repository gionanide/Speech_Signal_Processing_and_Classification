  Implementation of SVM algorithm for classification **svm_default.py** is using only the default parameters to initialize the procedure.
  
 In this  folder there are variations as concerns the methods of training and the method of evaluation of SVM algorithm. Experiment resutlts using different kernel functions, and different values of parameters. Training methods with balanced training set, the balance is about the number of samples of each class **svm_balancedSampleNumber_greedySearch.py**.

  Examples of this training is using the divided parts and keep only the samples that are support vectors in every iteration, continue this procedure until the class with more samples is finished of iterating. Last one, using greedy algorithms to calculate the kernel parameters.
  
  In the script **svm_keeping_supportVectors.py** the above experiment is taking place. As a first approach we train our model taking all the samples from class0 and devide them accrodingly just to balance our data, we continue this porcedure until we do not have more untis of samples from class0. From this iteration we keep all the support vectors, which contains samples from both classes. We erase the duplicates and we delete all the samples from class1, so we have a dataframe containing all the support_vectors from class0. And then we feed our model in order to train it with all the samples from class1 and only the samples that were support vectors from class0, and we repeat this procedure. In the end the amount of samples from class0 is going to be smaller than the amount of samples from class1 and when this becomes smaller than the half of the amount of class1 samples we stop.
  
  In general because class0 has 6 times more samples than class1 in order to reduce the amount of samples of class0 we try this procedure taking the support vectors and then the support vectors of support vectors and goes on.
  
  Furthermore in the script **svm_multiclass.py** we try to classify a dataset of three classes.
