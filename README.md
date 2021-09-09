# SVM-k-fold-cross-validation

(Project for 'Introduction to Machine Learning for Natural Language Processing' class at UniTrento, MSc in Cognitive Science)

Starting from the code in https://github.com/ml-for-nlp/SVMs, I expanded the code in the final part of SVM training, in order to produce a less biased result.

Before proceeding with the vectorization of the documents in our datasets, we
split the raw text into a training dataset (90%) and a testing dataset (10%). We
will use the training dataset to train the model and tune the hyperparameters
and the remaining testing dataset to test our final version of the classifier.
Once obtained the various vectorizations of each document of the training
dataset (one per each combination of type and number of features) we fit our
data into the model and calculate the accuracy. To avoid an accuracy score
biased by a particular split of our data, we performed a 9-fold cross-validation.
Moreover, the model can be implemented using different kinds of kernels and
parameters. In particular, we may want to vary the type of kernel across three
different possibilities: linear, polynomial, and RBF Gaussian; and the hyperparameter C across different numeric values (in this work 1-10-20-100). To verify
which of these combinations will allow us to obtain better performances we perform the hyperparameters tuning. To not be biased by a particular splitting of
the dataset into training and testing data, we use 5-fold cross-validation. So,
for each combination, we will obtain the accuracy for each split and we will
calculate the mean of these accuracy scores. We performed hyperparameters
tuning of both the model trained by taking words as features and the model
trained by taking 3-grams as features (in both cases considering the top 100
features).
We then test the model on the testing dataset we saved at the beginning of our
experiment.
