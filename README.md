# Assignment02

This project contains the implementation of the VIC algorithm for clustering validity index evaluation using the ML.NET library.

## VIC Validity Index
A validity index consists on a metric to evaluate how good or how bad is an specific clustering algorithm to find inherent data patterns in the processed information and 
thus determine if the clustering algorithm have performed a good job on an specific dataset. The easiest way to do this is comparing the clustering labels produced with the
ground truth of the data, but most of the times this data is not available. 

VIC validity index is one specific algorithm that is able to avoid this limitation and has proven good performance. It consists on generating the clusters in our data using the
clustering algorithm we are evaluating, and then using this clusters as labels for the data, this is, as data classes. Such data classes are used to train several supervised classifiers 
with a cross validation of an specific "n" number of folds. Once the classifiers have been trained on one of the n folds (this is trained partitioning the data in n parts and using every part but one to train and the one left over to test), this trained classifier
is used over the testing data of this fold to produce predictions, such predictions are then used to generate a confusion matrix, where the predicted classes are compared against the classes
provided by our clustering algorithm and True Positive, True Negatives, False Positives and False Negatives are calculated. This values are then used to calculate the Area Under the ROC curve of the fold in particular.

The AUC metric calculated in each fold is accumulated with the calculated in every other fold, and then divided by the number of folds, this is we obtain the mean of the auc metrics obtained in the cross validation. This evaluation is repeated on every supervised
classifier algorithm that we are including in our VIC implementation, and then the maximum of them all is reported as the validity index value. This is, works as the following pseudocode indicates:

```
inputs: D: dataset; S: set of supervised classifiers; C: a clustering algorithm
Execute C on D to compute the set of clusters P = {C1, C2, ..., Ck}
Create a dataset T with all the objects in D; where every object is labeled with the index of the cluster Ci ∈ P to which the object belongs
Randomly divide T into n (normally 5) subsets Z1, Z2, ..., Zn of size |T|/n each
Initialize the resulting index v <- 0
foreach classifier Sj in S do
  initialize the current result v' <- 0
  for i = 1..n do
    Train Sj using T without Zi as the training dataset and test using Zi.
    Compute the AUC and update v' <- v' + AUC(Sj, Zi)
  Update the computed index v <- max{v, v'/n}
Return v
```

## Our VIC implementation and paramethers for execution.

Since this index is based on the evaluations obtained by some supervised classifiers, the following ones are implemented:

- Random Forest.
- Averaged Perceptron.
- Gradient Boosting Desicion Tree.
- Logistic Regression.
- Maximum Entropy.
- Naive Bayes.
- Support Vector Machine.

The code must be compiled before execution, and in order to be executed the Assignment02.exe file generated must be in a folder the different .dll files generated for compilations, as well as with a Data Subdirectory 
from wich it will take its input. Such folders and data are also provided in this repository. Therefore it must be as follows:

    Assignment.exe
    Data/
      BinaryData/
        ex_database_threshold-0.24.txt
        ex_database_threshold-0.23.txt
        ...
        ex_database_threshold0.25.txt
      MultiClassData/
        ex_database_threshold_0.0_-0.25.txt
        ex_database_threshold_0.01_-0.25.txt
        ...
        ex_database_threshold_0.25_-0.01.txt
    Microsoft.ML.Core.dll
    ...
    System.Threading.Channels.dll

Where this files contain the data in a file separated by tabs. The names indicate the values used to partition the data and generate our clusters. This information might be found in the bin folder
if the project is download and compiled. So in order to run the program, we have to open our command line and navigate to the binaries folder and run the program providing from the command line
the following paramethers.

```
Assignment02.exe ClassifierParameter MultiThreadingParameter DataParameter
```

It is important to notice that every paramether MUST be provided and the code will not work with every one of them. The values and usage of each one of this three paramethers will be described below.
Also, the example above is using aliases and thus do not represent proper values to execute the program. In order to execute the program using all the classifiers, in a multithreading environment and using the binary data 
to generate the results, the code will be the following.

```
Assignment02.exe brpmslt m b
```

### Classifiers Selection.

The classifiers to use, are optional and the user is able to decide which ones to include during the execution, for this purpose when the binary file is executed, the 
user must provide an paramether indicating the classifiers to include. This parameter consists in a string where each character included represents each classifier. The optional
characters:

For the classifiers                

    * r -> Random Forest                            
    * b -> Naive Bayes                                    
    * p -> Averaged Perception
    * m -> Maximum Entropy
    * t -> Gradient Boosting Decision Tree 
    * s -> Support Vector Machine
    * l -> Logistic Regression

The order of the characters is not relevant, but at least one of them must be provided, providing none, or one that is not part of the implemented options will result in 
An Error validation message and the immediate termination of the program.

#### Example of Proper Execution:

```
Assignment02.exe brpmslt MultiThreadingParameter DataParameter
```

This works and executes the program using all the seven implemented classifiers.

#### Example of Wrong Execution:

```
Assignment02.exe bkmslt MultiThreadingParameter DataParameter
```

This second attempt fails since k is not in the possible options.

This parameter MUST always be provided as the first one.

### MultiThreading

The code is prepared to work using several threads, letting each classifier run its evaluations on its own thread. The user is not able to decide directly how many threads to create,
but this can be controlled by indicating the classifiers that will be used to use during execution. Yet it can also be runned in a single thread environment.
In order to decide if the classifiers will run on its own thread each or if the execution will be performed in a single thread, the user must provide an string during execution. This paramether 
might have one of the following options.

    * m -> Multiple Threading   
    * s -> Single Threading       

Providing nothing in this parameter, or one character that is not "s" or "m" will result in an Error validation message and the immediate termination of the program.

#### Example of Proper Execution:

```
Assignment02.exe brpmslt m DataParameter
```

This works and executes the program using multiple threads, one for each classifier.

#### Example of Wrong Execution:

```
Assignment02.exe bkmslt o DataParameter
```

This second attempt fails since o is not in the possible options.

This parameter MUST always be provided as the second one.

### Cluster Data Selection

Since the code is developed to produce an implementation of the VIC algorithm that evaluates an specific dataset composed of minutias attributes, such data is also provided and included in the repository.
The data was divided into two or three clusters using the score_change attribute values as the criteria to decide the cluster which each minutia belongs. 50 different partitions were performed for two clusters, 
as well as for three clusters, so 100 different files are provided. Since we are interested in evaluate the behaviour of the algorithm in binary and non binary environments, 50 of them belong to one experiment
while 50 of them to another. To decide which experiment to perform, user must provide when executing the binary, the experiment to perform (which results will then be saved in a results.xml file). This parameter may have
the following values.

    * m -> Three Clusters Data
    * b -> Two Clusters Data     

Providing nothing in this parameter, or one character that is not "b" or "m" will result in an Error validation message and the immediate termination of the program.

#### Example of Proper Execution:

```
Assignment02.exe brpmslt m b
```

This works and executes the program using the binary clusters for the experiments

#### Example of Wrong Execution:

```
Assignment02.exe bkmslt o 
```

This second attempt fails since no third paramether is provided.

This parameter MUST always be provided as the third one.

## Results

After executing the code and the finalization of the experiments, an XML file will be found on the inside of the Data folder, as follows:

    Assignment02.exe
    Data/
      *results.xml*

This file will include the Validity Index for each experiment, as well as the AUC obtained by each classifier. It will also calculate the Minimum and maximum value of the
validity index and of the classifiers on the totality of the experiments, and the Q1, Q2 and Q3 values, to identify how the results are distributed under the 50 experiments performed
and therefore having a better picture of the performance of each classifier.

## Team Members

* Luis Alberto Garnica López A00828182
* Arturo Daniel González Cañón A00513641
* Hernan Espinosa Rodríguez A01112132
