﻿using Assignment02.DataClasses;
using Assignment02.Utils;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace Assignment02.SupervisedClassifiers
{
    /*The averaged perceptron method is an early and very simple version of a neural network. In this approach, 
     * inputs are classified into several possible outputs based on a linear function, and then combined with a 
     * set of weights that are derived from the feature vector—hence the name "perceptron."
     * 
     * The simpler perceptron models are suited to learning linearly separable patterns, whereas neural 
     * networks (especially deep neural networks) can model more complex class boundaries. However, perceptrons 
     * are faster, and because they process cases serially, perceptrons can be used with continuous training.
     */

    class AveragedPerceptron
    {
        AveragedPerceptronTrainer.Options options;
        MLContext mlContext;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.PairwiseCouplingModelParameters>> pipeline;
        
        //Creates an instance of the classifier, configured with the different options that this classifier requires.
        public AveragedPerceptron(bool averaged, float learningRate, bool decreaseLR, int numOfIterations, float l2Regularization, bool shuffle, MLContext mlContext)
        {
            this.options = new AveragedPerceptronTrainer.Options
            {
                // Only use 80% of features to reduce over-fitting.
                FeatureColumnName = "Features",
                LabelColumnName = "Label",
                // Create a simpler model by penalizing usage of new features.
                Averaged = averaged,
                LearningRate = learningRate,
                DecreaseLearningRate = decreaseLR,
                NumberOfIterations = numOfIterations,
                L2Regularization = l2Regularization,
                Shuffle = shuffle
            };
            this.mlContext = mlContext;
        }

        //This method loads the trainer according to the configurations defined previously.
        public void prepareModel()
        {
            // Define the trainer.
            this.pipeline =
                this.mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.MulticlassClassification.Trainers
                .PairwiseCoupling(
                mlContext.BinaryClassification.Trainers.AveragedPerceptron(this.options)));
        }

        //Executes the training and evaluation of this classifier on every fold partition defined in the code.
        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i < numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                Console.WriteLine("AveragePerceptron Training Fold: " + i);

                var replacementEstimator = mlContext.Transforms.ReplaceMissingValues("Features", replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue);
                // Fit data to estimator
                // This is not suitable, as it takes to much.
                ITransformer replacementTransformer = replacementEstimator.Fit(trainingData);
                // Transform data
                IDataView transformedtrainingData = replacementTransformer.Transform(trainingData);

                ITransformer replacementtestingTransformer = replacementEstimator.Fit(testData);
                IDataView transformedtestingData = replacementtestingTransformer.Transform(testData);


                // Train the model.
                var model = this.pipeline.Fit(transformedtrainingData);

                // Run the model on test data set.
                var transformedTestData = model.Transform(transformedtestingData);

                // Convert IDataView object to a list.
                var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedTestData, reuseRowObject: false).ToList();

                // Evaluate the overall metrics.
                var metrics = mlContext.MulticlassClassification.Evaluate(transformedTestData);

                var areaUnderRocCurve = AUC.ComputeMultiClassAUC(metrics.ConfusionMatrix.Counts);

                vi = vi + areaUnderRocCurve;
            }
            return vi / numFolds;
        }
    }
}
