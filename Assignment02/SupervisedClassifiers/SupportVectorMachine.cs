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
    /*Support vector machines (SVMs) are an extremely popular and well-researched class of supervised learning models, 
     * which can be used in linear and non-linear classification tasks. Recent research has focused on ways to optimize 
     * these models to efficiently scale to larger training sets. 
     * This model is a supervised learning method, and therefore requires a tagged dataset, which includes a 
     * label column.
     */
    class SupportVectorMachine
    {
        LinearSvmTrainer.Options options;
        MLContext mlContext;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.PairwiseCouplingModelParameters>> pipeline;

        //Creates an instance of the classifier, configured with the different options that this classifier requires.
        public SupportVectorMachine(int batchSize, bool performProjection, int numberOfIterations, MLContext mlContext)
        {
            this.options = new LinearSvmTrainer.Options
            {
                BatchSize = batchSize,
                PerformProjection = performProjection,
                NumberOfIterations = numberOfIterations
            };
            this.mlContext = mlContext;
        }

        //This method loads the trainer according to the configurations defined previously.
        public void prepareModel()
        {
            this.pipeline =
                mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.MulticlassClassification.Trainers
                .PairwiseCoupling(
                mlContext.BinaryClassification.Trainers.LinearSvm(this.options)));
        }

        //Executes the training and evaluation of this classifier on every fold partition defined in the code.
        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i < numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                Console.WriteLine("SupportVectorMachine Training Fold: " + i);

                var replacementEstimator = mlContext.Transforms.ReplaceMissingValues("Features", replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue);
                // Fit data to estimator
                // This is not suitable, as it takes to much.
                ITransformer replacementTransformer = replacementEstimator.Fit(trainingData);
                // Transform data
                IDataView transformedtrainingData = replacementTransformer.Transform(trainingData);

                ITransformer replacementtestingTransformer = replacementEstimator.Fit(testData);
                IDataView transformedtestingData = replacementtestingTransformer.Transform(testData);

                var model = this.pipeline.Fit(transformedtrainingData);

                var transformedTestData = model.Transform(transformedtestingData);

                var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedTestData, reuseRowObject: false).ToList();

                var metrics = mlContext.MulticlassClassification.Evaluate(transformedTestData);

                var areaUnderRocCurve = AUC.ComputeMultiClassAUC(metrics.ConfusionMatrix.Counts);

                vi = vi + areaUnderRocCurve;
            }
            return vi / numFolds;
        }
    }
}
