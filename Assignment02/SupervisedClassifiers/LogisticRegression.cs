using Assignment02.DataClasses;
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
    /*Logistic regression is a well-known method in statistics that is used to predict the probability of an outcome, 
     * and is especially popular for classification tasks. The algorithm predicts the probability of occurrence of 
     * an event by fitting data to a logistic function. 
     */

    class LogisticRegression
    {
        LbfgsLogisticRegressionBinaryTrainer.Options options;
        MLContext mlContext;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.PairwiseCouplingModelParameters>> pipeline;
        
        //Creates an instance of the classifier, configured with the different options that this classifier requires.
        public LogisticRegression(int maxNumIterations, float optimizationTolerance, float l2Regularization, MLContext mlContext)
        {
            this.options = new LbfgsLogisticRegressionBinaryTrainer.Options
            {
                MaximumNumberOfIterations = maxNumIterations,
                OptimizationTolerance = optimizationTolerance,
                L2Regularization = l2Regularization
            };
            this.mlContext = mlContext;
        }

        //This method loads the trainer according to the configurations defined previously.
        public void prepareModel()
        {
            // Define the trainer.
            this.pipeline =
                // Convert the string labels into key types.
                mlContext.Transforms.Conversion.MapValueToKey("Label")
                // Apply PairwiseCoupling multiclass meta trainer on top of
                // binary trainer.
                .Append(mlContext.MulticlassClassification.Trainers
                .PairwiseCoupling(
                mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(this.options)));
        }
        
        //Executes the training and evaluation of this classifier on every fold partition defined in the code.
        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i < numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                Console.WriteLine("LogisticRegression Training Fold: " + i);

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
