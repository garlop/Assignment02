using Assignment02.DataClasses;
using Assignment02.Utils;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace Assignment02.SupervisedClassifiers
{
    /*A boosted decision tree is an ensemble learning method in which the second tree corrects for the errors of 
     *the first tree, the third tree corrects for the errors of the first and second trees, and so forth. 
     *Predictions are based on the entire ensemble of trees together that makes the prediction. 
     *
     *Generally, when properly configured, boosted decision trees are the easiest methods with which to get top 
     *performance on a wide variety of machine learning tasks. However, they are also one of the more memory-intensive 
     *learners, and the current implementation holds everything in memory. Therefore, a boosted decision tree model 
     *might not be able to process the very large datasets that some linear learners can handle.
     */

    class GradientBoostingDesicionTree
    {
        MLContext mlContext;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.OneVersusAllModelParameters>> pipeline;
        
        //Creates an instance of the classifier, configured with the different options that this classifier requires.
        public GradientBoostingDesicionTree(MLContext mlContext)
        {
            this.mlContext = mlContext;
        }

        //This method loads the trainer according to the configurations defined previously.
        public void prepareModel()
        {
            // Define the trainer.
            this.pipeline =
                // Convert the string labels into key types.
                mlContext.Transforms.Conversion
                .MapValueToKey(nameof(MinutiaData.Label))
                // Apply LightGbm multiclass trainer.
                .Append(mlContext.MulticlassClassification.Trainers.
                LightGbm());
        }

        //Executes the training and evaluation of this classifier on every fold partition defined in the code.
        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i< numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                Console.WriteLine("GradientBoosting Training Fold: " + i);

                var replacementEstimator = mlContext.Transforms.ReplaceMissingValues("Features", replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue);
                // Fit data to estimator
                // This is not suitable, as it takes to much.
                ITransformer replacementTransformer = replacementEstimator.Fit(trainingData);
                // Transform data
                IDataView transformedtrainingData = replacementTransformer.Transform(trainingData);

                ITransformer replacementtestingTransformer = replacementEstimator.Fit(testData);
                IDataView transformedtestingData = replacementtestingTransformer.Transform(testData);

                var model = pipeline.Fit(transformedtrainingData);

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
