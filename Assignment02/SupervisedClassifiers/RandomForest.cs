using Assignment02.DataClasses;
using Assignment02.Utils;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace Assignment02.SupervisedClassifiers
{
    class RandomForest
    {
        FastForestBinaryTrainer.Options options;
        MLContext mlContext;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.PairwiseCouplingModelParameters>> pipeline;
        //Microsoft.ML.Trainers.FastTree.FastForestBinaryTrainer pipeline;
        public RandomForest(double featureFraction, double featureFirstUsePenalty, int numberOfTrees, MLContext mlContext)
        {
            // Define trainer options.
            this.options = new FastForestBinaryTrainer.Options
            {
                // Only use 80% of features to reduce over-fitting.
                FeatureFraction = featureFraction,
                // Create a simpler model by penalizing usage of new features.
                FeatureFirstUsePenalty = featureFirstUsePenalty,
                // Reduce the number of trees to 50.
                NumberOfTrees = numberOfTrees
            };
            this.mlContext = mlContext;
        }

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
                mlContext.BinaryClassification.Trainers.FastForest(this.options)));

            // Define the trainer.
            //this.pipeline = this.mlContext.BinaryClassification.Trainers.FastForest(this.options);
        }

        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i < numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                Console.WriteLine("RandomForest Training Fold: " + i);
                
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
