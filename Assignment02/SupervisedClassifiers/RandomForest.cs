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
    /*The decision forest algorithm is an ensemble learning method for classification. The algorithm works by 
     * building multiple decision trees and then voting on the most popular output class. Voting is a form of 
     * aggregation, in which each tree in a classification decision forest outputs a non-normalized frequency 
     * histogram of labels. The aggregation process sums these histograms and normalizes the result to get the 
     * “probabilities” for each label. The trees that have high prediction confidence have a greater weight in the 
     * final decision of the ensemble.
     * 
     * Decision trees in general are non-parametric models, meaning they support data with varied distributions. 
     * In each tree, a sequence of simple tests is run for each class, increasing the levels of a tree structure 
     * until a leaf node (decision) is reached.
     */

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
