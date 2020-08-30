using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Data;

namespace Assignment02
{
    // This code requires installation of additional NuGet package for 
    // Microsoft.ML.FastTree at
    // https://www.nuget.org/packages/Microsoft.ML.FastTree/
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "index_dmc_new_attributes_8.txt");
        //static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static void Main(string[] args)
        {
            // Create a new context for ML.NET operations as the source of randomness. Setting the seed to a fixed number to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            TrainTestData splitDataView = LoadData(mlContext);

            var trainingData = splitDataView.TrainSet;

            // Define trainer options.
            var options = new FastForestBinaryTrainer.Options
            {
                // Only use 80% of features to reduce over-fitting.
                FeatureFraction = 0.8,
                // Create a simpler model by penalizing usage of new features.
                FeatureFirstUsePenalty = 0.1,
                // Reduce the number of trees to 50.
                NumberOfTrees = 50
            };

            // Define the trainer.
            var pipeline = mlContext.BinaryClassification.Trainers.FastForest(options);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different from training data.
            var testData = splitDataView.TestSet;

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedTestData,reuseRowObject: false).ToList();

            // Evaluate the overall metrics.
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(transformedTestData);

            PrintMetrics(metrics);

            // Expected output:
            //   Accuracy: 0.73
            //   AUC: 0.81
            //   F1 Score: 0.73
            //   Negative Precision: 0.77
            //   Negative Recall: 0.68
            //   Positive Precision: 0.69
            //   Positive Recall: 0.78
            //
            //   TEST POSITIVE RATIO:    0.4760 (238.0/(238.0+262.0))
            //   Confusion table
            //             ||======================
            //   PREDICTED || positive | negative | Recall
            //   TRUTH     ||======================
            //    positive ||      186 |       52 | 0.7815
            //    negative ||       77 |      185 | 0.7061
            //             ||======================
            //   Precision ||   0.7072 |   0.7806 |
        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<MinutiaData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        public class MinutiaData
        {
            [LoadColumn(0, 243), VectorType(244), ColumnName("Features")]
            public float[] features { get; set; }

            [LoadColumn(245), ColumnName("Label")]
            public bool clase { get; set; }
        }

        // Class used to capture predictions.
        private class Prediction
        {
            // Original label.
            public bool Label { get; set; }
            // Predicted label from the trainer.
            public bool PredictedLabel { get; set; }
        }

        // Pretty-print BinaryClassificationMetrics objects.
        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " + $"{metrics.NegativePrecision:F2}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " + $"{metrics.PositivePrecision:F2}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}
