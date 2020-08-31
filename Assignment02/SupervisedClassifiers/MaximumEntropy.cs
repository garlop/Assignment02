using Assignment02.DataClasses;
using Assignment02.Utils;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace Assignment02.SupervisedClassifiers
{
    class MaximumEntropy
    {
        MLContext mlContext;
        SdcaMaximumEntropyMulticlassTrainer.Options options;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.MaximumEntropyModelParameters>> pipeline;
        public MaximumEntropy(float convergenceTolerance, int maximumNumberOfIterations, MLContext mlContext)
        {
            this.options = new SdcaMaximumEntropyMulticlassTrainer.Options
            {
                // Make the convergence tolerance tighter.
                ConvergenceTolerance = convergenceTolerance,
                // Increase the maximum number of passes over training data.
                MaximumNumberOfIterations = maximumNumberOfIterations,
            };
            this.mlContext = mlContext;
        }

        public void prepareModel()
        {
            this.pipeline =
                // Convert the string labels into key types.
                mlContext.Transforms.Conversion.MapValueToKey("Label")
                // Apply SdcaMaximumEntropy multiclass trainer.
                .Append(mlContext.MulticlassClassification.Trainers
                .SdcaMaximumEntropy(options));
        }

        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i< numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                
                // ML.NET doesn't cache data set by default. Therefore, if one reads a
                // data set from a file and accesses it many times, it can be slow due
                // to expensive featurization and disk operations. When the considered
                // data can fit into memory, a solution is to cache the data in memory.
                // Caching is especially helpful when working with iterative algorithms 
                // which needs many data passes.
                trainingData = mlContext.Data.Cache(trainingData);
                
                Console.WriteLine("MaximumEntropy Training Fold: " + i);

                var model = this.pipeline.Fit(trainingData);

                var transformedTestData = model.Transform(testData);

                var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedTestData, reuseRowObject: false).ToList();

                var metrics = mlContext.MulticlassClassification.Evaluate(transformedTestData);

                var areaUnderRocCurve = AUC.ComputeMultiClassAUC(metrics.ConfusionMatrix.Counts);

                vi = vi + areaUnderRocCurve;
            }
            return vi / numFolds;
        }
    }
}
