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
    class NaiveBayes
    {
        MLContext mlContext;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.NaiveBayesMulticlassModelParameters>> pipeline;
        public NaiveBayes(MLContext mlContext) 
        {
            this.mlContext = mlContext;
        }

        public void prepareModel()
        {
            // Define the trainer.
            this.pipeline =
                // Convert the string labels into key types.
                mlContext.Transforms.Conversion
                .MapValueToKey(nameof(MinutiaData.Label))
                // Apply NaiveBayes multiclass trainer.
                .Append(mlContext.MulticlassClassification.Trainers
                .NaiveBayes());
        }

        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i < numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                Console.WriteLine("NaiveBayes Training Fold: " + i);
                // Train the model.
                var model = pipeline.Fit(trainingData);

                // Run the model on test data set.
                var transformedTestData = model.Transform(testData);

                // Convert IDataView object to a list.
                var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedTestData, reuseRowObject: false).ToList();

                // Evaluate the overall metrics
                var metrics = mlContext.MulticlassClassification.Evaluate(transformedTestData);

                var areaUnderRocCurve = AUC.ComputeMultiClassAUC(metrics.ConfusionMatrix.Counts);

                vi = vi + areaUnderRocCurve;
            }
            return vi / numFolds;
        }
    }
}
