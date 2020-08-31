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
    class LogisticRegression
    {
        LbfgsLogisticRegressionBinaryTrainer.Options options;
        MLContext mlContext;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.PairwiseCouplingModelParameters>> pipeline;
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

        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i < numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                Console.WriteLine("LogisticRegression Training Fold: " + i);
                // Train the model.
                var model = this.pipeline.Fit(trainingData);

                // Run the model on test data set.
                var transformedTestData = model.Transform(testData);

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
