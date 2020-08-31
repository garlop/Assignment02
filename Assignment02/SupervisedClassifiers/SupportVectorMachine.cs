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
    class SupportVectorMachine
    {
        LinearSvmTrainer.Options options;
        MLContext mlContext;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.PairwiseCouplingModelParameters>> pipeline;
    
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

        public void prepareModel()
        {
            this.pipeline =
                mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.MulticlassClassification.Trainers
                .PairwiseCoupling(
                mlContext.BinaryClassification.Trainers.LinearSvm(this.options)));
        }

        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i < numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                Console.WriteLine("SupportVectorMachine Training Fold: " + i);

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
