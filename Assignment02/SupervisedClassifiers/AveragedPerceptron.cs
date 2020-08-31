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
    class AveragedPerceptron
    {
        AveragedPerceptronTrainer.Options options;
        MLContext mlContext;
        Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.PairwiseCouplingModelParameters>> pipeline;
        //AveragedPerceptronTrainer pipeline;
        public AveragedPerceptron(bool averaged, float learningRate, bool decreaseLR, int numOfIterations, float l2Regularization, bool shuffle, MLContext mlContext)
        {
            this.options = new AveragedPerceptronTrainer.Options
            {
                // Only use 80% of features to reduce over-fitting.
                FeatureColumnName = "Features",
                LabelColumnName = "Label",
                // Create a simpler model by penalizing usage of new features.
                Averaged = averaged,
                LearningRate = learningRate,
                DecreaseLearningRate = decreaseLR,
                NumberOfIterations = numOfIterations,
                L2Regularization = l2Regularization,
                Shuffle = shuffle
            };
            this.mlContext = mlContext;
        }

        public void prepareModel()
        {
            // Define the trainer.
            this.pipeline =
                this.mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.MulticlassClassification.Trainers
                .PairwiseCoupling(
                mlContext.BinaryClassification.Trainers.AveragedPerceptron(this.options)));
        }

        public double trainAndEvaluateModel(int numFolds, IReadOnlyList<TrainTestData> splitDataView)
        {
            double vi = 0;
            for (int i = 0; i < numFolds; i++)
            {
                var trainingData = splitDataView[i].TrainSet;
                var testData = splitDataView[i].TestSet;
                Console.WriteLine("AveragePerceptron Training Fold: " + i);
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
