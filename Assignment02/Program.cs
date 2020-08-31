using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Data;
using Assignment02.SupervisedClassifiers;
using Assignment02.DataClasses;
using System.Threading;
using System.Linq;
using System.Globalization;

namespace Assignment02
{
    // This code requires installation of additional NuGet package for 
    // Microsoft.ML.FastTree at
    // https://www.nuget.org/packages/Microsoft.ML.FastTree/
    class Program
    {
        // Define an array with two AutoResetEvent WaitHandles.
        static WaitHandle[] waitHandles = new WaitHandle[]
        {
        new AutoResetEvent(false),
        new AutoResetEvent(false)
        };
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "index_dmc_new_attributes_8.txt");
        //static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static int Main(string[] args)
        {
            string classifiers = "";
            string threading = "";
            if (args.Length != 2){
                Console.WriteLine("Please provide the proper number of execution parameters");
                return 1;
            }
            else
            {
                classifiers = args[0];
                string allowableLetters = "rpbmtsl";
                foreach (char c in classifiers)
                {
                    if (!allowableLetters.Contains(c.ToString()))
                    {
                        Console.WriteLine("Invalid Classifier parameter");
                        return 1;
                    }
                }

                if ((args[1] != "s") && (args[1] != "m"))
                {
                    Console.WriteLine("Invalid Threading parameter");
                    return 1;
                }
                else
                {
                    threading = args[1];
                }
            }
            // Create a new context for ML.NET operations as the source of randomness. Setting the seed to a fixed number to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            IReadOnlyList<TrainTestData> splitDataView = LoadData(mlContext);

            double v = 0;

            if (threading == "s")
            {
                v = SingleThreadExecution(classifiers, v, splitDataView, mlContext);
            }
            else
            {
                v = MultipleThreadExecution(classifiers, splitDataView, mlContext);
            }

            Console.WriteLine($"Metric valuation:{ v }");
            return 0;
        }

        public static IReadOnlyList<TrainTestData> LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<MinutiaData>(_dataPath, hasHeader: false);

            // Cross validation splits your data randomly into set of "folds", and
            // creates groups of Train and Test sets, where for each group, one fold
            // is the Test and the rest of the folds the Train. So below, we specify
            // Group column as the column containing the sampling keys. If we pass
            // that column to cross validation it would be used to break data into
            // certain chunks.
            var folds = mlContext.Data.CrossValidationSplit(dataView, numberOfFolds: 10);
            return folds;
        }

        public static double SingleThreadExecution(string classifiers, double v, IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            if (classifiers.ToLower().Contains('r'))
            {
                RandomForest classifier = new RandomForest(0.8, 0.1, 50, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC RandomForest Value: " + vi);
                v = Math.Max(v, vi);
            }

            if (classifiers.ToLower().Contains('p'))
            {
                AveragedPerceptron classifier = new AveragedPerceptron(true, 0.1f, true, 50, 0.02f, true, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC AveragedPreceptron Value: " + vi);
                v = Math.Max(v, vi);
            }

            if (classifiers.ToLower().Contains('b'))
            {
                NaiveBayes classifier = new NaiveBayes(mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC NaiveBayes Value: " + vi);
                v = Math.Max(v, vi);
            }

            if (classifiers.ToLower().Contains('m'))
            {
                MaximumEntropy classifier = new MaximumEntropy(0.05f, 30, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC MaximumEntropy Value: " + vi);
                v = Math.Max(v, vi);
            }

            if (classifiers.ToLower().Contains("t"))
            {
                GradientBoostingDesicionTree classifier = new GradientBoostingDesicionTree(mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC GradientBoostingDecisionTree Value: " + vi);
                v = Math.Max(v, vi);
            }

            if (classifiers.ToLower().Contains("s"))
            {
                SupportVectorMachine classifier = new SupportVectorMachine(10, true, 10, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC SupportVectorMachine Value: " + vi);
                v = Math.Max(v, vi);
            }

            if (classifiers.ToLower().Contains("l"))
            {
                LogisticRegression classifier = new LogisticRegression(100, 1e-8f, 0.01f, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC LogisticRegression Value: " + vi);
                v = Math.Max(v, vi);
            }

            return v;
        }

        public static double TrainRandomForest(IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            RandomForest classifier = new RandomForest(0.8, 0.1, 50, mlContext);
            classifier.prepareModel();
            double vi = classifier.trainAndEvaluateModel(10, splitDataView);
            Console.WriteLine("AUC RandomForest Value: " + vi);
            return vi;
        }

        public static double TrainAveragePerceptron(IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            AveragedPerceptron classifier = new AveragedPerceptron(true, 0.1f, true, 50, 0.02f, true, mlContext);
            classifier.prepareModel();
            double vi = classifier.trainAndEvaluateModel(10, splitDataView);
            Console.WriteLine("AUC AveragedPreceptron Value: " + vi);
            return vi;
        }

        public static double TrainNaiveBayes(IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            NaiveBayes classifier = new NaiveBayes(mlContext);
            classifier.prepareModel();
            double vi = classifier.trainAndEvaluateModel(10, splitDataView);
            Console.WriteLine("AUC NaiveBayes Value: " + vi);
            return vi;
        }

        public static double TrainMaximumEntropy(IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            MaximumEntropy classifier = new MaximumEntropy(0.05f, 30, mlContext);
            classifier.prepareModel();
            double vi = classifier.trainAndEvaluateModel(10, splitDataView);
            Console.WriteLine("AUC MaximumEntropy Value: " + vi);
            return vi;
        }

        public static double TrainGradientBoosting(IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            GradientBoostingDesicionTree classifier = new GradientBoostingDesicionTree(mlContext);
            classifier.prepareModel();
            double vi = classifier.trainAndEvaluateModel(10, splitDataView);
            Console.WriteLine("AUC GradientBoostingDecisionTree Value: " + vi);
            return vi;
        }

        public static double TrainSVM(IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            SupportVectorMachine classifier = new SupportVectorMachine(10, true, 10, mlContext);
            classifier.prepareModel();
            double vi = classifier.trainAndEvaluateModel(10, splitDataView);
            Console.WriteLine("AUC SupportVectorMachine Value: " + vi);
            return vi;
        }

        public static double TrainLogisticRegression(IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            LogisticRegression classifier = new LogisticRegression(100, 1e-8f, 0.01f, mlContext);
            classifier.prepareModel();
            double vi = classifier.trainAndEvaluateModel(10, splitDataView);
            Console.WriteLine("AUC LogisticRegression Value: " + vi);
            return vi;
        }

        public static double MultipleThreadExecution(string classifiers, IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            ManualResetEvent[] syncEvent = new ManualResetEvent[classifiers.Length];
            for (int k = 0; k< classifiers.Length; k++)
            {
                syncEvent[k] = new ManualResetEvent(false);
            }
            int i = 0;
            double[] result = new double[7];
            if (classifiers.ToLower().Contains('r'))
            {
                Thread tread = new System.Threading.Thread(() =>
                {
                    int j = i;
                    i++;
                    result[0] = TrainRandomForest(splitDataView, mlContext);
                    syncEvent[j].Set();
                });
                tread.Start();
            }

            if (classifiers.ToLower().Contains('p'))
            {
                Thread tread2 = new System.Threading.Thread(() =>
                {
                    int j = i;
                    i++;
                    result[1] = TrainAveragePerceptron(splitDataView, mlContext);
                    syncEvent[j].Set();
                });
                tread2.Start();
            }

            if (classifiers.ToLower().Contains('b'))
            {
                Thread tread3 = new System.Threading.Thread(() =>
                {
                    int j = i;
                    i++;
                    result[2] = TrainNaiveBayes(splitDataView, mlContext);
                    syncEvent[j].Set();
                });
                tread3.Start();
            }

            if (classifiers.ToLower().Contains('m'))
            {
                Thread tread4 = new System.Threading.Thread(() =>
                {
                    int j = i;
                    i++;
                    result[3] = TrainMaximumEntropy(splitDataView, mlContext);
                    syncEvent[j].Set();
                });
                tread4.Start();
            }

            if (classifiers.ToLower().Contains("t"))
            {
                Thread tread5 = new System.Threading.Thread(() =>
                {
                    int j = i;
                    i++;
                    result[4] = TrainGradientBoosting(splitDataView, mlContext);
                    syncEvent[j].Set();
                });
                tread5.Start();
            }

            if (classifiers.ToLower().Contains("s"))
            {
                Thread tread6 = new System.Threading.Thread(() =>
                {
                    int j = i;
                    i++;
                    result[5] = TrainSVM(splitDataView, mlContext);
                    syncEvent[j].Set();
                });
                tread6.Start();
            }

            if (classifiers.ToLower().Contains("l"))
            {
                Thread tread7 = new System.Threading.Thread(() =>
                {
                    int j = i;
                    i++;
                    result[6] = TrainLogisticRegression(splitDataView, mlContext);
                    syncEvent[j].Set();
                });
                tread7.Start();
            }

            ManualResetEvent.WaitAll(syncEvent);

            return result.Max();
        }

        // Pretty-print BinaryClassificationMetrics objects.
        //private static void PrintMetrics(BinaryClassificationMetrics metrics)
        //{
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

        //Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
        //Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
        //Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
        //Console.WriteLine($"Negative Precision: " + $"{metrics.NegativePrecision:F2}");
        //Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
        //Console.WriteLine($"Positive Precision: " + $"{metrics.PositivePrecision:F2}");
        //Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
        //Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        //}
    }
}
