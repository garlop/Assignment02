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
using System.Xml.Serialization;
using System.Xml;
using System.Xml.Linq;

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

        //Main program structure, this needs a array of strings in order to work properly
        static int Main(string[] args)
        {
            //Set initial variables which are going to be used later
            string classifiers = "";
            string threading = "";

            //Evaluate the string array in order to determine if the array has 2 elements or not, if not, ask for a proper
            //array
            if (args.Length != 2){
                Console.WriteLine("Please provide the proper number of execution parameters");
                return 1;
            }
            /* If the array args has 2 elements, define the first element as the classifiers types
             * and evaluate if the said classifier includes the letters "rpbmtsl" in their name, if not, consider it 
             * as an invalid classifier selection, also, check the second element of the args array to define 
             * if its a valid threading selection or not
             * The letter meaning are as follow:
             * For the classifiers                                  For the threading
             * r -> Random Forest                                   s -> Single Threading
             * b -> Naive Bayes                                     m -> Multiple Threading
             * p -> Averaged Perception
             * m -> Maximum Entropy
             * t -> Gradient Boosting Decision Tree 
             * s -> Support Vector Machine
             * l -> Logistic Regression
             */
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

            string filePath = Path.Combine(Environment.CurrentDirectory, "Data", "results.xml");

            using (XmlWriter writer = XmlWriter.Create(filePath))
            {
                writer.WriteStartElement("Experiments");
                writer.WriteEndElement();
                writer.Flush();
            }

            /*Prepare the program to evaluate the dataset using the mlContext to train the classifiers and then evaluate
            *the results, first chech if the threading is gonna be single thread or multiple thread
            *execution, to call the corresponding function, and store the result metric valuation in variable "v" 
            */

            int numberOfFiles = 50;

            double[][] aucsForClassifier = new double[7][];

            for (int i = 0; i < 7; i++)
                aucsForClassifier[i] = new double[numberOfFiles];

            //Thresholds for binary classes
            string[] thresholds = { "-0.24", "-0.23", "-0.22", "-0.21", "-0.2", "-0.19", "-0.18", "-0.17", "-0.16", "-0.15",
                "-0.14", "-0.13", "-0.12", "-0.11", "-0.1", "-0.09", "-0.08", "-0.07","-0.06", "-0.05", "-0.04", "-0.03", "-0.02",
                "-0.01", "0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1", "0.11", "0.12", "0.13", "0.14",
                "0.15", "0.16", "0.17", "0.18", "0.19", "0.2", "0.21", "0.22", "0.23", "0.24", "0.25"};

            for (int i = 0; i< numberOfFiles; i++)
            {
                //Set the variable datapath as the path where the .csv is saved to use in futur in the program
                //This line for binary classes
                string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data\\BinaryData", "ex_database_threshold" + thresholds[i]+ ".txt");

                IReadOnlyList<TrainTestData> splitDataView = LoadData(mlContext, _dataPath);

                Tuple<double, double[]> results;

                if (threading == "s")
                {
                    results = SingleThreadExecution(classifiers, splitDataView, mlContext);
                }
                else
                {
                    results = MultipleThreadExecution(classifiers, splitDataView, mlContext);
                }

                Console.WriteLine($"Metric valuation:{ results.Item1 }");

                saveResultToXMLFile(thresholds[i], results.Item1, results.Item2);

                aucsForClassifier[0][i] = results.Item2[0];
                aucsForClassifier[1][i] = results.Item2[1];
                aucsForClassifier[2][i] = results.Item2[2];
                aucsForClassifier[3][i] = results.Item2[3];
                aucsForClassifier[4][i] = results.Item2[4];
                aucsForClassifier[5][i] = results.Item2[5];
                aucsForClassifier[6][i] = results.Item2[6];
            }

            calculateQuartilesForClassifiers(aucsForClassifier, numberOfFiles);

            return 0;
        }

        public static void calculateQuartilesForClassifiers(double[][] aucsForClassifiers, int numberOfFiles)
        {
            for(int i = 0; i<7; i++)
            {
                int Q1Index = numberOfFiles / 4;
                int Q2Index = numberOfFiles / 2;
                int Q3Index = Q1Index + Q2Index;

                Array.Sort(aucsForClassifiers[i]);
                double Q1 = aucsForClassifiers[i][Q1Index];
                double Q2 = aucsForClassifiers[i][Q2Index];
                double Q3 = aucsForClassifiers[i][Q3Index];
                double min = aucsForClassifiers[i].Min();
                double max = aucsForClassifiers[i].Max();

                string filePath = Path.Combine(Environment.CurrentDirectory, "Data", "results.xml");

                XDocument doc = XDocument.Load(filePath);
                XElement root = new XElement("Classifier"+i);
                root.Add(new XElement("Q1", "" + Q1));
                root.Add(new XElement("Q2", "" + Q2));
                root.Add(new XElement("Q3", "" + Q3));
                root.Add(new XElement("Min", "" + min));
                root.Add(new XElement("Max", "" + max));
                doc.Element("Experiments").Add(root);
                doc.Save(filePath);

            }
        }

        public static void saveResultToXMLFile(string fileThreshold, double auc, double[] classifiersauc)
        {
            string filePath = Path.Combine(Environment.CurrentDirectory, "Data", "results.xml");

            XDocument doc = XDocument.Load(filePath);
            XElement root = new XElement("threshold" + fileThreshold);
            root.Add(new XElement("AUC", "" + auc));
            root.Add(new XElement("RFAUC", ""+classifiersauc[0]));
            root.Add(new XElement("APAUC", ""+classifiersauc[1]));
            root.Add(new XElement("NBAUC", ""+classifiersauc[2]));
            root.Add(new XElement("MEAUC", ""+classifiersauc[3]));
            root.Add(new XElement("GBDTAUC", ""+classifiersauc[4]));
            root.Add(new XElement("SVMAUC", ""+classifiersauc[5]));
            root.Add(new XElement("LRAUC", ""+classifiersauc[6]));
            doc.Element("Experiments").Add(root);
            doc.Save(filePath);
        }

        public static IReadOnlyList<TrainTestData> LoadData(MLContext mlContext, string _dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<MinutiaData>(_dataPath, hasHeader: true);

            // Cross validation splits your data randomly into set of "folds", and
            // creates groups of Train and Test sets, where for each group, one fold
            // is the Test and the rest of the folds the Train. So below, we specify
            // Group column as the column containing the sampling keys. If we pass
            // that column to cross validation it would be used to break data into
            // certain chunks.
            var folds = mlContext.Data.CrossValidationSplit(dataView, numberOfFolds: 10);
            return folds;
        }


        //This is the function called before for single thread execution 
        public static Tuple<double, double[]> SingleThreadExecution(string classifiers, IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
        {
            /*First select the classifier type by comparing the char values as described before, 
             * generate a new instance of the classifier type named classifier, which will have different parameters
             * depending on the classifier, and pass the data "mlContext" to the classifier. After that prepare the model
             * using the function defined inside the classifier class and evaluate the classifier using the corresponding
             * function.
             * Finally, output to console the value of each classifier evaluated and return the maximum value for the 
             * metric valuation of the previous evaluated classifiers as the "v" value 
             */
            double[] result = new double[7];
            double v = 0;
            if (classifiers.ToLower().Contains('r'))
            {
                RandomForest classifier = new RandomForest(0.8, 0.1, 50, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC RandomForest Value: " + vi);
                v = Math.Max(v, vi);
                result[0] = vi;
            }

            if (classifiers.ToLower().Contains('p'))
            {
                AveragedPerceptron classifier = new AveragedPerceptron(true, 0.1f, true, 50, 0.02f, true, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC AveragedPreceptron Value: " + vi);
                v = Math.Max(v, vi);
                result[1] = vi;
            }

            if (classifiers.ToLower().Contains('b'))
            {
                NaiveBayes classifier = new NaiveBayes(mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC NaiveBayes Value: " + vi);
                v = Math.Max(v, vi);
                result[2] = vi;
            }

            if (classifiers.ToLower().Contains('m'))
            {
                MaximumEntropy classifier = new MaximumEntropy(0.05f, 30, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC MaximumEntropy Value: " + vi);
                v = Math.Max(v, vi);
                result[3] = vi;
            }

            if (classifiers.ToLower().Contains("t"))
            {
                GradientBoostingDesicionTree classifier = new GradientBoostingDesicionTree(mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC GradientBoostingDecisionTree Value: " + vi);
                v = Math.Max(v, vi);
                result[4] = vi;
            }

            if (classifiers.ToLower().Contains("s"))
            {
                SupportVectorMachine classifier = new SupportVectorMachine(10, true, 10, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC SupportVectorMachine Value: " + vi);
                v = Math.Max(v, vi);
                result[5] = vi;
            }

            if (classifiers.ToLower().Contains("l"))
            {
                LogisticRegression classifier = new LogisticRegression(100, 1e-8f, 0.01f, mlContext);
                classifier.prepareModel();
                double vi = classifier.trainAndEvaluateModel(10, splitDataView);
                Console.WriteLine("AUC LogisticRegression Value: " + vi);
                v = Math.Max(v, vi);
                result[6] = vi;
            }

            return Tuple.Create(v, result);
        }

        /*The next function definitions are used for the multithread processing, where each definition is a different
        *classifier, which are going to be called later by the multithread algorithm
        */
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


        /* This routine is in charge of making the processing for the different classifiers in different threads
         * using as classifier selection the "classifiers" string and evaluating each of them according to the 
         * before declared functions.
         * At the end it will wait all the threads to end the processing in order to calculate the highest AUC value
         * reached in the different tests
         */
        public static Tuple<double, double[]> MultipleThreadExecution(string classifiers, IReadOnlyList<TrainTestData> splitDataView, MLContext mlContext)
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

            return Tuple.Create(result.Max(), result);
        }
    }
}
