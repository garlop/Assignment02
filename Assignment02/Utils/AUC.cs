using System;
using System.Collections.Generic;
using System.Text;

namespace Assignment02.Utils
{
    class AUC
    {
        public static double ComputeMultiClassAUC(IReadOnlyList<IReadOnlyList<double>> confusionMatrix)
        {
            var eval = new BasicEvaluation();
            for (int i = 0; i < confusionMatrix.Count; i++)
            {
                eval.TP += confusionMatrix[i][i];
                for (int j = 0; j < confusionMatrix.Count; j++)
                    if (i != j)
                    {
                        eval.FN += confusionMatrix[i][j];
                        eval.FP += confusionMatrix[j][i];

                        eval.TN += confusionMatrix[j][j];
                    }

            }
            return ComputeTwoClassAUC(eval);
        }

        public static double ComputeTwoClassAUC(BasicEvaluation basicEvaluation)
        {
            double positives = basicEvaluation.TP + basicEvaluation.FN;
            double negatives = basicEvaluation.TN + basicEvaluation.FP;
            var tprate = positives > 0.0 ? basicEvaluation.TP / positives : 1.0;
            var fprate = negatives > 0.0 ? basicEvaluation.TN / negatives : 1.0;
            return (tprate + fprate) / 2.0;
        }
    }

    class BasicEvaluation
    {
        public double TP { get; set; }
        public double TN { get; set; }
        public double FP { get; set; }
        public double FN { get; set; }
    }
}
