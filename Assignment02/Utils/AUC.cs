using System;
using System.Collections.Generic;
using System.Text;

namespace Assignment02.Utils
{
    class AUC
    {

        //This function is called by the classifiers rutines in order to compute the AUC of each classifier
        public static double ComputeMultiClassAUC(IReadOnlyList<IReadOnlyList<double>> confusionMatrix)
        {
            //First creat a evaluation variable named eval to gather the TP, TN, FP and FN values
            var eval = new BasicEvaluation();

            //This nested for cycles gather the information from the confusion matrix for the true and false 
            //positives and negatives adding them to the corresponding category according to the position in the
            //confusion matrix 
            //After the complete TP, TN, FP and FN are counted this are passed to the ComputeTwoClassAUC function
            //for the final evaluation
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


        /*This routine calculates the AUC by checking the fraction of correct classified elements, for this
         * the first thing made is to calculate the total of positive and negatives elements, not regarding if 
         * are true or false classified, after that, it is calculated the correct classified portion for calculating
         * the final AUC which is returned to the classifiers that called the function 
         */
        public static double ComputeTwoClassAUC(BasicEvaluation basicEvaluation)
        {
            double positives = basicEvaluation.TP + basicEvaluation.FN;
            double negatives = basicEvaluation.TN + basicEvaluation.FP;
            var tprate = positives > 0.0 ? basicEvaluation.TP / positives : 1.0;
            var fprate = negatives > 0.0 ? basicEvaluation.TN / negatives : 1.0;
            return (tprate + fprate) / 2.0;
        }
    }


    //This is a class definition just to set the parameters of True Positives / True Negatives / False positives
    // and False negatives values obtained by each classifier while evaluating the AUC
    class BasicEvaluation
    {
        public double TP { get; set; }
        public double TN { get; set; }
        public double FP { get; set; }
        public double FN { get; set; }
    }
}
