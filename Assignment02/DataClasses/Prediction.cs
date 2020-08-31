using System;
using System.Collections.Generic;
using System.Text;

namespace Assignment02.DataClasses
{
    // Class used to capture predictions.
    public class Prediction
    {
        // Original label.
        public uint Label { get; set; }
        // Predicted label from the trainer.
        public uint PredictedLabel { get; set; }
    }
}
