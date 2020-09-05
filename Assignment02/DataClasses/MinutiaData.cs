using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Assignment02.DataClasses
{

    //This is the definition for the minutiadata class, which is used to set and get the features of each minutia
    public class MinutiaData
    {
        [LoadColumn(0, 243), VectorType(244), ColumnName("Features")]
        public float[] features { get; set; }

        [LoadColumn(245), ColumnName("Label")]
        public uint Label { get; set; }
    }
}
