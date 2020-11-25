using Microsoft.ML.Data;

namespace mi
{
    internal class SalaryPred
    {
        [ColumnName("Score")]
        public float ProdictedSalary { get; set; }
    }
}