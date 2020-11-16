using Microsoft.ML.Data;

namespace mi
{
    internal class HousingData
    {
        [LoadColumn(0)]
        public float Longitude { get; set; }
        [LoadColumn(1)]
        public float latitude { get; set; }
        [LoadColumn(2)]
        public float housingMedianAge { get; set; }
        [LoadColumn(3)]
        public float totalRooms { get; set; }
        [LoadColumn(4)]
        public float totalBedrooms { get; set; }
        [LoadColumn(5)]
        public float population { get; set; }
        [LoadColumn(6)]
        public float households { get; set; }
        [LoadColumn(7)]
        public float medianIncome { get; set; }
        [LoadColumn(8), ColumnName("Label")]
        public float medianHouseValue { get; set; }
        [LoadColumn(9)]
        public string oceanproximity { get; set; }
    }
}
