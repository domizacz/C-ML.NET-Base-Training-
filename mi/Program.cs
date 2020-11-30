using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace mi
{
    class Program
    {
        static void Main(string[] args)
        {


            //Step 1. Create an ML Context
            var context = new MLContext();

            //Step 2. Read in the input data from a text file for model training
            IDataView trainingData = context.Data.LoadFromTextFile<SalaryData>("./s.txt", separatorChar: ',', hasHeader: true);

            var testtraondata = context.Data.TrainTestSplit(trainingData, testFraction: 0.2);

            var pipeline = context.Transforms.Concatenate("Features", "YearsExperience")
                 .Append(context.Regression.Trainers.LbfgsPoissonRegression());
            var model = pipeline.Fit(testtraondata.TrainSet);

            var predictions = model.Transform(testtraondata.TestSet);

            var metric = context.Regression.Evaluate(predictions);

            Console.WriteLine($"R 2 - {metric.RSquared}");


            var newdata = new SalaryData
            {
                YearsExperience = 2
        };
            var predictionfun = context.Model.CreatePredictionEngine<SalaryData, SalaryPred>(model);
            var prodiction = predictionfun.Predict(newdata);

            Console.WriteLine($"Prodiction -{prodiction.ProdictedSalary} ");

            Console.ReadLine();
            
            
            //var context = new MLContext();
            //var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            //var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

            //var features = split.TrainSet.Schema
            //    .Select(col => col.Name)
            //    .Where(colName => colName != "Label" && colName != "oceanproximity")
            //    .ToArray();
            //var pipeline = context.Transforms.Text.FeaturizeText("Text", "oceanproximity")
            //    .Append(context.Transforms.Concatenate("Features", features))
            //    .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            //var model = pipeline.Fit(split.TrainSet);

            //var predictions = model.Transform(split.TestSet);

            //var metrics = context.Regression.Evaluate(predictions);

            //Console.WriteLine($"R^2 - {metrics.RSquared}");

            //var newdata = new HousingData
            //{
            //oceanproximity = "OCEAN"
            //};

            //var predictionFunc = context.Model.CreatePredictionEngine<HousingData, SallePRodukction >(model);
            //var prodiction = predictionFunc.Predict(newdata);
            //Console.WriteLine($"Prediction - {prodiction.prodoceanproximity}");

            //Console.ReadLine();
        }
    }

   
}
