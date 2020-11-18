﻿using Microsoft.ML;
using System;
using System.Linq;

namespace mi
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');
            
            var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var features = split.TrainSet.Schema
                .Select(col => col.Name)
                .Where(colName => colName != "Label" && colName != "oceanproximity")
                .ToArray();
            var pipeline = context.Transforms.Text.FeaturizeText("Text", "oceanproximity")
                .Append(context.Transforms.Concatenate("Features", features))
                .Append(context.Transforms.Concatenate("Features", "Features", "Text"))
                .Append(context.Transforms.Concatenate("Features", "Text", "Features"))
                .Append(context.Transforms.Concatenate("Text", "Features", "Features"))
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"R^2 - {metrics.RSquared}");
            Console.ReadLine();
        }
    }
}
