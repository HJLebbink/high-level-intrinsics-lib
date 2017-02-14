using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace hli_lib_cli_unittest
{
    [TestClass]
    public class Test_mm_corr_epu8
    {
        [TestMethod]
        public void Test_mm_corr_epu8_1()
        {
            const bool hasMissingValues = false;
            List<Byte> data1 = new List<Byte> { 0, 0, 1 };
            List<Byte> data2 = new List<Byte> { 3, 1, 0 };
            double expected_Result = -2.0 / Math.Sqrt(7.0);//   -0.755928946018454454429;
            double observed_Result = hli_cli.HliCli._mm_corr_epu8(data1, data2, hasMissingValues);
            double diff = Math.Abs(expected_Result - observed_Result);
            const double threshold = 1E-14;
            Console.WriteLine("diff=" + diff+"; threshold="+threshold);
            Assert.IsTrue(diff <= threshold, "expected_Result=" + expected_Result + "; observed_Result=" + observed_Result + "; diff=" + diff);
        }

        [TestMethod]
        public void Test_mm_corr_epu8_equal_pd()
        {
            const bool hasMissingValues = false;
            const int nRuns = 100000;
            const double threshold = 1E-25;

            Random rand = new Random();
            for (int run = 0; run < nRuns; ++run)
            {
                int nElements = rand.Next(2, 1000);
                const int maxValue = 4;

                List<Byte> data1 = new List<Byte>(nElements);
                List<Byte> data2 = new List<Byte>(nElements);
                List<Double> data1d = new List<Double>(nElements);
                List<Double> data2d = new List<Double>(nElements);

                for (int i = 0; i < nElements; ++i)
                {
                    data1.Add((Byte)(rand.Next(maxValue)));
                    data2.Add((Byte)(rand.Next(maxValue)));

                    data1d.Add((Double)data1[i]);
                    data2d.Add((Double)data2[i]);
                }

                double corr1 = hli_cli.HliCli._mm_corr_epu8(data1, data2, hasMissingValues);
                double corr2 = hli_cli.HliCli._mm_corr_pd(data1d, data2d, hasMissingValues);

                double diff = Math.Abs(corr1 - corr2);
                if (Double.IsNaN(diff))
                {
                    Console.WriteLine("run "+run +": corr1=" + corr1 + "; corr2=" + corr2 + "; diff=" + diff +"; nElements="+nElements);
                    for (int i = 0; i<nElements; ++i)
                    {
                        Console.WriteLine("i " + i + ": element1=" + data1[i]+"; element2="+data2[i]);
                    }
                } else
                {
                    Assert.IsTrue(diff <= threshold, "corr1=" + corr1 + "; corr2=" + corr2 + "; diff=" + diff + "; threshold=" + threshold +"; nElements =" + nElements);
                }
            }
        }
    }
}
