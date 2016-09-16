using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace hli_lib_cli_unittest
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void Test_mm_corr_epu8()
        {
            Random rand = new Random();

            const int nElements = 16 * 64;
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

            double corr1 = StatsLibCli.Class1._mm_corr_epu8(data1, data2);
            double corr2 = StatsLibCli.Class1._mm_corr_pd(data1d, data2d);

            double diff = Math.Abs(corr1 - corr2);
            Assert.IsTrue(diff < 0.00000001, "corr1=" + corr1 + "; corr2=" + corr2 + "; diff=" + diff);
        }
    }
}
