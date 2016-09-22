using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace hli_lib_cli_unittest
{
    [TestClass]
    public class Test_mm_mi_epu8
    {
        [TestMethod]
        public void Test_mm_mi_epu8_1()
        {
            const int nRuns = 1000;
            const double margin = 1E-15;


            Random rand = new Random();
            for (int run = 0; run < nRuns; ++run)
            {
                #region Create Data
                int nElements = rand.Next(2, 1000);
                int nBits1 = rand.Next(1, 4);
                int nBits2 = rand.Next(1, 4);
                int maxValue1 = 1 << nBits1;
                int maxValue2 = 1 << nBits2;
                double maxMi = nBits1 + nBits2;

                List<Byte> data1 = new List<Byte>(nElements);
                List<Byte> data2 = new List<Byte>(nElements);

                for (int i = 0; i < nElements; ++i)
                {
                    data1.Add((Byte)(rand.Next(maxValue1)));
                    data2.Add((Byte)(rand.Next(maxValue2)));
                }
                #endregion

                double mi = hli_cli.HliCli._mm_mi_epu8(data1, nBits1, data2, nBits2);
                Assert.IsTrue(mi >= -margin,     "nElements=" + nElements + "; nBits1=" + nBits1 + "; nBits2=" + nBits2 + ": mi=" + mi + " is smaller than zero");
                Assert.IsTrue(mi <= maxMi, "nElements=" + nElements + "; nBits1=" + nBits1 + "; nBits2=" + nBits2 + ": mi=" + mi + " is larger than maxValue="+maxMi);
            }
        }
    }
}
