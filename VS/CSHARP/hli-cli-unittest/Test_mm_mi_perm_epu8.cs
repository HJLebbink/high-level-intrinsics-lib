using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace hli_lib_cli_unittest
{
    [TestClass]
    public class Test_mm_mi_perm_epu8
    {
        [TestMethod]
        public void Test_mm_mi_perm_epu8_1()
        {
            const double margin = 1E-15;

            const int nRuns = 100;
            const int nPermutations = 512;
            Random rand = new Random();

            List<double> results = new List<double>(nPermutations);
            for (int i = 0; i < nPermutations; ++i) results.Add(0);
            uint[] randInts = { (uint)rand.Next(), (uint)rand.Next(), (uint)rand.Next(), (uint)rand.Next() };


            for (int run = 0; run < nRuns; ++run)
            {
                #region Create Data
                int nElements = rand.Next(2, 1000);
                int nBits1 = rand.Next(1, 4);
                int nBits2 = rand.Next(1, 4);
                int maxValue1 = 1 << nBits1;
                int maxValue2 = 1 << nBits2;
                double maxMi = nBits2 + nBits1;

                List<Byte> data1 = new List<Byte>(nElements);
                List<Byte> data2 = new List<Byte>(nElements);

                for (int i = 0; i < nElements; ++i)
                {
                    data1.Add((Byte)(rand.Next(maxValue1)));
                    data2.Add((Byte)(rand.Next(maxValue2)));
                }
                #endregion


                double mi = hli_cli.HliCli._mm_mi_epu8(data1, nBits1, data2, nBits2, false);
                Assert.IsTrue(mi >= -margin, "nElements=" + nElements + "; nBits1=" + nBits1 + "; nBits2=" + nBits2 + ": mi=" + mi + " is smaller than zero");
                Assert.IsTrue(mi <= maxMi,   "nElements=" + nElements + "; nBits1=" + nBits1 + "; nBits2=" + nBits2 + ": mi=" + mi + " is larger than maxValue=" + maxMi);

                hli_cli.HliCli._mm_mi_epu8_perm(data1, nBits1, data2, nBits2, false, results, randInts);

                for (int j = 0; j<nPermutations; ++j)
                {
                    double result = results[j];
                    Assert.IsTrue(result >= -margin, "nElements=" + nElements + "; nBits1=" + nBits1 + "; nBits2=" + nBits2 + "; permutation " + j + ": mi=" + result + " is smaller than zero");
                    Assert.IsTrue(result <= maxMi,   "nElements=" + nElements + "; nBits1=" + nBits1 + "; nBits2=" + nBits2 + "; permutation " + j + ": mi=" + result + " is larger than maxValue=" + maxMi);
                }
            }
        }
    }
}
