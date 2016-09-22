using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace hli_lib_cli_unittest
{
    [TestClass]
    public class Test_mm_entropy_epu8
    {
        [TestMethod]
        public void Test_mm_entropy_epu8_1()
        {
            const double margin = 1E-15;

            const int nRuns = 10000;
            Random rand = new Random();

            for (int run = 0; run < nRuns; ++run)
            {
                #region Create Data
                int nElements = rand.Next(2, 1000);
                int nBits1 = rand.Next(1, 4);
                int maxValue1 = 1 << nBits1;
                double maxH = nBits1;

                List<Byte> data1 = new List<Byte>(nElements);

                for (int i = 0; i < nElements; ++i)
                {
                    data1.Add((Byte)(rand.Next(maxValue1)));
                }
                #endregion

                double h1 = hli_cli.HliCli._mm_entropy_epu8(data1, nBits1);

                Assert.IsTrue(h1 >= -margin, "nElements=" + nElements + "; nBits1=" + nBits1 + ": H=" + h1 + " is smaller than zero");
                Assert.IsTrue(h1 <= maxH,   "nElements=" + nElements + "; nBits1=" + nBits1 + ": H=" + h1 + " is larger than maxH=" + maxH);
            }
        }

        [TestMethod]
        public void Test_mm_entropy_epu8_2()
        {
            const double margin = 1E-15;

            const int nRuns = 1000;
            Random rand = new Random();

            for (int run = 0; run < nRuns; ++run)
            {
                #region Create Data
                int nElements = rand.Next(2, 1000);
                int nBits1 = rand.Next(1, 4);
                int nBits2 = rand.Next(1, 4);
                int maxValue1 = 1 << nBits1;
                int maxValue2 = 1 << nBits2;
                double maxH1 = nBits1;
                double maxH2 = nBits2;
                double maxMi = nBits1 + nBits2;

                List<Byte> data1 = new List<Byte>(nElements);
                List<Byte> data2 = new List<Byte>(nElements);

                for (int i = 0; i < nElements; ++i)
                {
                    data1.Add((Byte)(rand.Next(maxValue1)));
                    data2.Add((Byte)(rand.Next(maxValue2)));
                }
                #endregion

                double h1 = hli_cli.HliCli._mm_entropy_epu8(data1, nBits1);
                double h2 = hli_cli.HliCli._mm_entropy_epu8(data2, nBits2);
                double h1Plush2 = h1 + h2;
                double h1Andh2 = hli_cli.HliCli._mm_entropy_epu8(data1, nBits1, data2, nBits2);

                Assert.IsTrue(h1 >= -margin, "nElements=" + nElements + "; nBits1=" + nBits1 + ": H1=" + h1 + " is smaller than zero");
                Assert.IsTrue(h1 <= maxH1,   "nElements=" + nElements + "; nBits1=" + nBits1 + ": H1=" + h1 + " is larger than maxH2=" + maxH1);
                Assert.IsTrue(h2 >= -margin, "nElements=" + nElements + "; nBits2=" + nBits2 + ": H2=" + h2 + " is smaller than zero");
                Assert.IsTrue(h2 <= maxH2,   "nElements=" + nElements + "; nBits2=" + nBits2 + ": H2=" + h2 + " is larger than maxH1=" + maxH2);
                Assert.IsTrue(h1Andh2 >= -margin, "nElements=" + nElements + "; nBits1=" + nBits1 + "; nBits2=" + nBits2 + ": h1Andh2=" + h1Andh2 + " is smaller than zero");
                Assert.IsTrue(h1Andh2 <= maxMi,   "nElements=" + nElements + "; nBits1=" + nBits1 + "; nBits2=" + nBits2 + ": h1Andh2=" + h1Andh2 + " is larger than maxMi=" + maxMi);
                Assert.IsTrue(h1Plush2 >= h1Andh2, "nElements=" + nElements + "; nBits1=" + nBits1 + "; nBits2=" + nBits2 + ": h1Plush2=" + h1Plush2 + ", h1Andh2=" + h1Andh2);
            }
        }
    }
}
