using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace hli_lib_cli_unittest
{
    [TestClass]
    public class Test_mm_mi_epu8
    {
        private (List<Byte> Data1, int NBits1, List<Byte> Data2, int NBits2, int NElements) Create_Random_Data(Random rand, bool has_Missing_Values)
        {
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
            return (Data1:data1, NBits1:nBits1, Data2:data2, NBits2:nBits2, NElements:nElements);
        }

        [TestMethod]
        public void Test_mm_mi_epu8_1()
        {
            const int nRuns = 50000;
            const bool has_Missing_Values = false;
            const double margin = 1E-15;

            Random rand = new Random();
            for (int run = 0; run < nRuns; ++run)
            {
                var data = Create_Random_Data(rand, has_Missing_Values);
                double maxMi = data.NBits1 + data.NBits2;

                double mi = hli_cli.HliCli._mm_mi_epu8(data.Data1, data.NBits1, data.Data2, data.NBits2, has_Missing_Values);
                Assert.IsTrue(mi >= -margin,     "nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is smaller than zero");
                Assert.IsTrue(mi <= maxMi, "nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is larger than maxValue="+maxMi);
            }
        }

        [TestMethod]
        public void Test_mm_mi_epu8_2()
        {
            const int nRuns = 50000;
            const bool has_Missing_Values = true;
            const double margin = 1E-15;

            Random rand = new Random();
            for (int run = 0; run < nRuns; ++run)
            {
                var data = Create_Random_Data(rand, has_Missing_Values);
                double maxMi = data.NBits1 + data.NBits2;

                double mi = hli_cli.HliCli._mm_mi_epu8(data.Data1, data.NBits1, data.Data2, data.NBits2, has_Missing_Values);
                Assert.IsTrue(mi >= -margin, "nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is smaller than zero");
                Assert.IsTrue(mi <= maxMi, "nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is larger than maxValue=" + maxMi);
            }
        }
    }
}
