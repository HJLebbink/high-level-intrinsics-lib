using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace hli_lib_cli_unittest
{
    [TestClass]
    public class Test_mm_mi_epu8
    {

        private (List<Byte> Data1, int NBits1, List<Byte> Data2, int NBits2, int NElements, bool Has_Missing_Values) Create_Random_Data(Random rand, double chance_Missing_Values)
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
                data1.Add(Create_Data_Element(maxValue1));
                data2.Add(Create_Data_Element(maxValue2));
            }
            return (Data1:data1, NBits1:nBits1, Data2:data2, NBits2:nBits2, NElements:nElements, Has_Missing_Values: chance_Missing_Values > 0);


            Byte Create_Data_Element(int maxValue)
            {
                return (Byte)(rand.Next(maxValue1));
                /*
                if (chance_Missing_Values > 0)
                {

                }
                else
                {

                }
                */
            }
        }
        
        /// <summary>
        /// Test MI epu8 with no missing values
        /// </summary>
        [TestMethod]
        public void Test_mm_mi_epu8_1()
        {
            const int nRuns = 50000;
            const double chance_Missing_Values = 0;
            const double margin = 1E-15;

            Random rand = new Random();
            for (int run = 0; run < nRuns; ++run)
            {
                var data = Create_Random_Data(rand, chance_Missing_Values);
                double maxMi = data.NBits1 + data.NBits2;

                double mi = hli_cli.HliCli._mm_mi_epu8(data.Data1, data.NBits1, data.Data2, data.NBits2, data.Has_Missing_Values);
                Assert.IsTrue(mi >= -margin,     "nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is smaller than zero");
                Assert.IsTrue(mi <= maxMi, "nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is larger than maxValue="+maxMi);
            }
        }

        /// <summary>
        /// Test MI epu8 with missing values
        /// </summary>
        [TestMethod]
        public void Test_mm_mi_epu8_2()
        {
            const int nRuns = 5000;
            const double chance_Missing_Values = 0.05;
            const double margin = 1E-15;

            Random rand = new Random();
            for (int run = 0; run < nRuns; ++run)
            {
                var data = Create_Random_Data(rand, chance_Missing_Values);
                double maxMi = data.NBits1 + data.NBits2;

                double mi = hli_cli.HliCli._mm_mi_epu8(data.Data1, data.NBits1, data.Data2, data.NBits2, data.Has_Missing_Values);
                Assert.IsTrue(mi >= -margin, "nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is smaller than zero");
                Assert.IsTrue(mi <= maxMi, "nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is larger than maxValue=" + maxMi);
            }
        }
    }
}
