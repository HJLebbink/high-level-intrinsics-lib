using System;
using System.Collections.Generic;
using System.Reflection;

using hli_cli;

namespace hli_cli_main
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            DateTime startTime = DateTime.Now;
            Assembly thisAssem = typeof(Program).Assembly;
            AssemblyName thisAssemName = thisAssem.GetName();
            System.Version ver = thisAssemName.Version;
            Console.WriteLine(string.Format("Loaded hli-cli-main version {0}.", ver));

            Test_mm_mi_epu8_2();

            double elapsedSec = (double)(DateTime.Now.Ticks - startTime.Ticks) / 10000000;
            Console.WriteLine(string.Format("Elapsed time " + elapsedSec + " sec"));
            Console.WriteLine(string.Format("Press any key to continue."));
            Console.ReadKey();
        }

        private static (List<Byte> Data1, int NBits1, List<Byte> Data2, int NBits2, int NElements, bool Has_Missing_Values) Create_Random_Data(Random rand, double chance_Missing_Values, Byte missing_value)
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
            return (Data1: data1, NBits1: nBits1, Data2: data2, NBits2: nBits2, NElements: nElements, Has_Missing_Values: chance_Missing_Values > 0);


            Byte Create_Data_Element(int maxValue)
            {
                if ((chance_Missing_Values > 0) && (rand.NextDouble() < chance_Missing_Values))
                {
                    return missing_value;
                }
                return (Byte)(rand.Next(maxValue));
            }
        }

        private static void Test_mm_mi_epu8_2()
        {
            const int nRuns = 500000;
            const double chance_Missing_Values = 0.05;
            const double margin = 1E-15;
            const Byte missing_Value = 0xFF;

            Random rand = new Random();
            for (int run = 0; run < nRuns; ++run)
            {
                var data = Create_Random_Data(rand, chance_Missing_Values, missing_Value);
                double maxMi = data.NBits1 + data.NBits2;

                double mi = HliCli._mm_mi_epu8(data.Data1, data.NBits1, data.Data2, data.NBits2, data.Has_Missing_Values);
                if (!(mi >= -margin)) Console.WriteLine("nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is smaller than zero");
                if (!(mi <= maxMi)) Console.WriteLine("nElements=" + data.NElements + "; nBits1=" + data.NBits1 + "; nBits2=" + data.NBits2 + ": mi=" + mi + " is larger than maxValue=" + maxMi);



                { // remove missing data
                    List<Byte> data1b = new List<Byte>();
                    List<Byte> data2b = new List<Byte>();

                    for (int i = 0; i<data.NElements; ++i)
                    {
                        if ((data.Data1[i] == missing_Value) || (data.Data2[i] == missing_Value))
                        {
                            // do nothing
                        }
                        else
                        {
                            data1b.Add(data.Data1[i]);
                            data2b.Add(data.Data2[i]);
                        }
                    }

                    double mi2 = HliCli._mm_mi_epu8(data1b, data.NBits1, data2b, data.NBits2, false);
                    double mi3 = HliCli._mm_mi_epu8(data1b, data.NBits1, data2b, data.NBits2, true); // there are no missing values but it should yield the same results.

                    if (mi != mi2) Console.WriteLine("run=" + run + "; mi=" + mi + "; mi2=" + mi2);
                    if (mi != mi3) Console.WriteLine("run=" + run + "; mi=" + mi + "; mi3=" + mi2);
                }
            }
        }
    }
}
