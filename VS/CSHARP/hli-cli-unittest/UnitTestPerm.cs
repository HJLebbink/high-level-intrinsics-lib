using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace hli_lib_cli_unittest
{
	[TestClass]
	public class UnitTestPerm
	{
		[TestMethod]
		public void Test_mm_perm_epu8()
		{
            UInt32[,] destination = new UInt32[100, 100];
            
            for (int i = 0; i<100; ++i)
            {
                for (int j = 0; j<100; ++j)
                {
                    destination[i, j] = 0;
                }
            }

        }
	}
}
