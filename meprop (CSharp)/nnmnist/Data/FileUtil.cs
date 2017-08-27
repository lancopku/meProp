using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using nnmnist.Simple;
using System.Linq;

namespace nnmnist.Common
{
    static class FileUtil
    {

		public static List<Example> ReadFromFile(string imgFile, string lbFile)
		{
			var imgBytes = File.ReadAllBytes(imgFile);
			var laBytes = File.ReadAllBytes(lbFile);

			int nImg = GetFromFourBytes(imgBytes, 4);
			int nLabel = GetFromFourBytes(laBytes, 4);
			if (nImg != nLabel)
			{
				throw new Exception("data number mismatch");
			}
			int nRow = GetFromFourBytes(imgBytes, 8);
			int nCol = GetFromFourBytes(imgBytes, 12);
			var set = new List<Example>();
			//Image[] set = new Image[nImg];
			for (var i = 0; i < nImg; i++)
			{
				var values = new int[nRow * nCol];
				var label = GetFromByte(laBytes, 8 + i);
				//set[i] = new Image(nRow, nCol);
				//set[i].label = GetFromByte(laBytes, 8 + i);
				for (var j = 0; j < nRow * nCol; j++)
				{
					values[j] = GetFromByte(imgBytes, 16 + (nRow * nCol) * i + j);
				}
				set.Add(new Example(values, label));

			}
			return set;

		}

		private static int GetFromFourBytes(byte[] arr, int idx)
		{
			if (BitConverter.IsLittleEndian)
			{
				byte[] tmp = new byte[4];
				Array.Copy(arr, idx, tmp, 0, 4);
				tmp = tmp.Reverse().ToArray();
				return BitConverter.ToInt32(tmp, 0);
			}
			return BitConverter.ToInt32(arr, idx);
		}

		private static int GetFromByte(byte[] arr, int idx)
		{
			return (int)arr[idx];
		}
	}
}
