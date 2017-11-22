using nnmnist.Common;
using System;

namespace nnmnist
{
	static class Global
    {
        public static readonly long TimeStamp; // timestamp to identify the "run"
        public static Logger Logger; // logger that prints both to a file and the console

        static Global()
        {
            TimeStamp = DateTime.Now.ToFileTime();
        }
    }
}