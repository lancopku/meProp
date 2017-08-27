using nnmnist.Common;
using System;

namespace nnmnist
{
	static class Global
    {
        public static readonly long TimeStamp;
        public static Logger Logger;

        static Global()
        {
            TimeStamp = DateTime.Now.ToFileTime();
        }
    }
}