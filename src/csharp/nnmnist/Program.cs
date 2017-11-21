using System.IO;
using nnmnist.Application;
using nnmnist.Common;

namespace nnmnist
{
    static class Program
    {
        static int Main(string[] args)
        {
            // get the config in the following order:
            //   file specified by the first command line argument
            //   "default.json"
            //   defaults
            Config conf;
			if (args.Length > 0)
			{
			    if (File.Exists(args[0]))
			    {
			        conf = Config.ReadFromJson(args[0]);
			    }
			    else
			    {
			        return -1;
			    }
			}
			else if (File.Exists("default.json"))
			{
				conf = Config.ReadFromJson("default.json");
			}
			else
			{
				conf = new Config();
			}

            //conf.WriteToJson("default.json");
            // init logger
            Global.Logger = new Logger(true);
			Global.Logger.SetAppName(conf.Name());

            // print the config
            Global.Logger.WriteLine($"{Config.Sep}{Config.Sep}");
			Global.Logger.Write(conf.ToString());
			Global.Logger.WriteLine($"{Config.Sep}{Config.Sep}");

            // run the application
            var mnist = new Mnist(conf);
            mnist.Run();
            return 0;
        }
        
    }
}
