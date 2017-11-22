using System;
using System.IO;
using System.Text;

namespace nnmnist.Common
{
    class Logger
    {
        // this print to both a file and the console
        // but the code is bad, very bad

        public delegate void OutputEventHandler(object sender, OutputEventArgs e);
        public delegate void OutputConsoleEventHandler(object sender, OutputEventArgs e);
        public delegate void OutputLineEventHandler(object sender, OutputEventArgs e);

        public event OutputEventHandler Output;
        public event OutputLineEventHandler OutputLine;
        public event OutputConsoleEventHandler OutputConsole;

        private string _filename;

        public void SetAppName(string name)
        {
            _filename = name + "-" + _filename;
        }

        public Logger(bool writeToFile)
        {
            Output += (sender, args) => { Console.Write(args.Format); };
            OutputConsole += (sender, args) => { Console.Write(args.Format); };
            OutputLine += (sender, args) => { Console.WriteLine(args.Format); };
            if (writeToFile)
            {
                _filename = $"log-{Global.TimeStamp}.txt";
                OutputLine +=
                    (sender, args) =>
                    {
                        File.AppendAllText(_filename, args.Format + Environment.NewLine,
                            Encoding.UTF8);
                    };
                Output +=
                    (sender, args) =>
                    {
                        File.AppendAllText(_filename, args.Format,
                            Encoding.UTF8);
                    };
            }
            else
            {
                _filename = null;
            }
        }

        public class OutputEventArgs : EventArgs
        {
            public readonly string Format;

            public OutputEventArgs(string format)
            {
                Format = format;
            }
        }

        protected virtual void OnWrite(OutputEventArgs e)
        {
            Output?.Invoke(this, e);
        }

        protected virtual void OnWriteLine(OutputEventArgs e)
        {
            OutputLine?.Invoke(this, e);
        }

        protected virtual void OnWriteConsole(OutputEventArgs e)
        {
            OutputConsole?.Invoke(this, e);
        }

        public void WriteLine(string format)
        {
            OnWriteLine(new OutputEventArgs(format));
        }

        public void WriteLine()
        {
            OnWriteLine(new OutputEventArgs(""));
        }

        public void Write(string format)
        {
            OnWrite(new OutputEventArgs(format));
        }

        public void WriteConsole(string format)
        {
            OnWriteConsole(new OutputEventArgs(format));
        }
    }
}
