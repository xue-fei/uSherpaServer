// See https://aka.ms/new-console-template for more information
using System;

namespace uSherpaServer
{
    internal class Program
    {
        // 声明配置和识别器变量
        private SherpaNcnn.OnlineRecognizer recognizer;
        private SherpaNcnn.OnlineStream onlineStream;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
        }
    }
}