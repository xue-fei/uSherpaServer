// See https://aka.ms/new-console-template for more information
using System;

namespace uSherpaServer
{
    internal class Program
    {
        // 声明配置和识别器变量
        static SherpaNcnn.OnlineRecognizer recognizer;
        static SherpaNcnn.OnlineStream onlineStream;

        static string tokensPath = "tokens.txt";
        static string encoderParamPath = "encoder_jit_trace-pnnx.ncnn.param";
        static string encoderBinPath = "encoder_jit_trace-pnnx.ncnn.bin";
        static string decoderParamPath = "decoder_jit_trace-pnnx.ncnn.param";
        static string decoderBinPath = "decoder_jit_trace-pnnx.ncnn.bin";
        static string joinerParamPath = "joiner_jit_trace-pnnx.ncnn.param";
        static string joinerBinPath = "joiner_jit_trace-pnnx.ncnn.bin";
        static int numThreads = 2;
        static string decodingMethod = "greedy_search";

        static string modelPath;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            modelPath = Environment.CurrentDirectory + "/sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16";
            // 初始化配置
            SherpaNcnn.OnlineRecognizerConfig config = new SherpaNcnn.OnlineRecognizerConfig
            {
                FeatConfig = { SampleRate = 16000, FeatureDim = 80 },
                ModelConfig = {
                Tokens = Path.Combine(modelPath,tokensPath),
                EncoderParam =  Path.Combine(modelPath,encoderParamPath),
                EncoderBin =Path.Combine(modelPath, encoderBinPath),
                DecoderParam =Path.Combine(modelPath, decoderParamPath),
                DecoderBin = Path.Combine(modelPath, decoderBinPath),
                JoinerParam = Path.Combine(modelPath,joinerParamPath),
                JoinerBin =Path.Combine(modelPath,joinerBinPath),
                UseVulkanCompute = 0,
                NumThreads = numThreads
            },
                DecoderConfig = {
                DecodingMethod = decodingMethod,
                NumActivePaths = 4
            },
                EnableEndpoint = 1,
                Rule1MinTrailingSilence = 2.4F,
                Rule2MinTrailingSilence = 1.2F,
                Rule3MinUtteranceLength = 20.0F
            };

            // 创建识别器和在线流
            recognizer = new SherpaNcnn.OnlineRecognizer(config);

            onlineStream = recognizer.CreateStream();
        }
    }
}