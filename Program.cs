using Fleck;
using Newtonsoft.Json;
using SherpaOnnx;
using System.Text;

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
        static int numThreads = 1;
        static string decodingMethod = "greedy_search";

        static string modelPath;
        static float sampleRate = 16000;

        static IWebSocketConnection client;

        static SherpaOnnx.OfflinePunctuation offlinePunctuation = null;
        static SherpaOnnx.VoiceActivityDetector vad = null;

        static void Main(string[] args)
        {
            //需要将此文件夹拷贝到exe所在的目录
            modelPath = Environment.CurrentDirectory + "/sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16";
            // 初始化配置
            SherpaNcnn.OnlineRecognizerConfig config = new SherpaNcnn.OnlineRecognizerConfig
            {
                FeatConfig = { SampleRate = sampleRate, FeatureDim = 80 },
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

            SherpaOnnx.SherpaOnnxOfflinePunctuationConfig soopc = new SherpaOnnx.SherpaOnnxOfflinePunctuationConfig();
            SherpaOnnx.SherpaOnnxOfflinePunctuationModelConfig soopmc =
                new SherpaOnnx.SherpaOnnxOfflinePunctuationModelConfig
                (Environment.CurrentDirectory + "/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx", 1, false, "cpu");
            soopc.model = soopmc;
            offlinePunctuation = new SherpaOnnx.OfflinePunctuation(soopc);

            VadModelConfig vadModelConfig = new VadModelConfig();
            vadModelConfig.SileroVad.Model = Environment.CurrentDirectory + "/silero_vad.onnx";
            vadModelConfig.Debug = 0;
            vad = new VoiceActivityDetector(vadModelConfig, 60);

            StartWebServer();
            Update();
            Console.ReadLine();
        }

        static void StartWebServer()
        {
            //存储连接对象的池
            var connectSocketPool = new List<IWebSocketConnection>();
            //创建WebSocket服务端实例并监听本机的9999端口
            var server = new WebSocketServer("wss://172.32.151.240:9999");
            server.Certificate =
                new System.Security.Cryptography.X509Certificates.X509Certificate2(
                    Environment.CurrentDirectory + "/usherpa.xuefei.net.cn.pfx", "xb5ceehg");
            server.EnabledSslProtocols = System.Security.Authentication.SslProtocols.Tls12;
            //开启监听
            server.Start(socket =>
            {
                //注册客户端连接建立事件
                socket.OnOpen = () =>
                {
                    client = socket;
                    Console.WriteLine("Open");
                    //将当前客户端连接对象放入连接池中
                    connectSocketPool.Add(socket);
                };
                //注册客户端连接关闭事件
                socket.OnClose = () =>
                {
                    client = null;
                    Console.WriteLine("Close");
                    //将当前客户端连接对象从连接池中移除
                    connectSocketPool.Remove(socket);
                };
                //注册客户端发送信息事件
                socket.OnBinary = message =>
                {
                    float[] floatArray = new float[message.Length / 4];
                    Buffer.BlockCopy(message, 0, floatArray, 0, message.Length);

                    vad.AcceptWaveform(floatArray);
                    if (vad.IsSpeechDetected())
                    {
                        //Console.Write(" 有人讲话 ");

                        if (!vad.IsEmpty())
                        {
                            SpeechSegment segment = vad.Front();
                            float startTime = segment.Start / (float)sampleRate;
                            float duration = segment.Samples.Length / (float)sampleRate;

                            //Console.Write(" " + startTime + "");
                            //Console.Write(" " + duration + "");
                            // 将采集到的音频数据传递给识别器
                            onlineStream.AcceptWaveform(sampleRate, floatArray);
                        }
                    }
                    else
                    {
                        //Console.Write(" 无人语我 ");
                    }
                };
            });
        }

        static string lastText = "";

        static void Update()
        {
            while (true)
            {
                // 每帧更新识别器状态
                if (recognizer.IsReady(onlineStream))
                {
                    recognizer.Decode(onlineStream);
                }

                var text = recognizer.GetResult(onlineStream).Text;
                bool isEndpoint = recognizer.IsEndpoint(onlineStream);
                if (!string.IsNullOrWhiteSpace(text) && lastText != text)
                {
                    if (string.IsNullOrWhiteSpace(lastText))
                    {
                        lastText = text;
                        if (client != null)
                        {
                            TextMsg textMsg = new TextMsg();
                            textMsg.isEndpoint = false;
                            textMsg.message = text.ToLower();
                            client.Send(Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(textMsg)));
                            //Console.WriteLine("text1:" + text);
                        }
                    }
                    else
                    {
                        if (client != null)
                        {
                            //client.Send(Encoding.UTF8.GetBytes(text.Replace(lastText, "")));
                            TextMsg textMsg = new TextMsg();
                            textMsg.isEndpoint = false;
                            textMsg.message = text.Replace(lastText, "").ToLower();
                            client.Send(Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(textMsg)));
                            lastText = text;
                        }
                    }
                }

                if (isEndpoint)
                {
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        if (client != null)
                        {
                            //client.Send(Encoding.UTF8.GetBytes("。"));
                            TextMsg textMsg = new TextMsg();
                            textMsg.isEndpoint = true;
                            textMsg.message = offlinePunctuation.AddPunctuation(text.ToLower());
                            client.Send(Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(textMsg)));
                        }
                        //Console.WriteLine(offlinePunctuation.AddPunctuation(text));
                    }
                    recognizer.Reset(onlineStream);
                    //Console.WriteLine("Reset");
                }
                Thread.Sleep(200); // ms
            }
        }
    }
}