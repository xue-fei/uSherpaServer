using System;

namespace uSherpaServer
{
    /// <summary>
    /// 语音识别的消息
    /// </summary>
    [Serializable]
    public class TextMsg
    {
        public string message;
        public bool isEndpoint;
    }
}