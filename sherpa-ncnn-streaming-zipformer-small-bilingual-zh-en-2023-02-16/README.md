---
license: apache-2.0
---

# Streaming zipformer for sherpa-ncnn

The torchscript model is from
https://huggingface.co/pfluo/k2fsa-zipformer-bilingual-zh-en-t

Different from https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed, this
model is much smaller and thus it is faster to run.

The training code is from
https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming
