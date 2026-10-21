# MOSS-TTS-Nano CoreML

`MossTTSNano` implements MOSS-TTS-Nano-100M and MOSS-Audio-Tokenizer-Nano with
four compiled CoreML ML Programs. The speech checkpoint has 117,311,232 parameters;
the folded codec has 21,042,560. The upstream model and codec use Apache-2.0.

Bundle: [aufklarer/MOSS-TTS-Nano-100M-CoreML](https://huggingface.co/aufklarer/MOSS-TTS-Nano-100M-CoreML).
Requires Swift 6 and macOS 15+ or iOS 18+ for CoreML state. Local validation uses an M5 Pro Mac;
iPhone quality, memory and speed have not been measured.

| Graph | Purpose | Precision |
|---|---|---|
| `GlobalDecoder.mlmodelc` | 12-layer, 768-wide text/audio transformer; 1,024-row KV state | FP16 matrix operations; FP32 normalization, rotary phases and residuals |
| `LocalDecoder.mlmodelc` | Predict continue/end, then 16 codec codebooks | FP32 math, FP16 state |
| `CodecEncoder.mlmodelc` | Encode a reference voice into 16 codes per frame | FP32 math, FP16 state |
| `CodecDecoder.mlmodelc` | Stream codes into 48 kHz stereo PCM | FP16 matrix operations; FP32 normalization, rotary phases and residuals |

The compiled graphs total 367.6 MB. Each codec frame carries 16 codebook IDs in
0...1023 and produces 3,840 samples per channel (80 ms, 12.5 frames/second).
The host runs SentencePiece BPE and sampling. `tokenizer_swift.json` preserves the
original vocabulary, scores, user-defined tokens, byte fallback and compiled
normalization trie. Vocabulary identity uses UTF-8 bytes. There is no MLX or
Python inference dependency in this product.

The global decoder accepts rows of 17 Int32 values: text/control ID, followed by
16 audio IDs. Text rows pad audio IDs with 1024. Reference rows use control ID 8;
generated rows use 9. The local decoder resets at step 0, predicts continue/end
from two logits, then predicts each codebook in steps 1...16. Global prefill uses
up to 256 contiguous rows per call. Codec attention retains bounded sliding
windows. Each synthesis/reference creates fresh MLState objects; requests on
one actor are serialized. Encoded reference codes can be reused across requests.

Matrix-only FP16 preserves rotary arithmetic at large codec positions. The local
decoder and encoder use FP32 to preserve logits and residual code selection.
Compiled graph tests compare both CPU and CPU+GPU against unmodified upstream
PyTorch, including recurrent updates and a 10.4-second codec stream crossing all
attention windows. See the bundle's `config.json` and `speech_validation.json`.

Language follows the input text. Reference audio is optional and does not need a
transcript. Preset speakers, explicit language selection and instruction controls
are not exposed. Sampling uses the source defaults (text temperature 1, audio
temperature 0.8, top-k 25, top-p 0.95, repetition penalty 1.2). Swift uses a stable
PCG64 host generator with NumPy-compatible SeedSequence initialization for UInt64
seeds. Unit tests match NumPy random draws and sampling sequences; matched-host
E2E checks reproduce Python codec tokens exactly on both CPU and GPU.
Greedy generation can repeat without EOS in the source model as well.

See [inference](../inference/moss-tts-nano.md) and
[measurements](../benchmarks/moss-tts-nano.md).

Sources: [model](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M),
[codec](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano),
[implementation](https://github.com/OpenMOSS/MOSS-TTS-Nano).
