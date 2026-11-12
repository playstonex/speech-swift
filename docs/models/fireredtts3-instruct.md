# FireRedTTS3 Instruct

`FireRedTTS3TTS` implements reference-free voice design natively in MLX Swift.
It uses the Instruct checkpoint's text planning stage: the model first writes
voice attributes from the instruction, then synthesizes the exact supplied
English or Chinese script. `generateWithPlan` returns those attributes and
separate planning, acoustic generation, and decoding times.

The runtime contains a 28-layer Qwen3 backbone, an 8-layer patch encoder,
an 11-layer diffusion transformer, and the decoder side of RedAE. Each acoustic
patch generates four 25 Hz latents. RedAE upsamples them twice and converts
spectra to 3,840 mono samples at 24 kHz, or 160 ms per acoustic patch.

The initial export retains the original Float32 tensors and key layout. It
requires approximately 9.44 GB of weight storage, plus tokenizer files. The
unused RedAE encoder and decoder token embedding are omitted. Voice cloning,
audio references, and speech editing are not exposed by this implementation.
The runtime does not invoke Python or normalize, split, or rewrite the script.

The backbone and RedAE use split-half rotary embeddings, with bases 1,000,000
and 10,000 respectively. RedAE has 128-wide attention heads even though its
hidden width is 896 and it has 14 query heads. Every RedAE layer attends to a
64-position causal window. The patch encoder and flow model instead use
adjacent-pair rotary embeddings and full attention. RedAE uses the stored
periodic Hann window, same-padding inverse FFT, and window-envelope division.

The upstream CUDA wrapper uses BF16 autocast for its backbone. This initial
runtime computes that backbone in Float32; source-preserving weights do not
imply bit-identical sampled speech across frameworks. Quantized exports are
not supported by the initial format.

The official source and checkpoint are Apache-2.0. Pinned sources:
[implementation](https://github.com/FireRedTeam/FireRedTTS3/tree/7a1f3a7282ff184cc1c7f070556baaf5f08b5216),
[checkpoint](https://huggingface.co/FireRedTeam/FireRedTTS3/tree/dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610).
See [inference](../inference/fireredtts3-instruct.md) and
[validation](../benchmarks/fireredtts3-instruct.md).
