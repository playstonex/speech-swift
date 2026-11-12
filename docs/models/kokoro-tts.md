# Kokoro TTS Architecture

Kokoro-82M is a non-autoregressive text-to-speech model based on [StyleTTS 2](https://github.com/yl4579/StyleTTS2) with an ISTFTNet vocoder. It runs on the Neural Engine via an end-to-end CoreML model, producing 24 kHz speech.

## Overview

- **Parameters**: 82M
- **Backend**: CoreML (Neural Engine)
- **Output**: 24 kHz mono Float32 PCM
- **Inference**: Non-autoregressive, end-to-end model
- **Voices**: 54 presets across 10 languages, plus two German voice bundles (see [German voices](#german-voices))
- **License**: Apache-2.0

## Pipeline

```
Text → Phonemizer → Duration Model → Alignment → Prosody Model → Decoder → 24kHz Audio
         ↓              ↓                              ↓             ↓
   [Dictionary]    [Style Embedding]             [Style Embed]  [Style Embed]
   [G2P BART]         [Speed]
```

1. **Phonemizer** converts text to phoneme token IDs — English uses dictionary lookup + suffix stemming + neural G2P; Chinese uses CFStringTransform pinyin-to-IPA; Japanese uses CFStringTokenizer + katakana-to-IPA; Korean/Hindi use Apple transliteration-to-IPA; French/Spanish/Portuguese use rule-based grapheme-to-IPA
2. **Duration model** predicts per-phoneme durations + intermediate features
3. **Alignment** (Swift-side) builds a frame-to-phoneme matrix from predicted durations
4. **Prosody model** predicts F0 (pitch) and noise features from aligned prosody features
5. **Decoder** synthesizes the audio waveform from aligned text features + prosody

## Model I/O

### Stage 1: Duration Model

**Inputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `input_ids` | [1, N] | Int32 | Phoneme token IDs (N = phoneme bucket: 16, 32, 64, or 128) |
| `attention_mask` | [1, N] | Int32 | 1 for real tokens, 0 for padding |
| `ref_s` | [1, 256] | Float32 | Voice style embedding |
| `speed` | [1] | Float32 | Speed multiplier (1.0 = normal) |

**Outputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `pred_dur` | [1, N] | Float16 | Predicted phoneme durations (frames per phoneme) |
| `d_transposed` | [1, 640, N] | Float16 | Prosody features for alignment |
| `t_en` | [1, 512, N] | Float16 | Text encoding for alignment |

### Stage 2: Prosody Model

**Inputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `en` | [1, 640, F] | Float32 | Aligned prosody features (F = decoder bucket frames) |
| `s` | [1, 128] | Float32 | Style embedding (second half of ref_s) |

**Outputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `F0_pred` | [1, F*2] | Float16 | Predicted pitch contour |
| `N_pred` | [1, F*2] | Float16 | Predicted noise features |

### Stage 3: Decoder

**Inputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `asr` | [1, 512, F] | Float32 | Aligned text features |
| `F0_pred` | [1, F*2] | Float32 | Pitch contour |
| `N_pred` | [1, F*2] | Float32 | Noise features |
| `ref_s` | [1, 128] | Float32 | Style embedding (first half) |

**Outputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `audio` | [1, 1, F*600] | Float16 | 24 kHz waveform |

### Swift-Side Alignment

Between stages 1 and 2, Swift builds an alignment matrix from predicted durations:

```
pred_aln_trg[phoneme_idx, frame_idx] = 1.0  // for each phoneme's allocated frames
en  = d_transposed @ pred_aln_trg   // [640, F] aligned prosody features
asr = t_en @ pred_aln_trg           // [512, F] aligned text features
```

## Model Buckets

### Phoneme Buckets (Duration Model)

The duration model uses enumerated input shapes to minimize backward LSTM contamination from padding. Input is padded to the smallest bucket that fits:

| Bucket | Max Phonemes | Use Case |
|--------|-------------|----------|
| p16 | 16 | Short phrases (1-2 words) |
| p32 | 32 | Short sentences |
| p64 | 64 | Medium sentences |
| p128 | 128 | Long sentences |

### Decoder Buckets

Fixed-shape decoder models for different maximum output lengths:

| Bucket | Max Frames | Max Audio | Samples |
|--------|-----------|-----------|---------|
| `decoder_5s` | 200 | 5.0s | 120,000 |
| `decoder_10s` | 400 | 10.0s | 240,000 |
| `decoder_15s` | 600 | 15.0s | 360,000 |

Each frame produces 600 audio samples at 24 kHz (25ms per frame).

## Long text

The end-to-end graph takes 128 phoneme tokens but renders at most 5 s per pass, about 56 tokens at the models' speech rate (upstream Kokoro's, ~10.5 tokens/s). Text is split the way speech-core splits it on Android, and text within one chunk renders in a single call:

- Chunks are packed greedily up to 56 tokens, preferring sentence ends, then clause punctuation (`,` `;` `:`), then words, then characters for a single oversized word. Punctuation is a boundary only when whitespace or the end of the text follows it, so `3.5`, `1.000`, `z.B.` and `18:30` reach the phonemizer intact.
- A last chunk under 12 tokens joins the previous one when the merged chunk stays within 64 tokens.
- A chunk whose speech would reach the last 200 ms of the output window is split at its best sentence, clause or word boundary and rendered again, up to four times.
- `synthesize` returns the chunks' audio concatenated. `generateStream` yields each chunk as soon as it is rendered, with `frameIndex` set to the chunk's first sample in the stream.

A sentence longer than one chunk, about ten words without a comma, is split between words and gets a short pause there.

## Phonemizer

Three-tier pipeline, all Apache-2.0 licensed (no GPL dependencies):

1. **Dictionary lookup** — US and British English pronunciation dictionaries with heteronym support (POS-based disambiguation)
2. **Suffix stemming** — Morphological decomposition for known suffixes
3. **BART G2P** — Neural grapheme-to-phoneme fallback using a separate CoreML encoder-decoder model for OOV words

### Custom pronunciations

Proper nouns outside the shipped dictionaries reach the BART G2P fallback, which guesses from *English* spelling rules and mangles non-Anglo names. `addPronunciations` injects `word -> IPA` entries ahead of tier 1:

```swift
let tts = try await KokoroTTSModel.fromPretrained()
tts.addPronunciations(["tatiana": "tɑtiˈɑnə"])
```

Keys are matched lowercased and must be single words — text is split on whitespace and punctuation before resolution, so a key containing a space, hyphen, or apostrophe is never looked up. IPA must use symbols from the model vocabulary; anything else is silently dropped at tokenization. Call `addPronunciations` after the dictionaries are loaded — `loadDictionaries` replaces the gold dictionary wholesale.

Entries apply to the English path only: the `language` argument selects a dedicated phonemizer for Chinese, Japanese, Italian, French, Spanish, Portuguese, and Hindi, and none of those consult the gold dictionary. The lookup is unlocked, so register entries before synthesis starts running concurrently on the same model.

## Voice Embeddings

Each voice is a 256-dimensional Float32 vector stored in a per-voice JSON file. The embedding captures speaker identity and style characteristics. The first 128 dimensions are used by the decoder, the second 128 by the prosody model.

Naming convention: `[language][gender]_[name]`
- `a` = American English, `b` = British English
- `e` = Spanish, `f` = French, `h` = Hindi, `i` = Italian
- `j` = Japanese, `k` = Korean, `p` = Portuguese, `z` = Chinese
- `f` = female, `m` = male

## Weight Files

| Component | Size | Format |
|-----------|------|--------|
| Duration model | ~39 MB | .mlmodelc |
| Prosody model | ~17 MB | .mlmodelc |
| Decoder models (3 buckets) | ~107 MB each | .mlmodelc |
| Voice embeddings (54 voices) | ~0.3 MB | JSON |
| G2P encoder + decoder | ~1.5 MB | .mlmodelc |
| Dictionaries + vocab | ~6 MB | JSON |
| **Total (1 decoder)** | **~170 MB** | |

Weights: [aufklarer/Kokoro-82M-CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML)

## German voices

German is not one of the multilingual bundle's languages: the German voices are
separate Kokoro-architecture checkpoints fine-tuned by
[kikiri-tts](https://huggingface.co/kikiri-tts) (StyleTTS2 stage 2, Apache 2.0),
one voice per weight set, converted with the same end-to-end pipeline:

| Model ID | Voice | Files |
|---|---|---|
| `aufklarer/Kokoro-82M-German-Martin-CoreML` | `dm_martin` (male) | `kokoro_5s.mlmodelc`, `vocab_index.json`, `dict_de.json`, `voices/dm_martin.json` |
| `aufklarer/Kokoro-82M-German-Victoria-CoreML` | `df_victoria` (female) | same layout, `voices/df_victoria.json` |

```swift
let tts = try await KokoroTTSModel.fromPretrained(modelId: KokoroTTSModel.germanMartinModelId)
let audio = try tts.synthesize(text: "Am 3. Oktober um 18:30 Uhr kostet es 12,80 €.",
                               voice: "dm_martin", language: "de")
```

`GermanPhonemizer.swift` reproduces the front-end the voices were trained
with (espeak-ng 1.52 German as post-processed by misaki, short ü written `y`,
`ʦ`/`ʧ`/`ʤ` affricates, `I`/`W` diphthongs) without espeak: it spells out
numbers, ordinals, dates, times, currency, units and abbreviations, resolves
words through `dict_de.json` (the 50k most frequent German words plus the
brand/tech overrides of the kikiri front-end, downloaded with the bundle),
derives inflected and compound forms from dictionary words, and falls back to
rule-based G2P. The structure is shared with speech-core's `kokoro_german.cpp`;
`GermanPhonemizerTests` keeps the two in lockstep and can run a full parity
check against the C++ output (`KOKORO_GERMAN_PARITY_TSV`). Passing `"de"` to
the multilingual bundle still works (German phonemes through an English voice);
a German bundle has exactly one voice.

## Performance (M2 Max)

| Metric | Value |
|--------|-------|
| Inference RTFx | ~0.7 (faster than real-time) |
| Weight memory | ~170 MB (1 decoder bucket) |
| Backend | CoreML (Neural Engine) |

Non-autoregressive — no sampling loop, latency scales linearly with output length but remains faster than real-time.

## Compute Unit Override

`fromPretrained(computeUnits:)` selects which hardware the main CoreML model runs on:

```swift
// Default: Neural Engine preferred (recommended)
let tts = try await KokoroTTSModel.fromPretrained()

// Fallback: bypass ANE (use on platforms where the ANE compiler
// produces incorrect output for this model — e.g., some iOS 26 builds)
let tts = try await KokoroTTSModel.fromPretrained(computeUnits: .cpuAndGPU)
```

The G2P encoder/decoder always run on CPU regardless of this setting; they are small enough that CPU is the fastest path.

## Warm-up

A model's first prediction is several times slower than the ones after it, so `fromPretrained` runs one short synthesis before it returns. On an M5 Pro this adds about 0.55 s to loading, and the first sentence after loading takes about 0.13 s instead of 0.5–0.8 s. Pass `warmUp: false` to skip it, for example when the model is only being downloaded, and call `warmUp()` before the first reply:

```swift
let tts = try await KokoroTTSModel.fromPretrained(warmUp: false)
try tts.warmUp()
```

## Conversion

```bash
python scripts/convert_kokoro_coreml.py --output /tmp/kokoro-coreml
python scripts/convert_kokoro_coreml.py --output /tmp/kokoro-coreml --quantize int8
```

## Source Files

```
Sources/KokoroTTS/
  Configuration.swift        Model config, voice/language selection
  KokoroModel.swift          End-to-end CoreML model loading and inference
  KokoroTTS.swift            High-level API (fromPretrained, synthesize, alignment)
  TextChunker.swift          Sentence/clause/word chunking of long text (mirrors speech-core)
  Phonemizer.swift           English G2P + multilingual routing (en/zh/ja/hi/fr/es/pt/it/de)
  GermanPhonemizer.swift     German: normalization + dict_de.json + inflection/compounds + rule-based G2P
  ChinesePhonemizer.swift    Chinese: CFStringTransform pinyin → IPA
  JapanesePhonemizer.swift   Japanese: CFStringTokenizer → katakana → IPA (M2P table)
  HindiPhonemizer.swift      Hindi: Apple IAST transliteration → IPA
  LatinPhonemizer.swift      French/Spanish/Portuguese/Italian: rule-based grapheme → IPA
  PronunciationDicts.swift   Pronunciation dictionaries (JSON resource loading + inline)
  KokoroTTS+Protocols.swift  Protocol conformance
  KokoroTTS+Memory.swift     Memory reporting
```
