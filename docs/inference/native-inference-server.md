# Native inference service

Source builds of `speech-server` can load Gemma 4 E4B IT INT4, VoxCPM2 with FP16
network parameters, full Whisper Large-v3 CoreML, and CoreML Silero VAD into one
process. All model assets must already exist locally. Startup rejects incompatible
checkpoints and performs no downloads or model substitution.

Failed non-streaming inference requests log the request ID, operation, and
underlying error in the server process. HTTP failures remain errors; inspect
the local server log when a framework-generated HTTP 500 has an empty body.

An optional `agentDirectory` loads a staged Gemma 4 12B Unified INT4 text model
alongside E4B. The 12B checkpoint must use affine quantization with group size 64.
Select `model: "gemma4-12b-it"` for chat requests; E4B remains available as
`gemma4-e4b-it`. Both models use the same serialized inference queue. The 12B
model supports native tools, result history, and cancellation; structured-output
capabilities remain associated with E4B.

```json
{
  "runtimeRevision": "your-source-revision",
  "gemmaDirectory": "/models/gemma4-e4b-it-int4",
  "agentDirectory": "/models/gemma-4-12B-it-4bit",
  "voxDirectory": "/models/voxcpm2-unquantized",
  "whisperDirectory": "/models/whisper-large-v3-coreml",
  "vadDirectory": "/models/silero-vad-coreml"
}
```

```bash
make build
.build/release/speech-server --port 8080 --inference-manifest models.json
```

## Capturing failed chat requests

Set the optional `failureCaptureDirectory` field in the manifest to a local
directory to retain failed chat requests. Capture is disabled when this field
is absent. New directories use permissions `0700`; an existing directory must
already deny group and other access. Each failure receives a new UUID filename
with permissions `0600`, independent of its request ID.

The record includes the original HTTP body (UTF-8 text when applicable and
lossless base64), request ID, model, runtime revision, and the original error.
For completed Gemma 4 12B generations it also includes the exact rendered
prompt, input/generated token IDs, raw output before tool parsing, stop token
ID, and `stop` or `length` finish reason. A failure before completion has no
generation record. E4B failures retain request metadata without a raw generation.

These files contain conversation and tool content. They stay on the local
filesystem; neither ordinary HTTP errors nor successful requests include them.
Recording a failure does not retry it, change parsing, or convert it to success.

The manifest can also carry an `assets` JSON value containing source revisions
and file hashes. `GET /v1/inference/provenance` returns it with the configured
runtime revision, process ID, actual VoxCPM2 parameter dtype counts, and MLX
memory counters. `GET /v1/inference/capabilities` lists exact model IDs, engines,
precisions, and supported features. These inventories describe this service,
separately from the general model catalog at `/v1/models`.

## Requests

Send JSON to `POST /v1/inference` with a unique active `requestID`, `operation`,
and exact `model`. Unknown operation/model combinations fail. Audio input and
output use base64 mono PCM16 little-endian, without a WAV header.

| Operation | Model | Additional fields | Result |
| --- | --- | --- | --- |
| `chat` | `gemma4-e4b-it` | `messages`, optional `tools`, `maxTokens`, `temperature` | `content`, parsed `toolCalls`, `generatedTokens` |
| `chat` | `gemma4-12b-it` (when loaded) | Same chat fields | Same text/tool response |
| `speech` | `voxcpm2` | `text`, optional `voiceInstruction`, `seed`, `maxTokens`, `stream` | `audioBase64`, `sampleRate: 48000`, or an NDJSON stream |
| `transcribe` | `whisper-large-v3` | `audioBase64`, `sampleRate` | English `text` |
| `vad` | `silero-vad` | `audioBase64`, `sampleRate: 16000`, `streamID`, optional `reset`, `endStream` | One probability per 512-sample frame |

Audio requests are limited to 60 seconds, 8–48 kHz. VAD requires complete
512-sample frames at 16 kHz. Its LSTM and context state are isolated by
`streamID`. Set `endStream: true` on the final frame to release that state.
At most 64 streams may remain open.

```bash
curl http://127.0.0.1:8080/v1/inference \
  -H 'Content-Type: application/json' \
  -d '{"operation":"chat","model":"gemma4-e4b-it","requestID":"chat-1","messages":[{"role":"user","content":"Hello"}],"maxTokens":128,"temperature":0}'
```

Chat messages support `system`, `user`, `assistant`, and `tool` roles. An
assistant's `toolCalls` contain `id`, `name`, and an `arguments` JSON object.
Each following tool result references that ID through `toolCallID` and carries
its serialized result in `content`. Supply all results before requesting another
reply. Tool declarations contain `name`, `description`, and `parameters` as a
JSON schema. The runtime uses Gemma's native declaration/call/result tokens,
expands local schema references, and validates generated argument syntax.
Clients own validation of application-specific JSON replies.

For 12B, duplicate adjacent closing markers immediately after a complete,
validated tool call are consumed with that call. They do not create another
call or alter its arguments. Other orphan markers and malformed arguments remain
errors. E4B and the public codec retain strict terminator handling.

## Streaming and cancellation

Speech with `stream: true` returns `application/x-ndjson`. Each `audio` event
contains `audioBase64` and `sampleRate: 48000`. A successful stream ends with
`{"event":"done"}`. An `error` event or a stream without `done` indicates failure.
VoxCPM2 decodes bounded causal windows every four patches, retains the decoder's
left context and one patch at the moving edge, and emits the remaining tail at
completion. Window decoding uses
an independent random state so it does not change diffusion sampling. The
default request seed is 42.

Cancel an active or queued request with:

```bash
curl http://127.0.0.1:8080/v1/inference/cancel \
  -H 'Content-Type: application/json' -d '{"requestID":"chat-1"}'
```

Gemma checks cancellation between prefill batches and generated tokens;
VoxCPM2 checks between audio patches. Submitted GPU work can finish. Disconnecting
a speech stream cancels its producer. Transcription checks cancellation around
the CoreML operation.

## Precision, scheduling, and memory

VoxCPM2 FP16 mode converts an unquantized checkpoint's network parameters to
Float16. AudioVAE parameters and decoding remain Float32; diffusion inputs can
promote arithmetic to Float32. This is mixed computation with FP16 main weights,
not a claim that every operation runs in FP16. INT8 checkpoints are rejected
for this mode. Existing automatic loading keeps its previous precision policy.
The service loads the tokenizer from `voxDirectory` and leaves its files
unchanged. It needs no separate tokenizer cache.

The service serializes chat, synthesis, and transcription across requests.
Lightweight CoreML VAD runs separately. Gemma prefill evaluates each 256-token
batch before starting the next,
request-local KV state is released after generation, MLX's buffer cache is
capped at 128 MiB, and its allocator limit is 20 GiB. Unused cached buffers are
cleared between heavy requests. Whisper releases per-step CoreML temporaries
while decoding. These limits apply to this explicitly enabled native service;
CoreML and other process allocations also contribute to total memory use.

Full Large-v3 must have 32 decoder layers. A bundle without the optional
`TextDecoderContextPrefill.mlmodelc` builds the language/task prefix through
the decoder using the current audio embeddings. The default Whisper CLI model
continues to use its published Turbo bundle.
