# MLX reusable-buffer cache

MLX keeps freed Metal buffers for reuse. Repeated requests with different input
lengths can retain many buffer sizes even though the model's live arrays are
stable. MLX model runtimes in this package now register a default cache ceiling
of `min(2 GiB, physical RAM / 4)` before loading weights and retain it for the
model's lifetime. This applies across TTS, ASR, chat, translation, VAD, speaker
models, enhancement, separation, music generation and avatar motion.

The ceiling limits spare buffers. It does not limit active model weights,
attention/KV caches, intermediate arrays that are still in use, or total process
memory. A large model or input can still require more memory than the ceiling.
MLX trims cached buffers on allocation: lowering the budget does not immediately
clear an existing cache, and the last freed buffer can temporarily exceed the
budget until the next allocation reclaims it.

## Caller budgets and model lifetimes

The shared `MLXCacheLimitCoordinator` applies the minimum of the caller's budget
and every live registration. Set `MLX.Memory.cacheLimit` before loading models
to choose a smaller budget; zero disables buffer caching. Loading another model
never raises that budget. CoreML-only model paths do not acquire MLX leases.

Existing model-specific policies remain: Qwen3-ASR uses up to 1 GiB for 0.6B
and 4 GiB for 1.7B, scaled to RAM. When other runtimes are also loaded, the
tightest live ceiling wins. Chatterbox's balanced generation policy temporarily
tightens the shared cap to 512 MiB and clears buffers at stage boundaries.

For models exposing `unload()`, unloading releases the registration. Other
runtimes release it when the model is destroyed. Failed loads release their
temporary loading registration. Unload order does not matter: the last owner
restores the caller's budget; earlier unloads keep the remaining models capped.
A caller's changed limit is recognized at the next registration or release, so
a tighter limit set during use survives unload. Directly raising the global
limit bypasses the cap until that next coordinator operation.

`unrestricted` generation options for Chatterbox and Fish Audio disable their
stage-level controls; the runtime's lifetime cap still applies. The Kyutai CLI's
256 MiB budget and the server's 128 MiB budget remain lower than the default.

## Validation

Pure coordinator tests cover RAM sizing, disabled caching, mixed budgets,
overlapping owners, failure cleanup, caller changes and destruction. Real-model
tests compare seeded OmniVoice output across cache budgets and varying lengths,
check overlapping TTS/VAD ownership, exercise zero-cache VAD inference, and verify
CoreML-only and failed-load paths. Qwen3-ASR, MOSS and VoxCPM2 retain their
existing coordinator regression tests.
