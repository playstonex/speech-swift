The network fixture uses 42,300 bytes of synthetic Float32 CPU weights. It runs
the original PatchEncoder, DiT, and RedAE decoder class bodies from
FireRedTeam/FireRedTTS3 commit `7a1f3a7282ff184cc1c7f070556baaf5f08b5216`,
with Transformers Qwen3 models. Its metadata records framework versions and
source hashes. No pretrained weights or recorded voices are included.

The ISTFT fixture is an independent NumPy frame-loop overlap-add reference,
including one-frame boundaries and nonzero imaginary DC/Nyquist bins. The
stored periodic Hann window is used exactly.

The tokenizer fixture verifies six exact input strings against the official
loader and against a persisted copy of its 255 added speech/language tokens.
The resulting vocabulary has 151,924 entries. It includes the complete English
and Chinese voice-design prompts and whitespace-sensitive text.

FireRedTTS3 source is Apache-2.0; its license is included alongside these
fixtures. Frameworks are used under their respective upstream licenses.
