# Model-free music analysis

The unreleased MusicTools library estimates a periodic onset pulse and a
major/minor key from a selected audio window. It requires no model download.
These are estimates for inspection, not beat-grid annotations, chord detection
or note/MIDI transcription.

MusicTools uses native Apple frameworks without a model runtime or downloader
dependency. Its file APIs are synchronous. macOS verification does not establish
iOS device performance or musical accuracy.

## Command

This command requires a source build containing the change. It is not available
in the 0.0.29 release.

~~~bash
speech analyze-music song.wav
speech analyze-music song.m4a --start 30 --max-seconds 60 --json
~~~

The default analyzes at most the first 120 seconds. Use **--start** to inspect a
different passage; **--max-seconds** must be greater than zero and no more than
120. Mono and stereo files supported by the system decoder are accepted.
The source is read in bounded blocks and only the downsampled selected window
is retained. Stereo channel powers are combined after the FFT so opposite-phase
channels do not cancel during analysis.

**--json** emits one deterministic JSON document on stdout with schema version 1.
Unavailable BPM and key estimates are explicit nulls. The output reports:

- Source filename, sample rate, channel count and complete duration.
- Actual analysis start/duration, analysis sample rate and frame count, and
  whether the selected window is partial.
- RMS and peak signal statistics.
- Ranked tempo and key candidates, fit scores, method names and uncertainty
  notes.

There are no progress messages mixed into JSON stdout. Errors use the ordinary
command failure path.

## Methods and limits

Tempo uses positive spectral flux, local mean subtraction, separated onset
peaks and normalized envelope autocorrelation. Stable onset spacing refines the
estimate without rounding every beat to an integer FFT hop. Half/double tempo
alternatives remain visible; listening is needed to choose the musical beat.
Silence, sustained tones and weakly periodic input can return no tempo.

Key uses spectral peaks accumulated into twelve pitch classes and correlation
with the Krumhansl–Kessler major/minor profiles. Broadband/noisy frames are
excluded. A single pitch or a flat pitch-class distribution cannot establish a
major/minor key. Close templates, including possible relative keys, are returned
as candidates without selecting one key.

Fit scores measure signal/template agreement. They are not calibrated
probabilities or accuracy percentages. Rubato, syncopation, modulation,
percussion, detuning and arrangements outside twelve-tone major/minor music can
produce ambiguous or inaccurate results. No real-song accuracy benchmark is
claimed.

## Swift API

~~~swift
import MusicTools

let result = try MusicAnalyzer.analyze(
    url: audioURL,
    options: MusicAnalysisOptions(startSeconds: 30, maxDurationSeconds: 60)
)
let json = try result.jsonData()
~~~

In-memory mono and stereo overloads accept Float PCM and a sample rate. The same
window limit and validation apply.

## Verification

**MusicAnalysisTests** covers synthetic click tracks at several tempos,
half-tempo ambiguity, major/minor progressions, silence, noise, a sustained
single pitch, opposite-phase stereo, sample-rate conversion, file bounds and
explicit-null JSON. **AnalyzeMusicCommandTests** covers argument parsing and
invalid windows.

The opt-in **E2EMusicAnalysisTests** reads
**SPEECH_MUSIC_ANALYSIS_TEST_AUDIO** and validates an actual file through the
bounded decoder and analyzer. It checks output integrity, not musical accuracy.
