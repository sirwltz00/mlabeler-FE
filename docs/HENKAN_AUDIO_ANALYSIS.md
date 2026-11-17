# Henkan acoustic analysis in MLabeler

MLabeler's auto-oto generator now includes an optional Henkan-style acoustic
boundary refinement stage. It is enabled by default in Auto-oto settings and
can be disabled without changing the existing segmentation or oto-entry rules.

## Architecture

- `core/dsp/HenkanAudioAnalysis.kt` implements the short-time acoustic feature
  extraction and transition detector in common Kotlin.
- `core/oto/HenkanOtoRefiner.kt` adapts those measurements to MLabeler's
  existing syllable timings and phoneme interval tiers.
- `AutoOto.segment()`, `AutoOto.fromPhonemes()` and `AutoOto.entries()` remain
  the existing MLabeler algorithms. The refiner only adjusts timing inputs;
  it does not create oto.ini entries itself.
- The auto-oto settings dialog exposes **Henkan acoustic boundary refinement**.

## Analysis features

The shared implementation follows the relevant parts of Henkan FE's
`u2v/audio.py`: a 5 ms analysis hop, a 40 ms Hann window, a 12 ms short-window
edge cue, energy in dB, normalized autocorrelation periodicity, log-frequency
spectral bands, spectral centroid/flatness, high-frequency balance, spectral
flux, voicing confidence and local transition confidence. The detector handles
silence transitions, voiced/unvoiced transitions, stop closures, vowel-to-vowel
spectral changes and voiced-fricative edges.

This is a targeted Kotlin port of the boundary-analysis path, not a Python
runtime or a wholesale port of Henkan's audio conversion, repair, formant,
voice-quality or audio-quality subsystems.

## Conservative timing policy

Each proposed boundary is searched near MLabeler's original timing. Acoustic
confidence and, when present, aligner label confidence gate the adjustment.
The movement is limited by the neighbouring interval lengths and by a maximum
shift. If analysis cannot find a reliable edge, the original boundary remains.
The generated entries are still produced and reviewed using MLabeler's current
`AutoOto.entries()` and preview workflow.

## Cross-platform behavior

All analysis and refinement code is in `core/src/commonMain`. It has no Python,
JVM process, filesystem, native library, server or platform-specific dependency.
It therefore shares the same implementation across the core module's current
JVM, Android, iOS and WebAssembly targets.
