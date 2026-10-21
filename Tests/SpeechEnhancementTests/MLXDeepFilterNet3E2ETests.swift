import XCTest
@testable import SpeechEnhancement

/// E2E tests for the MLX DeepFilterNet3 engine.
///
/// Downloads ``aufklarer/DeepFilterNet3-MLX`` (fp32 safetensors) on first
/// run. CI filters this class out via the `--skip E2E` regex, matching the
/// CoreML E2E tests.
final class MLXDeepFilterNet3E2ETests: XCTestCase {

    /// Deterministic pseudo-noise (LCG) so runs are reproducible.
    private func whiteNoise(count: Int, amplitude: Float, seed: UInt64 = 0x5EED) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float(state >> 40) / Float(1 << 24)   // [0, 1)
            return (unit * 2 - 1) * amplitude
        }
    }

    /// Speech-band test signal: amplitude-modulated harmonic stack at 48 kHz.
    private func syntheticSpeech(seconds: Double, sampleRate: Int) -> [Float] {
        let count = Int(seconds * Double(sampleRate))
        let fundamental: Float = 180
        return (0..<count).map { i in
            let t = Float(i) / Float(sampleRate)
            let envelope = 0.5 + 0.5 * sin(2 * .pi * 3.1 * t)   // syllable-rate AM
            let harmonics = sin(2 * .pi * fundamental * t)
                + 0.5 * sin(2 * .pi * fundamental * 2 * t)
                + 0.25 * sin(2 * .pi * fundamental * 3 * t)
            return 0.2 * envelope * harmonics
        }
    }

    /// Full pipeline through the MLX engine: download, safetensors load,
    /// DSP, network, iSTFT. Mirrors the CoreML `testHubFromPretrained`.
    func testHubFromPretrainedMLX() async throws {
        let enhancer = try await SpeechEnhancer.fromPretrained(engine: .mlx)
        XCTAssertEqual(enhancer.engine, .mlx)

        let sampleRate = 48000
        let samples = [Float](repeating: 0, count: sampleRate + 1)
        let enhanced = try enhancer.enhance(audio: samples, sampleRate: sampleRate)

        XCTAssertEqual(enhanced.count, samples.count,
                       "Enhanced signal should match input length")
        let peak = enhanced.map(abs).max() ?? 0
        XCTAssertLessThan(peak, 0.01, "Enhanced silence should stay near 0")
    }

    /// Engine auto-detection: an "MLX" model ID must select the MLX backend
    /// without an explicit engine parameter (server + CLI path).
    func testEngineAutoDetectionFromModelIdE2E() async throws {
        let enhancer = try await SpeechEnhancer.fromPretrained(
            modelId: SpeechEnhancer.defaultMLXModelId)
        XCTAssertEqual(enhancer.engine, .mlx)
    }

    /// The two engines run different numerics (INT8-palettized fp16 CoreML vs
    /// fp32 MLX) but the same model — outputs on noisy speech must agree
    /// closely and both must attenuate the added noise.
    func testMLXMatchesCoreMLOnNoisySpeech() async throws {
        let sampleRate = 48000
        let clean = syntheticSpeech(seconds: 2.0, sampleRate: sampleRate)
        let noise = whiteNoise(count: clean.count, amplitude: 0.05)
        let noisy = zip(clean, noise).map(+)

        let coreml = try await SpeechEnhancer.fromPretrained(engine: .coreml)
        let mlx = try await SpeechEnhancer.fromPretrained(engine: .mlx)

        let outCoreML = try coreml.enhance(audio: noisy, sampleRate: sampleRate)
        let outMLX = try mlx.enhance(audio: noisy, sampleRate: sampleRate)

        XCTAssertEqual(outCoreML.count, outMLX.count)

        // Pearson correlation between the two engine outputs.
        let n = Float(outMLX.count)
        let meanA = outCoreML.reduce(0, +) / n
        let meanB = outMLX.reduce(0, +) / n
        var num: Float = 0, varA: Float = 0, varB: Float = 0
        for i in 0..<outCoreML.count {
            let a = outCoreML[i] - meanA
            let b = outMLX[i] - meanB
            num += a * b
            varA += a * a
            varB += b * b
        }
        let correlation = num / max(sqrt(varA * varB), .leastNormalMagnitude)
        XCTAssertGreaterThan(correlation, 0.90,
                             "CoreML and MLX outputs should be strongly correlated")

        // Both engines should reduce the noise energy: compare residual vs the
        // clean signal against the injected noise energy.
        func residualEnergy(_ output: [Float]) -> Float {
            var energy: Float = 0
            for i in 0..<output.count {
                let r = output[i] - clean[i]
                energy += r * r
            }
            return energy
        }
        let noiseEnergy = noise.reduce(Float(0)) { $0 + $1 * $1 }
        XCTAssertLessThan(residualEnergy(outCoreML), noiseEnergy * 0.5,
                          "CoreML engine should remove at least half the noise energy")
        XCTAssertLessThan(residualEnergy(outMLX), noiseEnergy * 0.5,
                          "MLX engine should remove at least half the noise energy")
    }
}
