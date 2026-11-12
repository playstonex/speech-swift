import AudioCommon
import Foundation
import MLX
import MossVoiceGeneratorTTS
import Qwen3TTS
import XCTest

/// Opt-in, sequential generation with two models resident in one process.
/// Uses public APIs and explicit local bundles; it never downloads weights.
final class E2EMossVoiceGeneratorCoexistenceTests: XCTestCase {
    func testLoadedQwenPreservesCompleteAudioAndBudgetAcrossMossUse() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let qwenPath = env["QWEN_CUSTOMVOICE_BUNDLE"],
              let tokenizerPath = env["QWEN_TOKENIZER_BUNDLE"],
              let mossPath = env["MOSS_VOICEGEN_BUNDLE"] else {
            throw XCTSkip("Set QWEN_CUSTOMVOICE_BUNDLE, QWEN_TOKENIZER_BUNDLE and MOSS_VOICEGEN_BUNDLE")
        }
        let callerBudget = Memory.cacheLimit
        let qwen = try Qwen3TTSModel.fromLocal(modelDirectory: URL(fileURLWithPath: qwenPath),
            tokenizerDirectory: URL(fileURLWithPath: tokenizerPath))
        defer { qwen.unload() }
        let qwenBudget = Memory.cacheLimit
        let baseline = try await completeWAV(qwen)

        let moss = try await MossVoiceGeneratorModel.fromBundle(URL(fileURLWithPath: mossPath))
        defer { moss.unload() }
        XCTAssertEqual(Memory.cacheLimit, min(qwenBudget, 512 * 1024 * 1024))
        _ = try await moss.generate(text: "The quick brown fox jumps over the lazy dog.",
            instruction: "A clear, calm English narrator.", options: .init(maxFrames: 150, seed: 42))
        XCTAssertTrue(qwen.isLoaded)
        XCTAssertTrue(moss.isLoaded)
        let resident = try await completeWAV(qwen)
        XCTAssertEqual(resident, baseline, "MOSS changed an already loaded Qwen model's complete WAV")

        moss.unload()
        XCTAssertFalse(moss.isLoaded)
        XCTAssertEqual(moss.memoryFootprint, 0)
        XCTAssertEqual(Memory.cacheLimit, qwenBudget, "Qwen's previous cache budget was not restored")
        let after = try await completeWAV(qwen)
        XCTAssertEqual(after, baseline, "Qwen did not recover identical audio after MOSS unload")
        qwen.unload()
        XCTAssertEqual(Memory.cacheLimit, callerBudget, "The original caller cache budget was not restored")
    }

    private func completeWAV(_ model: Qwen3TTSModel) async throws -> Data {
        MLXRandom.seed(42)
        let sampling = SamplingConfig(temperature: 0.9, topK: 50, topP: 1,
            repetitionPenalty: 1.05, maxTokens: 500)
        let stream = model.synthesizeStream(text: "The quick brown fox jumps over the lazy dog.",
            language: "english", speaker: "ryan", instruct: "A clear, calm English narrator.",
            sampling: sampling,
            streaming: StreamingConfig(firstChunkFrames: 3, chunkFrames: 25, decoderLeftContext: 10),
            languageExplicit: true)
        var samples: [Float] = []
        var finals = 0
        for try await chunk in stream {
            samples += chunk.samples
            if chunk.isFinal { finals += 1 }
        }
        XCTAssertEqual(finals, 1)
        XCTAssertFalse(samples.isEmpty)
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: file) }
        try WAVWriter.write(samples: samples, sampleRate: 24000, to: file)
        return try Data(contentsOf: file)
    }
}
