import AudioCommon
import Foundation
import MLX
import Qwen3ASR
import XCTest

@testable import Qwen3TTS

#if os(macOS)
final class E2EICLJapaneseSpeechTests: E2ETestCase {
    func testJapaneseFirstWordAndSentenceEndingSurviveDecoding() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let referenceText = "こんにちは。今日は落ち着いた声で、日本語の音声をゆっくり読み上げます。"
        let referenceURL = directory.appendingPathComponent("reference.aiff")
        let say = Process()
        say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        say.arguments = ["-v", "Kyoko", "-r", "165", "-o", referenceURL.path, referenceText]
        try say.run()
        say.waitUntilExit()
        guard say.terminationStatus == 0 else {
            throw XCTSkip("The Kyoko Japanese reference voice is unavailable")
        }
        let reference = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 24_000)

        let target = "タロットで読み取っていきましょう。まずは、ゆっくりと心を落ち着けてください。"
        let waveforms = try await render(target: target, reference: reference, referenceText: referenceText)
        MLX.Memory.clearCache()
        let asr = try await Qwen3ASRModel.fromPretrained(modelId: Qwen3ASRModel.largeModelId)
        for (seed, waveform) in waveforms {
            XCTAssertEqual(waveform.count % 1920, 0, "Decoder lost a fraction of a codec frame for seed \(seed)")
            XCTAssertGreaterThan(waveform.count, 4 * 24_000, "The target sentence must finish")
            let transcript = asr.transcribe(audio: waveform, sampleRate: 24_000, language: "Japanese")
            print("Japanese ICL seed \(seed): \(transcript)")
            XCTAssertTrue(transcript.hasPrefix("タロット"), "First word was damaged: \(transcript)")
            XCTAssertTrue(transcript.contains("心を落ち着けてください"), "Sentence ending was damaged: \(transcript)")
            if let output = ProcessInfo.processInfo.environment["QWEN_TTS_TEST_OUTPUT_DIR"] {
                let outputURL = URL(fileURLWithPath: output, isDirectory: true)
                try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
                try WAVWriter.write(samples: waveform, sampleRate: 24_000,
                    to: outputURL.appendingPathComponent("japanese-\(seed).wav"))
            }
        }
    }

    private func render(
        target: String, reference: [Float], referenceText: String
    ) async throws -> [(UInt64, [Float])] {
        let (tts, encoder) = try await Qwen3TTSModel.fromPretrainedWithEncoder()
        return [UInt64(1_596_910_849), UInt64(1000)].map { seed in
            MLX.seed(seed)
            let waveform = tts.synthesizeWithVoiceCloneICL(
                text: target, referenceAudio: reference, referenceText: referenceText,
                language: "japanese", sampling: SamplingConfig(temperature: 0.3, maxTokens: 160),
                codecEncoder: encoder)
            return (seed, waveform)
        }
    }
}
#endif
