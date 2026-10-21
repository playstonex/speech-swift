import AudioCommon
import CoreML
import Foundation
import Qwen3ASR
import XCTest
@testable import MossTTSNano

final class E2EMossTTSNanoTests: XCTestCase {
    private func load() async throws -> MossTTSNanoModel {
        if let path = ProcessInfo.processInfo.environment["MOSS_NANO_MODEL_DIR"] {
            return try MossTTSNanoModel(modelDirectory: URL(fileURLWithPath: path), computeUnits: .cpuOnly)
        }
        return try await MossTTSNanoModel.fromPretrained(computeUnits: .cpuOnly)
    }

    func testSynthesisCloningStreamingAndASR() async throws {
        let model = try await load()
        let text = "Hello world. This is a speech test."
        let original = try await model.synthesize(text: text, options: .init(maxFrames: 150, seed: 0))
        XCTAssertEqual(original.left.count, original.codes.count * 3840)
        XCTAssertEqual(original.left.count, original.right.count)
        XCTAssertGreaterThan(original.duration, 0.5)
        XCTAssertTrue(original.left.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(original.left.map(abs).max() ?? 0, 0.01)
        let reference = try await model.encodeReference(channels: [original.left, original.right], sampleRate: 48000)
        XCTAssertEqual(reference.codes.count, original.codes.count)
        let sentence = "The weather is nice today."
        let options = MossTTSNanoOptions(maxFrames: 150, seed: 1)
        let cloned = try await model.synthesize(text: sentence, reference: reference, options: options)
        var streamedLeft: [Float] = [], streamedRight: [Float] = [], finalCount = 0
        for try await chunk in model.synthesizeStream(text: sentence, reference: reference, options: options) {
            XCTAssertEqual(chunk.frameIndex, streamedLeft.count)
            streamedLeft += chunk.left; streamedRight += chunk.right
            if chunk.isFinal { finalCount += 1 }
        }
        XCTAssertEqual(finalCount, 1)
        XCTAssertEqual(streamedLeft, cloned.left)
        XCTAssertEqual(streamedRight, cloned.right)
        let asr = try await CoreMLASRModel.fromPretrained()
        func normalized(_ value: String) -> String {
            value.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }.joined(separator: " ")
        }
        for (expected, audio) in [(text, original.mono), (sentence, cloned.mono)] {
            let transcript = asr.transcribe(audio: audio, sampleRate: 48000, language: "english")
            print("[MOSS-NANO-ASR] \(transcript)")
            XCTAssertEqual(normalized(transcript), normalized(expected))
        }
    }

    func testFreshStateAndInvalidRequests() async throws {
        let model = try await load()
        let options = MossTTSNanoOptions(maxFrames: 120, seed: 0)
        let a = try await model.synthesize(text: "Hello world.", options: options)
        let b = try await model.synthesize(text: "Hello world.", options: options)
        XCTAssertEqual(a.codes, b.codes)
        do {
            _ = try await model.synthesize(text: "Hello", options: .init(maxFrames: 1024))
            XCTFail("expected context overflow")
        } catch MossTTSNanoError.contextExceeded {}
        do {
            _ = try await model.synthesize(text: "Hello world.", options: .init(maxFrames: 1))
            XCTFail("expected truncation error")
        } catch MossTTSNanoError.generationLimit {}
        do {
            _ = try await model.encodeReference(channels: [[.nan]], sampleRate: 48000)
            XCTFail("expected invalid reference")
        } catch MossTTSNanoError.invalidInput {}
        let longReference = FileManager.default.temporaryDirectory.appendingPathComponent("moss-reference-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: longReference) }
        try WAVWriter.write(samples: [Float](repeating: 0, count: 48000 * 83), sampleRate: 48000, to: longReference)
        do {
            _ = try await model.encodeReference(url: longReference)
            XCTFail("expected oversized reference rejection")
        } catch MossTTSNanoError.contextExceeded {}
        // A failed/truncated request must not contaminate the next request's state.
        let c = try await model.synthesize(text: "Hello world.", options: options)
        XCTAssertEqual(a.codes, c.codes)
    }

    func testPythonCPUCodeParity() async throws {
        guard ProcessInfo.processInfo.environment["MOSS_NANO_VERIFY_CODES"] == "1" else {
            throw XCTSkip("Set MOSS_NANO_VERIFY_CODES=1 for exact Python codec-token comparison on the matched M5 Pro export host")
        }
        let model = try await load()
        let result = try await model.synthesize(text: "Hello from Core ML.", options: .init(maxFrames: 150, seed: 0))
        let url = try XCTUnwrap(Bundle.module.url(forResource: "hello_cpu_codes", withExtension: "json"))
        let expected = try JSONDecoder().decode([[Int32]].self, from: Data(contentsOf: url))
        XCTAssertEqual(result.codes, expected)
        let referenceURL = try XCTUnwrap(Bundle.module.url(forResource: "reference", withExtension: "wav"))
        let reference = try await model.encodeReference(url: referenceURL)
        let clone = try await model.synthesize(text: "The weather is nice today.", reference: reference, options: .init(maxFrames: 150, seed: 1))
        let codesURL = try XCTUnwrap(Bundle.module.url(forResource: "clone_cpu_only_codes", withExtension: "json"))
        XCTAssertEqual(clone.codes, try JSONDecoder().decode([[Int32]].self, from: Data(contentsOf: codesURL)))
        let asr = try await CoreMLASRModel.fromPretrained()
        let transcript = asr.transcribe(audio: clone.mono, sampleRate: 48000, language: "english")
        print("[MOSS-NANO-CPU-ASR] \(transcript)")
        XCTAssertEqual(transcript.lowercased().filter { $0.isLetter || $0.isWhitespace }.trimmingCharacters(in: .whitespacesAndNewlines), "the weather is nice today")
    }

    func testGPUCloningCodeParityAndASR() async throws {
        guard CoreMLComputeUnitsResolver.resolved(default: .cpuAndGPU) == .cpuAndGPU else {
            throw XCTSkip("GPU execution was overridden by the CoreML compute-unit setting")
        }
        let model: MossTTSNanoModel
        if let path = ProcessInfo.processInfo.environment["MOSS_NANO_MODEL_DIR"] {
            model = try MossTTSNanoModel(modelDirectory: URL(fileURLWithPath: path), computeUnits: .cpuAndGPU)
        } else { model = try await MossTTSNanoModel.fromPretrained(computeUnits: .cpuAndGPU) }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "reference", withExtension: "wav"))
        let reference = try await model.encodeReference(url: url)
        let result = try await model.synthesize(text: "The weather is nice today.", reference: reference, options: .init(maxFrames: 150, seed: 1))
        let expectedURL = try XCTUnwrap(Bundle.module.url(forResource: "clone_cpu_and_gpu_codes", withExtension: "json"))
        let expected = try JSONDecoder().decode([[Int32]].self, from: Data(contentsOf: expectedURL))
        if ProcessInfo.processInfo.environment["MOSS_NANO_VERIFY_CODES"] == "1" { XCTAssertEqual(result.codes, expected) }
        let asr = try await CoreMLASRModel.fromPretrained()
        let transcript = asr.transcribe(audio: result.mono, sampleRate: 48000, language: "english")
        print("[MOSS-NANO-GPU-ASR] \(transcript)")
        XCTAssertEqual(transcript.lowercased().filter { $0.isLetter || $0.isWhitespace }.trimmingCharacters(in: .whitespacesAndNewlines), "the weather is nice today")
    }
}
