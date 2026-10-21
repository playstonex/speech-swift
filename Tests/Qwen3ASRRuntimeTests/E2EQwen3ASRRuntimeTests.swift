import XCTest
import Darwin
import AudioCommon
import Qwen3ASR
import Qwen3ASRRuntime

/// Real-weight parity for the optional ownership wrapper. Run in its own
/// process after the no-download contract tests.
final class E2EQwen3ASRRuntimeTests: XCTestCase {
    func testMLXRuntimeMatchesNativeTranscriptAndReloads() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let audio = try AudioFileLoader.load(
            url: root.appendingPathComponent("Qwen3ASRTests/Resources/test_audio.wav"), targetSampleRate: 16000)
        let native = try await Qwen3ASRModel.fromPretrained(offlineMode: true)
        let expected = native.transcribe(audio: audio, sampleRate: 16000, language: "english")
        native.unload()
        XCTAssertThrowsError(try native.transcribeThrowing(audio: audio)) { error in
            XCTAssertTrue(error is AudioModelError, "Unloaded models must report an inference error")
        }
        XCTAssertTrue(expected.lowercased().contains("replacement"))
        XCTAssertTrue(expected.lowercased().contains("tomorrow"))
        let runtime = Qwen3ASRRuntime(offlineMode: true)
        let managed = try await runtime.transcribe(audio: audio, language: "english")
        XCTAssertEqual(managed, expected)
        try await runtime.unload()
        let reloaded = try await runtime.transcribe(audio: audio, language: "english")
        XCTAssertEqual(reloaded, expected)
        let cancelled = Task {
            try await runtime.transcribe(audio: Array(repeating: audio, count: 8).flatMap { $0 },
                                         language: "english")
        }
        try await Task.sleep(nanoseconds: 20_000_000)
        cancelled.cancel()
        do { _ = try await cancelled.value; XCTFail("Managed inference must observe cancellation") }
        catch is CancellationError { }
        let recovered = try await runtime.transcribe(audio: audio, language: "english")
        XCTAssertEqual(recovered, expected)
        try await runtime.unload()
    }
}

/// Separate suite so the CoreML and MLX model lifetimes are process-isolated.
final class E2EQwen3ASRCoreMLRuntimeTests: XCTestCase {
    func testCoreMLDescriptorLoadsCoreMLPipelineAndTranscribes() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let audio = try AudioFileLoader.load(
            url: root.appendingPathComponent("Qwen3ASRTests/Resources/test_audio.wav"), targetSampleRate: 16000)
        let runtime = Qwen3ASRRuntime(descriptor: Qwen3ASRDescriptors.coreML, offlineMode: true)
        let text = try await runtime.transcribe(audio: audio, language: "english")
        XCTAssertTrue(text.lowercased().contains("replacement"), text)
        XCTAssertTrue(text.lowercased().contains("tomorrow"), text)
        try await runtime.unload()
    }

    func testRepeatedCoreMLCallsReleaseScratchBuffersOnBackgroundExecutor() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let audio = try AudioFileLoader.load(
            url: root.appendingPathComponent("Qwen3ASRTests/Resources/test_audio.wav"), targetSampleRate: 16000)
        // A single background task must remain bounded without relying on a
        // main run loop, a request task ending, or an executor hop to drain it.
        let growth = try await Task.detached(priority: .userInitiated) {
            let runtime = Qwen3ASRRuntime(descriptor: Qwen3ASRDescriptors.coreML, offlineMode: true)
            for _ in 0..<3 { _ = try await runtime.transcribe(audio: audio, language: "english") }
            let before = physicalFootprintBytes()
            for _ in 0..<12 {
                let text = try await runtime.transcribe(audio: audio, language: "english")
                guard text.lowercased().contains("tomorrow") else {
                    throw AudioModelError.inferenceFailed(operation: "transcribe", reason: "Repeated inference changed output")
                }
            }
            let after = physicalFootprintBytes()
            try await runtime.unload()
            return max(Int64(after) - Int64(before), 0)
        }.value
        // Allows allocator/cache noise while rejecting the former ~100 MB
        // retained per utterance. Models and compiled graphs are already warm.
        XCTAssertLessThan(growth, 256 * 1024 * 1024, "CoreML scratch memory accumulated across requests")
    }
}

private func physicalFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    precondition(status == KERN_SUCCESS)
    return UInt64(info.phys_footprint)
}
