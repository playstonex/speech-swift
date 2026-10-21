#if canImport(AVFoundation)
import AVFoundation
import XCTest
@testable import AudioCommon

final class MicrophoneResamplingTests: XCTestCase {
    /// Regression for live AEC: a 1024-frame 48 kHz callback represents
    /// 341 1/3 samples at 16 kHz. Truncating each callback independently made
    /// the microphone lose 2,000 samples in 128 seconds while a system-audio
    /// reference kept correct time, invalidating a previously acquired delay.
    func testStreamingConversionDoesNotAccumulatePerCallbackRoundingDrift() throws {
        let inputRate = 48_000.0
        let outputRate = 16_000.0
        let inputFormat = try XCTUnwrap(AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: inputRate,
            channels: 1,
            interleaved: false))
        let outputFormat = try XCTUnwrap(AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: outputRate,
            channels: 1,
            interleaved: false))
        let converter = try XCTUnwrap(AVAudioConverter(
            from: inputFormat, to: outputFormat))

        let callbackCount = 6_000
        let inputFramesPerCallback = 1_024
        var actualOutputFrames = 0
        for callback in 0..<callbackCount {
            let input = try XCTUnwrap(AVAudioPCMBuffer(
                pcmFormat: inputFormat,
                frameCapacity: AVAudioFrameCount(inputFramesPerCallback)))
            input.frameLength = AVAudioFrameCount(inputFramesPerCallback)
            if let samples = input.floatChannelData?[0] {
                for frame in 0..<inputFramesPerCallback {
                    samples[frame] = sin(Float(callback * inputFramesPerCallback + frame) * 0.01)
                }
            }
            let output = try XCTUnwrap(AudioIO.resampleMicrophoneBuffer(
                input, with: converter, to: outputFormat))
            actualOutputFrames += output.count
        }

        let expectedOutputFrames = Int((
            Double(callbackCount * inputFramesPerCallback)
                * outputRate / inputRate).rounded())
        XCTAssertLessThanOrEqual(
            abs(actualOutputFrames - expectedOutputFrames),
            16,
            "Streaming conversion accumulated callback-by-callback clock drift")
    }
}
#endif
