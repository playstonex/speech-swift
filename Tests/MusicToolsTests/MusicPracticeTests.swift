import AVFoundation
import Foundation
import XCTest
@testable import MusicTools

final class MusicPracticeTests: XCTestCase {
    func testTrimAndRepeatPreserveFrameCountStereoBalanceAndSource() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("source.caf")
            try makeTone(input, seconds: 2, frequency: 440, channels: 2)
            let original = try Data(contentsOf: input)
            let output = directory.appendingPathComponent("practice.wav")
            let result = try MusicPracticeExporter.export(
                input: input, output: output,
                options: MusicPracticeOptions(startSeconds: 0.25, durationSeconds: 0.5, loops: 3))
            let pcm = try readPCM(output)
            XCTAssertEqual(result.schemaVersion, 1)
            XCTAssertEqual(result.channels, 2)
            XCTAssertEqual(result.outputFrames, 66_150)
            XCTAssertEqual(result.outputDurationSeconds, 1.5, accuracy: 1 / 44_100)
            XCTAssertEqual(result.selectedStartSeconds, 0.25)
            XCTAssertEqual(result.selectedDurationSeconds, 0.5)
            XCTAssertEqual(pcm.samples[0].count, 66_150)
            XCTAssertEqual(pcm.samples[0][0], 0)
            XCTAssertEqual(pcm.samples[0][22_049], 0)
            XCTAssertEqual(Array(pcm.samples[0][0..<22_050]), Array(pcm.samples[0][22_050..<44_100]))
            XCTAssertEqual(rms(pcm.samples[1]) / rms(pcm.samples[0]), 0.5, accuracy: 0.001)
            XCTAssertEqual(try Data(contentsOf: input), original)
            XCTAssertEqual(result.appliedGain, 1)
            XCTAssertEqual(try JSONSerialization.jsonObject(with: JSONEncoder().encode(result)) is [String: Any], true)
        }
    }

    func testSlowerPassageKeepsPitchAndHasAudibleLeadingAndFinalContent() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("tone.caf")
            try makeTone(input, seconds: 2, frequency: 440)
            let output = directory.appendingPathComponent("slow.wav")
            let result = try MusicPracticeExporter.export(
                input: input, output: output,
                options: MusicPracticeOptions(startSeconds: 0.25, durationSeconds: 1, speed: 0.5))
            let pcm = try readPCM(output)
            XCTAssertEqual(result.outputDurationSeconds, 2, accuracy: 0.001)
            XCTAssertEqual(pcm.samples[0].count, 88_200)
            XCTAssertEqual(dominantFrequency(pcm.samples[0], rate: pcm.rate, around: 440), 440, accuracy: 3)
            XCTAssertGreaterThan(rms(Array(pcm.samples[0][2_205..<6_615])), 0.03)
            XCTAssertGreaterThan(rms(Array(pcm.samples[0][81_585..<85_995])), 0.03)
        }
    }

    func testOctavePitchShiftKeepsDuration() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("tone.caf")
            try makeTone(input, seconds: 2, frequency: 220)
            let output = directory.appendingPathComponent("octave.wav")
            let result = try MusicPracticeExporter.export(
                input: input, output: output,
                options: MusicPracticeOptions(startSeconds: 0.25, durationSeconds: 1, semitones: 12))
            let pcm = try readPCM(output)
            XCTAssertEqual(result.outputDurationSeconds, 1, accuracy: 0.001)
            XCTAssertEqual(pcm.samples[0].count, 44_100)
            XCTAssertEqual(dominantFrequency(pcm.samples[0], rate: pcm.rate, around: 440), 440, accuracy: 3)
        }
    }

    func testCombinedTransposeSpeedAndRepetitionAreBakedIntoWAV() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("tone.caf")
            try makeTone(input, seconds: 2, frequency: 440)
            let output = directory.appendingPathComponent("practice.wav")
            let result = try MusicPracticeExporter.export(
                input: input, output: output,
                options: MusicPracticeOptions(startSeconds: 0.1, durationSeconds: 1, speed: 0.8, semitones: -2, loops: 3))
            let pcm = try readPCM(output)
            XCTAssertEqual(result.outputDurationSeconds, 3.75, accuracy: 0.001)
            XCTAssertEqual(pcm.samples[0].count, 165_375)
            let expected = 440 * pow(2, -2.0 / 12)
            XCTAssertEqual(dominantFrequency(pcm.samples[0], rate: pcm.rate, around: expected), expected, accuracy: 3)
            XCTAssertTrue(pcm.samples.allSatisfy { $0.allSatisfy { $0.isFinite && abs($0) <= 1 } })
        }
    }

    func testSparseBeginningAndFinalBurstsSurviveSlowAndFastPassageExport() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("sparse-edge-bursts.caf")
            try makeEdgeBursts(input)
            for (speed, semitones) in [(0.25, 0.0), (2.0, 0.0), (0.25, 3.0), (2.0, -3.0)] {
                let label = "speed \(speed), semitones \(semitones)"
                let output = directory.appendingPathComponent("edges-\(speed)-\(semitones).wav")
                let result = try MusicPracticeExporter.export(
                    input: input, output: output,
                    options: MusicPracticeOptions(
                        startSeconds: 0.2, durationSeconds: 1, speed: speed, semitones: semitones))
                let pcm = try readPCM(output)
                let expectedFrames = Int(ceil(44_100 / speed))
                XCTAssertEqual(result.outputFrames, Int64(expectedFrames), label)
                XCTAssertEqual(pcm.samples[0].count, expectedFrames, label)
                guard pcm.samples[0].count == expectedFrames else { continue }
                let samples = pcm.samples[0]

                // Each source burst is 80 ms. Permit an additional 80 ms of
                // output spread for TimePitch's transient/window behavior.
                // This guards against losing a whole edge phrase, rather
                // than asserting sample-perfect phase-vocoder transients.
                let edgeSeconds = min(result.outputDurationSeconds * 0.3, 0.08 / speed + 0.08)
                let edgeFrames = Int((edgeSeconds * pcm.rate).rounded())
                let firstRMS = rms(Array(samples.prefix(edgeFrames)))
                let finalRMS = rms(Array(samples.suffix(edgeFrames)))
                XCTAssertGreaterThan(firstRMS, 0.025, "Beginning burst was lost: \(label)")
                XCTAssertGreaterThan(finalRMS, 0.025, "Final burst was lost: \(label)")

                let quietStart = edgeFrames + Int((0.04 * pcm.rate).rounded())
                let quietEnd = samples.count - quietStart
                XCTAssertGreaterThan(quietEnd, quietStart, label)
                guard quietEnd > quietStart else { continue }
                let quietRMS = rms(Array(samples[quietStart..<quietEnd]))
                XCTAssertLessThan(quietRMS, 0.01, "Silent interior was filled with audio: \(label)")
                XCTAssertLessThan(quietRMS, min(firstRMS, finalRMS) * 0.2, label)
            }
        }
    }

    func testLoudInputUsesOneGainAndDoesNotBoostQuietInput() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("loud.caf")
            try makeTone(input, seconds: 0.1, frequency: 250, channels: 2, amplitude: 2)
            let output = directory.appendingPathComponent("bounded.wav")
            let result = try MusicPracticeExporter.export(
                input: input, output: output,
                options: MusicPracticeOptions(fadeMilliseconds: 0))
            let pcm = try readPCM(output)
            XCTAssertEqual(result.appliedGain, 0.5, accuracy: 0.001)
            XCTAssertLessThanOrEqual(pcm.samples[0].map { abs($0) }.max()!, 1)
            XCTAssertEqual(rms(pcm.samples[1]) / rms(pcm.samples[0]), 0.5, accuracy: 0.001)
        }
    }

    func testExistingOutputSurvivesRejectionAndOnlyExplicitOverwriteReplacesIt() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("tone.caf")
            try makeTone(input, seconds: 0.1, frequency: 440)
            let output = directory.appendingPathComponent("existing.wav")
            let sentinel = Data("original destination".utf8)
            try sentinel.write(to: output)
            XCTAssertThrowsError(try MusicPracticeExporter.export(input: input, output: output))
            XCTAssertEqual(try Data(contentsOf: output), sentinel)
            _ = try MusicPracticeExporter.export(input: input, output: output, options: MusicPracticeOptions(overwrite: true))
            XCTAssertNotEqual(try Data(contentsOf: output), sentinel)
            XCTAssertEqual(try readPCM(output).samples[0].count, 4_410)
        }
    }

    func testInputAndOutputIdentityAndSymlinksAreRejectedEvenWithOverwrite() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("source.wav")
            try makeTone(input, seconds: 0.1, frequency: 440)
            let original = try Data(contentsOf: input)
            let alias = directory.appendingPathComponent("alias.wav")
            try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: input)
            for output in [input, alias] {
                XCTAssertThrowsError(try MusicPracticeExporter.export(input: input, output: output, options: MusicPracticeOptions(overwrite: true)))
            }
            XCTAssertEqual(try Data(contentsOf: input), original)
        }
    }

    func testInvalidSelectionsAndOptionsLeaveNoOutputOrScratchFiles() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("source.caf")
            try makeTone(input, seconds: 1, frequency: 440)
            let output = directory.appendingPathComponent("out.wav")
            for options in [
                MusicPracticeOptions(startSeconds: 1),
                MusicPracticeOptions(startSeconds: 0.5, durationSeconds: 1),
                MusicPracticeOptions(speed: 0),
                MusicPracticeOptions(speed: .nan),
                MusicPracticeOptions(semitones: .infinity),
                MusicPracticeOptions(loops: 0),
                MusicPracticeOptions(durationSeconds: 601),
                MusicPracticeOptions(durationSeconds: 600, speed: 0.25, loops: 100),
            ] {
                XCTAssertThrowsError(try MusicPracticeExporter.export(input: input, output: output, options: options))
                XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
                XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: directory.path).contains { $0.hasPrefix(".speech-practice-") })
            }
        }
    }

    func testNonFiniteInputIsRejectedBeforePublishing() throws {
        try withFixture { directory in
            let input = directory.appendingPathComponent("invalid.caf")
            let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)!
            try {
                let file = try AVAudioFile(forWriting: input, settings: floatSettings(channels: 1))
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8)!
                buffer.frameLength = 8
                for index in 0..<8 { buffer.floatChannelData![0][index] = index == 3 ? .nan : 0.1 }
                try file.write(from: buffer)
            }()
            let output = directory.appendingPathComponent("out.wav")
            XCTAssertThrowsError(try MusicPracticeExporter.export(input: input, output: output)) { error in
                guard let practiceError = error as? MusicPracticeError else {
                    return XCTFail("Expected a practice error, received \(error)")
                }
                switch practiceError {
                case .nonFiniteAudio: break
                default: XCTFail("Expected non-finite sample rejection, received \(error)")
                }
            }
            XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        }
    }

    private func withFixture(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("music-practice-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    private func floatSettings(channels: Int) -> [String: Any] {
        [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 44_100,
         AVNumberOfChannelsKey: channels, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true]
    }

    private func makeTone(
        _ url: URL, seconds: Double, frequency: Double, channels: Int = 1, amplitude: Double = 0.25
    ) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: AVAudioChannelCount(channels))!
        let file = try AVAudioFile(forWriting: url, settings: floatSettings(channels: channels))
        let frames = AVAudioFrameCount((seconds * 44_100).rounded())
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for channel in 0..<channels {
            for frame in 0..<Int(frames) {
                buffer.floatChannelData![channel][frame] = Float(amplitude * (channel == 0 ? 1 : 0.5) * sin(2 * .pi * frequency * Double(frame) / 44_100))
            }
        }
        try file.write(from: buffer)
    }

    private func makeEdgeBursts(_ output: URL) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)!
        let file = try AVAudioFile(forWriting: output, settings: floatSettings(channels: 1))
        let frames: AVAudioFrameCount = 66_150
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        let selectionStart = 8_820
        let selectionFrames = 44_100
        let burstFrames = 3_528
        for frame in 0..<Int(frames) {
            let relative = frame - selectionStart
            let frequency: Double?
            if (0..<burstFrames).contains(relative) {
                frequency = 440
            } else if ((selectionFrames - burstFrames)..<selectionFrames).contains(relative) {
                frequency = 660
            } else {
                frequency = nil
            }
            buffer.floatChannelData![0][frame] = frequency.map {
                Float(0.35 * sin(2 * .pi * $0 * Double(frame) / 44_100))
            } ?? 0
        }
        try file.write(from: buffer)
    }

    private func readPCM(_ url: URL) throws -> (samples: [[Float]], rate: Double) {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: buffer)
        let channels = (0..<Int(buffer.format.channelCount)).map { channel in
            Array(UnsafeBufferPointer(start: buffer.floatChannelData![channel], count: Int(buffer.frameLength)))
        }
        return (channels, file.processingFormat.sampleRate)
    }

    private func rms(_ samples: [Float]) -> Double {
        sqrt(samples.reduce(0) { $0 + Double($1) * Double($1) } / Double(samples.count))
    }

    private func dominantFrequency(_ input: [Float], rate: Double, around expected: Double) -> Double {
        let middle = input.count / 2
        let length = min(16_384, input.count / 2)
        let samples = input[(middle - length / 2)..<(middle + length / 2)].map(Double.init)
        var winner = expected
        var bestPower = -Double.infinity
        // Goertzel bins directly test the rendered tone, independently of the exporter.
        for offset in -40...40 {
            let frequency = expected + Double(offset) * 0.5
            let coefficient = 2 * cos(2 * .pi * frequency / rate)
            var previous: Double = 0
            var previousPrevious: Double = 0
            for sample in samples {
                let next = sample + coefficient * previous - previousPrevious
                previousPrevious = previous
                previous = next
            }
            let power = previous * previous + previousPrevious * previousPrevious - coefficient * previous * previousPrevious
            if power > bestPower { bestPower = power; winner = frequency }
        }
        return winner
    }
}
