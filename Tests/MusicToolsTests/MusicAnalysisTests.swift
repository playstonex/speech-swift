import AVFoundation
import Foundation
import XCTest
@testable import MusicTools

final class MusicAnalysisTests: XCTestCase {
    private let rate = 11_025.0

    func testClickTracksAtSeveralTemposHaveUsefulEstimates() throws {
        for bpm in [90.0, 120.0, 150.0] {
            let analysis = try MusicAnalyzer.analyze(samples: clicks(bpm: bpm, seconds: 14), sampleRate: rate)
            let estimate = try XCTUnwrap(analysis.tempo.estimatedBPM, "Missing estimate for \(bpm) BPM")
            XCTAssertEqual(estimate, bpm, accuracy: 2.5)
            XCTAssertGreaterThan(analysis.tempo.onsetCount, 10)
            XCTAssertTrue(analysis.tempo.candidates.allSatisfy { (0...1).contains($0.fitScore) })
        }
    }

    func testRegularOnsetsExposeHalfTempoAmbiguity() {
        var novelty = [Double](repeating: 0, count: 1400)
        for index in stride(from: 50, to: novelty.count, by: 50) { novelty[index] = 1 }
        let result = TempoEstimator.estimate(novelty: novelty, frameRate: 100, durationSeconds: 14, rms: 0.1)
        XCTAssertEqual(result.estimatedBPM ?? 0, 120, accuracy: 0.1)
        XCTAssertTrue(result.ambiguous)
        XCTAssertTrue(result.candidates.contains { abs($0.bpm - 60) < 1 })
    }

    func testSilenceHasNullEstimatesAndDeterministicJSON() throws {
        let result = try MusicAnalyzer.analyze(samples: [Float](repeating: 0, count: Int(rate * 6)), sampleRate: rate)
        XCTAssertNil(result.tempo.estimatedBPM)
        XCTAssertNil(result.key.estimatedKey)
        XCTAssertTrue(result.tempo.candidates.isEmpty)
        XCTAssertTrue(result.key.candidates.isEmpty)
        let data = try result.jsonData()
        XCTAssertEqual(data, try result.jsonData())
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let tempo = try XCTUnwrap(object["tempo"] as? [String: Any])
        let key = try XCTUnwrap(object["key"] as? [String: Any])
        XCTAssertTrue(tempo["estimatedBPM"] is NSNull)
        XCTAssertTrue(key["estimatedKey"] is NSNull)
        XCTAssertEqual(try JSONDecoder().decode(MusicAnalysis.self, from: data), result)
    }

    func testSustainedSingleToneDoesNotInventATempoOrKey() throws {
        let result = try MusicAnalyzer.analyze(samples: chord(notes: [69], seconds: 7), sampleRate: rate)
        XCTAssertNil(result.tempo.estimatedBPM)
        XCTAssertNil(result.key.estimatedKey)
        XCTAssertTrue(result.key.candidates.isEmpty)
    }

    func testBroadbandNoiseDoesNotInventAKeyOrPeriodicTempo() throws {
        var seed: UInt64 = 1729
        let noise = (0..<Int(rate * 12)).map { _ -> Float in
            seed = seed &* 6_364_136_223_846_793_005 &+ 1
            return (Float(UInt32(truncatingIfNeeded: seed >> 32)) / Float(UInt32.max) * 2 - 1) * 0.05
        }
        let result = try MusicAnalyzer.analyze(samples: noise, sampleRate: rate)
        XCTAssertNil(result.tempo.estimatedBPM)
        XCTAssertNil(result.key.estimatedKey)
    }

    func testMajorAndMinorProgressionsHaveMatchingCandidates() throws {
        let major = progression([[48, 52, 55], [53, 57, 60], [55, 59, 62], [48, 52, 55]])
        let minor = progression([[45, 48, 52], [50, 53, 57], [52, 56, 59], [45, 48, 52]])
        let c = try MusicAnalyzer.analyze(samples: major, sampleRate: rate)
        let a = try MusicAnalyzer.analyze(samples: minor, sampleRate: rate)
        XCTAssertEqual(c.key.candidates.first?.name, "C major")
        XCTAssertEqual(a.key.candidates.first?.name, "A minor")
        XCTAssertGreaterThan(c.key.pitchedFrameCount, 3)
        XCTAssertGreaterThan(a.key.pitchedFrameCount, 3)
    }

    func testFlatPitchClassEvidenceDoesNotInventAKey() {
        let result = KeyEstimator.estimate(chroma: [Double](repeating: 1, count: 12), pitchedFrameCount: 100)
        XCTAssertNil(result.estimatedKey)
        XCTAssertTrue(result.candidates.isEmpty)
    }

    func testCloseRelativeTemplatesRemainAmbiguous() {
        let cMajor = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        let aMinor = [5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52]
        let mixed = zip(cMajor, aMinor).map { pair in (pair.0 + pair.1) / 2 }
        let result = KeyEstimator.estimate(chroma: mixed, pitchedFrameCount: 100)
        XCTAssertTrue(result.ambiguous)
        XCTAssertNil(result.estimatedKey)
        XCTAssertEqual(Set(result.candidates.prefix(2).map(\.name)), ["C major", "A minor"])
    }

    func testOppositePhaseStereoKeepsTheSameMusicalEvidence() throws {
        let samples = progression([[48, 52, 55], [53, 57, 60], [55, 59, 62], [48, 52, 55]])
        let mono = try MusicAnalyzer.analyze(samples: samples, sampleRate: rate)
        let stereo = try MusicAnalyzer.analyze(channels: [samples, samples.map { -$0 }], sampleRate: rate)
        XCTAssertGreaterThan(stereo.signal.rms, 0.01)
        XCTAssertEqual(stereo.key, mono.key)
    }

    func testWindowBoundsAreReportedWithoutPretendingToAnalyzeTheFullFile() throws {
        let result = try MusicAnalyzer.analyze(
            samples: clicks(bpm: 120, seconds: 12), sampleRate: rate,
            options: MusicAnalysisOptions(startSeconds: 2, maxDurationSeconds: 5)
        )
        XCTAssertEqual(result.source.durationSeconds, 12, accuracy: 1 / rate)
        XCTAssertEqual(result.analyzedRange.startSeconds, 2, accuracy: 1 / rate)
        XCTAssertEqual(result.analyzedRange.durationSeconds, 5, accuracy: 1 / rate)
        XCTAssertEqual(result.analyzedRange.analysisFrameCount, Int(rate * 5))
        XCTAssertTrue(result.analyzedRange.isPartial)
    }

    func testBoundedFileReaderPreservesStereoMetadataAndSelectedRange() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("music-analysis-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let samples = chord(notes: [48, 52, 55], seconds: 4)
        try writeAudio(channels: [samples, samples], sampleRate: rate, url: url)
        let result = try MusicAnalyzer.analyze(
            url: url, options: MusicAnalysisOptions(startSeconds: 1.25, maxDurationSeconds: 0.5)
        )
        XCTAssertEqual(result.source.channelCount, 2)
        XCTAssertEqual(result.source.durationSeconds, 4, accuracy: 1 / rate)
        XCTAssertEqual(result.analyzedRange.startSeconds, 1.25, accuracy: 1 / rate)
        XCTAssertEqual(result.analyzedRange.durationSeconds, 0.5, accuracy: 1 / rate)
        XCTAssertEqual(result.analyzedRange.analysisSampleRate, 11_025)
        XCTAssertLessThanOrEqual(abs(result.analyzedRange.analysisFrameCount - 5513), 1)
        XCTAssertTrue(result.analyzedRange.isPartial)
        XCTAssertGreaterThan(result.signal.rms, 0.01)
        XCTAssertNil(result.tempo.estimatedBPM)
    }

    func testNativeSampleRateConversionAnalyzesTheSelectedAudio() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("music-analysis-src-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let sourceRate = 44_100.0
        let samples = (0..<Int(sourceRate * 2)).map { i in Float(0.2 * sin(2 * .pi * 440 * Double(i) / sourceRate)) }
        try writeAudio(channels: [samples], sampleRate: sourceRate, url: url)
        let result = try MusicAnalyzer.analyze(url: url, options: MusicAnalysisOptions(maxDurationSeconds: 1))
        XCTAssertEqual(result.source.sampleRate, sourceRate)
        XCTAssertEqual(result.analyzedRange.analysisSampleRate, rate)
        XCTAssertLessThanOrEqual(abs(result.analyzedRange.analysisFrameCount - Int(rate)), 1)
        XCTAssertGreaterThan(result.signal.rms, 0.1)
        XCTAssertNil(result.tempo.estimatedBPM)
    }

    func testInvalidAudioAndOptionsFailCleanly() throws {
        let valid = [Float](repeating: 0, count: 100)
        XCTAssertThrowsError(try MusicAnalyzer.analyze(samples: [], sampleRate: rate))
        XCTAssertThrowsError(try MusicAnalyzer.analyze(samples: valid, sampleRate: .nan))
        XCTAssertThrowsError(try MusicAnalyzer.analyze(samples: valid, sampleRate: 0))
        XCTAssertThrowsError(try MusicAnalyzer.analyze(channels: [valid, [0]], sampleRate: rate))
        XCTAssertThrowsError(try MusicAnalyzer.analyze(samples: [.nan], sampleRate: rate))
        XCTAssertThrowsError(try MusicAnalyzer.analyze(samples: [.infinity], sampleRate: rate))
        XCTAssertThrowsError(try MusicAnalyzer.analyze(samples: valid, sampleRate: rate,
                                                     options: MusicAnalysisOptions(startSeconds: 10)))
        for seconds in [0.0, -1, .nan, .infinity, 121] {
            XCTAssertThrowsError(try MusicAnalyzer.analyze(samples: valid, sampleRate: rate,
                                                         options: MusicAnalysisOptions(maxDurationSeconds: seconds)))
        }
        let corrupt = FileManager.default.temporaryDirectory.appendingPathComponent("music-analysis-corrupt-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: corrupt) }
        try Data([1, 2, 3]).write(to: corrupt)
        XCTAssertThrowsError(try MusicAnalyzer.analyze(url: corrupt))
    }

    func testLargeFiniteSamplesStillProduceFiniteSerializableStatistics() throws {
        let result = try MusicAnalyzer.analyze(
            samples: [Float](repeating: .greatestFiniteMagnitude, count: Int(rate)), sampleRate: rate
        )
        XCTAssertTrue(result.signal.rms.isFinite)
        XCTAssertTrue(result.signal.peak.isFinite)
        XCTAssertNoThrow(try result.jsonData())
    }

    private func clicks(bpm: Double, seconds: Double) -> [Float] {
        var result = [Float](repeating: 0, count: Int(rate * seconds))
        var state: UInt64 = 42
        var position = 0.4
        while position < seconds - 0.15 {
            let first = Int(position * rate)
            for offset in 0..<90 where first + offset < result.count {
                state = state &* 6_364_136_223_846_793_005 &+ 1
                let noise = Double(UInt32(truncatingIfNeeded: state >> 32)) / Double(UInt32.max) * 2 - 1
                result[first + offset] += Float(noise * exp(-Double(offset) / 22) * 0.8)
            }
            position += 60 / bpm
        }
        return result
    }

    private func chord(notes: [Int], seconds: Double) -> [Float] {
        (0..<Int(rate * seconds)).map { i in
            let time = Double(i) / rate
            let sum = notes.reduce(0.0) { partial, note in
                let frequency = 440 * pow(2, Double(note - 69) / 12)
                return partial + sin(2 * .pi * frequency * time)
            }
            return Float(sum * 0.18 / Double(notes.count))
        }
    }

    private func progression(_ chords: [[Int]]) -> [Float] {
        var result: [Float] = []
        for notes in chords { result.append(contentsOf: chord(notes: notes, seconds: 2)) }
        return result
    }

    private func writeAudio(channels: [[Float]], sampleRate: Double, url: URL) throws {
        let format = try XCTUnwrap(AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
            channels: AVAudioChannelCount(channels.count), interleaved: false
        ))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(channels[0].count)))
        buffer.frameLength = buffer.frameCapacity
        let data = try XCTUnwrap(buffer.floatChannelData)
        for channel in channels.indices {
            for frame in channels[channel].indices { data[channel][frame] = channels[channel][frame] }
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }
}

final class E2EMusicAnalysisTests: XCTestCase {
    func testSuppliedRecordingHasValidWindowAndSerializableEstimates() throws {
        guard let path = ProcessInfo.processInfo.environment["SPEECH_MUSIC_ANALYSIS_TEST_AUDIO"] else {
            throw XCTSkip("Set SPEECH_MUSIC_ANALYSIS_TEST_AUDIO to an audio recording.")
        }
        let result = try MusicAnalyzer.analyze(
            url: URL(fileURLWithPath: path), options: MusicAnalysisOptions(maxDurationSeconds: 10)
        )
        XCTAssertGreaterThan(result.source.durationSeconds, 0)
        XCTAssertGreaterThan(result.analyzedRange.analysisFrameCount, 0)
        XCTAssertLessThanOrEqual(result.analyzedRange.durationSeconds, 10.001)
        XCTAssertTrue(result.signal.rms.isFinite)
        XCTAssertTrue(result.tempo.candidates.allSatisfy { $0.bpm.isFinite && $0.fitScore.isFinite })
        XCTAssertTrue(result.key.candidates.allSatisfy { $0.fitScore.isFinite })
        XCTAssertNoThrow(try result.jsonData())
    }
}
