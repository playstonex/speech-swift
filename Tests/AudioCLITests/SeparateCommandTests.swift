import ArgumentParser
import AVFoundation
import XCTest
@testable import AudioCLILib

final class SeparateCommandTests: XCTestCase {
    func testDefaultOutputsRemainTheFourStems() throws {
        let command = try parse(["song.wav"])
        XCTAssertFalse(command.instrumental)
        XCTAssertEqual(try command.outputTargets().map(\.rawValue), ["vocals", "drums", "bass", "other"])
        XCTAssertEqual(try command.inferenceTargets().map(\.rawValue), ["vocals", "drums", "bass", "other"])
    }

    func testInstrumentalFlagParsesForBothEngines() throws {
        for engine in ["umx", "htdemucs"] {
            let command = try parse(["song.wav", "--engine", engine, "--instrumental"])
            XCTAssertEqual(command.engine, engine)
            XCTAssertTrue(command.instrumental)
        }
    }

    func testSelectedOutputsKeepTheirFilterWhileInstrumentalInfersAllStems() throws {
        let command = try parse(["song.wav", "--stems", "vocals", "--instrumental"])
        XCTAssertEqual(try command.outputTargets().map(\.rawValue), ["vocals"])
        XCTAssertEqual(try command.inferenceTargets().map(\.rawValue), ["vocals", "drums", "bass", "other"])
    }

    func testBackingOnlySelectionStillKeepsVocalEstimateInWienerDenominator() throws {
        let command = try parse(["song.wav", "--stems", "bass", "--instrumental"])
        XCTAssertEqual(try command.outputTargets().map(\.rawValue), ["bass"])
        XCTAssertEqual(try command.inferenceTargets().map(\.rawValue), ["vocals", "drums", "bass", "other"])
    }

    func testSelectionWithoutInstrumentalKeepsExistingInferenceSubset() throws {
        let command = try parse(["song.wav", "--stems", " vocals, bass,vocals "])
        XCTAssertEqual(try command.outputTargets().map(\.rawValue), ["vocals", "bass"])
        XCTAssertEqual(try command.inferenceTargets().map(\.rawValue), ["vocals", "bass"])
    }

    func testInvalidStemListsFailInsteadOfSilentlyDroppingNames() {
        for stems in ["typo", "vocals,typo", "", ",", "vocals,"] {
            XCTAssertThrowsError(try parse(["song.wav", "--stems", stems, "--instrumental"])) { error in
                XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
            }
        }
    }

    private func parse(_ arguments: [String]) throws -> SeparateCommand {
        let root = try AudioCLI.parseAsRoot(["separate"] + arguments)
        return try XCTUnwrap(root as? SeparateCommand)
    }
}

final class E2ESeparateCommandTests: XCTestCase {
    func testUMXInstrumentalExportFromRealFile() throws {
        try assertInstrumentalExport(engine: "umx")
    }

    func testHTDemucsInstrumentalExportFromRealFile() throws {
        try assertInstrumentalExport(engine: "htdemucs")
    }

    private func assertInstrumentalExport(engine: String) throws {
        guard let input = ProcessInfo.processInfo.environment["SPEECH_SEPARATION_TEST_AUDIO"] else {
            throw XCTSkip("Set SPEECH_SEPARATION_TEST_AUDIO to a short recording with clearly audible singing and instrumental accompaniment")
        }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("separate-instrumental-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let parsed = try AudioCLI.parseAsRoot([
            "separate", input, "--engine", engine,
            "--htdemucs-precision", "int8", "--stems", "vocals", "--instrumental",
            "--output-dir", directory.path,
        ])
        let command = try XCTUnwrap(parsed as? SeparateCommand)
        try command.run()

        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertEqual(Set(files), ["vocals.wav", "instrumental.wav"])
        let source = try AVAudioFile(forReading: URL(fileURLWithPath: input))
        let expectedFrames = Double(source.length) * 44100 / source.processingFormat.sampleRate
        for name in files {
            let audio = try AVAudioFile(forReading: directory.appendingPathComponent(name))
            XCTAssertEqual(audio.fileFormat.channelCount, 2)
            XCTAssertEqual(audio.fileFormat.sampleRate, 44100)
            XCTAssertEqual(Double(audio.length), expectedFrames, accuracy: 2)
            XCTAssertGreaterThan(audio.length, 0)
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: audio.processingFormat, frameCapacity: 4096))
            var peak: Float = 0
            var squaredSum: Double = 0
            var sampleCount = 0
            while audio.framePosition < audio.length {
                try audio.read(into: buffer)
                let channels = try XCTUnwrap(buffer.floatChannelData)
                for channel in 0..<Int(buffer.format.channelCount) {
                    for frame in 0..<Int(buffer.frameLength) {
                        let value = channels[channel][frame]
                        XCTAssertTrue(value.isFinite)
                        peak = max(peak, abs(value))
                        squaredSum += Double(value) * Double(value)
                        sampleCount += 1
                    }
                }
            }
            XCTAssertGreaterThan(peak, 0.0001, "\(name) must contain audible nonzero audio")
            XCTAssertLessThanOrEqual(peak, 1)
            let rms = sqrt(squaredSum / Double(max(sampleCount, 1)))
            XCTAssertGreaterThan(rms, 0.001,
                "\(name) is nearly silent; use a fixture with audible singing AND accompaniment")
        }
    }
}
