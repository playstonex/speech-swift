import ArgumentParser
import AVFoundation
import Darwin
import Foundation
import XCTest
@testable import AudioCLILib

final class PracticeCommandTests: XCTestCase {
    func testParsesDefaultsAndRequiresOutput() throws {
        let root = try AudioCLI.parseAsRoot(["practice", "song.wav", "--output", "practice.wav"])
        let command = try XCTUnwrap(root as? PracticeCommand)
        XCTAssertEqual(command.start, 0)
        XCTAssertNil(command.duration)
        XCTAssertEqual(command.speed, 1)
        XCTAssertEqual(command.semitones, 0)
        XCTAssertEqual(command.loops, 1)
        XCTAssertEqual(command.fadeMs, 5)
        XCTAssertFalse(command.overwrite)
        XCTAssertFalse(command.json)
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["practice", "song.wav"]))
    }

    func testParsesIndependentTransformsRepeatAndJSON() throws {
        let root = try AudioCLI.parseAsRoot([
            "practice", "song.wav", "--start", "20", "--duration", "12", "--speed", "0.8",
            "--semitones", "-2", "--loops", "3", "--fade-ms", "10", "--output", "out.wav", "--overwrite", "--json",
        ])
        let command = try XCTUnwrap(root as? PracticeCommand)
        XCTAssertEqual(command.start, 20)
        XCTAssertEqual(command.duration, 12)
        XCTAssertEqual(command.speed, 0.8)
        XCTAssertEqual(command.semitones, -2)
        XCTAssertEqual(command.loops, 3)
        XCTAssertEqual(command.fadeMs, 10)
        XCTAssertTrue(command.overwrite)
        XCTAssertTrue(command.json)
    }

    func testRejectsInvalidTransformsAndOutputFormats() {
        let invalidOptions: [(arguments: [String], message: String)] = [
            (["--start", "-1"], "Start must be a finite, nonnegative number of seconds."),
            (["--duration", "-1"], "Duration must be greater than zero and at most 600 seconds."),
            (["--duration", "0"], "Duration must be greater than zero and at most 600 seconds."),
            (["--duration", "601"], "Duration must be greater than zero and at most 600 seconds."),
            (["--speed", "-1"], "Speed must be between 0.25 and 2.0."),
            (["--speed", "0"], "Speed must be between 0.25 and 2.0."),
            (["--speed", "3"], "Speed must be between 0.25 and 2.0."),
            (["--semitones", "-13"], "Pitch shift must be between -12 and 12 semitones."),
            (["--semitones", "13"], "Pitch shift must be between -12 and 12 semitones."),
            (["--loops", "-1"], "Loops must be between 1 and 100."),
            (["--loops", "0"], "Loops must be between 1 and 100."),
            (["--loops", "101"], "Loops must be between 1 and 100."),
            (["--fade-ms", "-1"], "Fade must be between 0 and 50 milliseconds."),
        ]
        for invalid in invalidOptions {
            XCTAssertThrowsError(try AudioCLI.parseAsRoot(
                ["practice", "song.wav", "--output", "out.wav"] + invalid.arguments)
            ) { error in
                XCTAssertEqual(AudioCLI.message(for: error), invalid.message)
            }
        }
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["practice", "song.wav", "--output", "out.mp3"]))
    }
}

final class E2EPracticeCommandTests: XCTestCase {
    func testRealWAVExportAndCleanJSONReport() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("practice-cli-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = directory.appendingPathComponent("source.wav")
        try writeTone(input)
        let output = directory.appendingPathComponent("practice.wav")
        let parsed = try AudioCLI.parseAsRoot([
            "practice", input.path, "--start", "0.1", "--duration", "0.25",
            "--loops", "2", "--output", output.path, "--json",
        ])
        let command = try XCTUnwrap(parsed as? PracticeCommand)
        let data = try captureStandardOutput { try command.run() }
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["schemaVersion"] as? Int, 1)
        XCTAssertEqual(json["outputFrames"] as? Int, 22_050)
        XCTAssertEqual(json["outputDurationSeconds"] as? Double, 0.5)
        XCTAssertEqual(json["loops"] as? Int, 2)
        let file = try AVAudioFile(forReading: output)
        XCTAssertEqual(file.length, 22_050)
        XCTAssertEqual(file.fileFormat.sampleRate, 44_100)
    }

    private func writeTone(_ output: URL) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)!
        let file = try AVAudioFile(forWriting: output, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 44_100,
            AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
        ])
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 44_100)!
        buffer.frameLength = 44_100
        for frame in 0..<44_100 {
            buffer.floatChannelData![0][frame] = Float(0.2 * sin(2 * .pi * 440 * Double(frame) / 44_100))
        }
        try file.write(from: buffer)
    }

    private func captureStandardOutput(_ body: () throws -> Void) throws -> Data {
        let pipe = Pipe()
        fflush(stdout)
        let original = dup(STDOUT_FILENO)
        guard original >= 0, dup2(pipe.fileHandleForWriting.fileDescriptor, STDOUT_FILENO) >= 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        defer {
            fflush(stdout)
            _ = dup2(original, STDOUT_FILENO)
            close(original)
        }
        try body()
        fflush(stdout)
        _ = dup2(original, STDOUT_FILENO)
        pipe.fileHandleForWriting.closeFile()
        return pipe.fileHandleForReading.readDataToEndOfFile()
    }
}
