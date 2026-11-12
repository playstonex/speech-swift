import AVFoundation
import Foundation
import Darwin

/// Parameters for exporting one repeated, optionally transformed audio passage.
public struct MusicPracticeOptions: Codable, Sendable, Equatable {
    public var startSeconds: Double
    public var durationSeconds: Double?
    public var speed: Double
    public var semitones: Double
    public var loops: Int
    public var fadeMilliseconds: Double
    public var overwrite: Bool

    public init(
        startSeconds: Double = 0,
        durationSeconds: Double? = nil,
        speed: Double = 1,
        semitones: Double = 0,
        loops: Int = 1,
        fadeMilliseconds: Double = 5,
        overwrite: Bool = false
    ) {
        self.startSeconds = startSeconds
        self.durationSeconds = durationSeconds
        self.speed = speed
        self.semitones = semitones
        self.loops = loops
        self.fadeMilliseconds = fadeMilliseconds
        self.overwrite = overwrite
    }

    public func validate() throws {
        guard startSeconds.isFinite, startSeconds >= 0 else {
            throw MusicPracticeError.invalidOptions("Start must be a finite, nonnegative number of seconds.")
        }
        if let durationSeconds {
            guard durationSeconds.isFinite, durationSeconds > 0, durationSeconds <= 600 else {
                throw MusicPracticeError.invalidOptions("Duration must be greater than zero and at most 600 seconds.")
            }
        }
        guard speed.isFinite, (0.25...2).contains(speed) else {
            throw MusicPracticeError.invalidOptions("Speed must be between 0.25 and 2.0.")
        }
        guard semitones.isFinite, (-12...12).contains(semitones) else {
            throw MusicPracticeError.invalidOptions("Pitch shift must be between -12 and 12 semitones.")
        }
        guard (1...100).contains(loops) else {
            throw MusicPracticeError.invalidOptions("Loops must be between 1 and 100.")
        }
        guard fadeMilliseconds.isFinite, (0...50).contains(fadeMilliseconds) else {
            throw MusicPracticeError.invalidOptions("Fade must be between 0 and 50 milliseconds.")
        }
        if let durationSeconds, durationSeconds * Double(loops) / speed > 3600 {
            throw MusicPracticeError.invalidOptions("The repeated output must not exceed one hour.")
        }
    }
}

/// Frame counts describe the completed file, rather than a playback preview.
public struct MusicPracticeResult: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let sourceDurationSeconds: Double
    public let selectedStartSeconds: Double
    public let selectedDurationSeconds: Double
    public let speed: Double
    public let semitones: Double
    public let loops: Int
    public let fadeMilliseconds: Double
    public let sampleRate: Double
    public let channels: Int
    public let passageFrames: Int64
    public let outputFrames: Int64
    public let outputDurationSeconds: Double
    public let appliedGain: Double
    public let outputPath: String
}

public enum MusicPracticeError: Error, LocalizedError {
    case invalidOptions(String)
    case inputIsOutput
    case outputExists
    case unsupportedAudioFormat
    case invalidRange(String)
    case rendererStalled
    case renderingFailed
    case nonFiniteAudio
    case publicationFailed(Int32)

    public var errorDescription: String? {
        switch self {
        case .invalidOptions(let message), .invalidRange(let message): return message
        case .inputIsOutput: return "Input and output must be different files, including symbolic links."
        case .outputExists: return "The output already exists. Choose another name or pass --overwrite."
        case .unsupportedAudioFormat: return "Practice export supports mono or stereo audio from 8 to 192 kHz."
        case .rendererStalled: return "The offline audio renderer stopped making progress."
        case .renderingFailed: return "The offline audio renderer failed."
        case .nonFiniteAudio: return "Audio processing produced a non-finite sample."
        case .publicationFailed(let code): return "Unable to publish the completed WAV file (POSIX error \(code))."
        }
    }
}

/// Native, offline passage rendering with bounded PCM buffers and staged output.
public enum MusicPracticeExporter {
    private static let blockFrames: AVAudioFrameCount = 4096

    public static func export(
        input inputURL: URL,
        output outputURL: URL,
        options: MusicPracticeOptions = MusicPracticeOptions()
    ) throws -> MusicPracticeResult {
        try options.validate()
        let input = inputURL.standardizedFileURL.resolvingSymlinksInPath()
        let output = outputURL.standardizedFileURL
        let manager = FileManager.default
        guard !sameFile(input, output) else { throw MusicPracticeError.inputIsOutput }
        guard output.pathExtension.lowercased() == "wav" else {
            throw MusicPracticeError.invalidOptions("The output filename must end in .wav.")
        }
        if manager.fileExists(atPath: output.path), !options.overwrite { throw MusicPracticeError.outputExists }
        var isDirectory: ObjCBool = false
        if manager.fileExists(atPath: output.path, isDirectory: &isDirectory), isDirectory.boolValue {
            throw MusicPracticeError.invalidOptions("The output path is a directory, not a WAV file.")
        }

        let source = try AVAudioFile(forReading: input, commonFormat: .pcmFormatFloat32, interleaved: false)
        let format = source.processingFormat
        guard (1...2).contains(format.channelCount), format.sampleRate.isFinite,
              (8_000...192_000).contains(format.sampleRate), source.length > 0 else {
            throw MusicPracticeError.unsupportedAudioFormat
        }
        let rate = format.sampleRate
        let sourceDuration = Double(source.length) / rate
        guard options.startSeconds < sourceDuration else {
            throw MusicPracticeError.invalidRange("Start must be before the end of the input recording.")
        }
        let start = AVAudioFramePosition(floor(options.startSeconds * rate))
        let available = source.length - start
        let selected: AVAudioFramePosition
        if let duration = options.durationSeconds {
            let requested = AVAudioFramePosition((duration * rate).rounded())
            guard requested > 0, requested <= available else {
                throw MusicPracticeError.invalidRange("The requested passage extends past the end of the input recording.")
            }
            selected = requested
        } else {
            selected = available
        }
        guard Double(selected) / rate <= 600 else {
            throw MusicPracticeError.invalidRange("Select a passage of at most 600 seconds with --start and --duration.")
        }
        let passageFrames = AVAudioFramePosition(ceil(Double(selected) / options.speed))
        let totalFrames = passageFrames * AVAudioFramePosition(options.loops)
        guard Double(totalFrames) / rate <= 3600 else {
            throw MusicPracticeError.invalidOptions("The repeated output must not exceed one hour.")
        }

        let parent = output.deletingLastPathComponent()
        try manager.createDirectory(at: parent, withIntermediateDirectories: true)
        let working = parent.appendingPathComponent(".speech-practice-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: working, withIntermediateDirectories: false)
        defer { try? manager.removeItem(at: working) }
        let passageURL = working.appendingPathComponent("passage.caf")
        let stagedOutput = working.appendingPathComponent("practice.wav")

        let peak: Double
        if options.speed == 1, options.semitones == 0 {
            peak = try copyPassage(source: source, start: start, frames: selected, to: passageURL)
        } else {
            peak = try transformPassage(
                source: source, start: start, inputFrames: selected,
                outputFrames: passageFrames, options: options, to: passageURL)
        }
        let gain = peak > 1 ? 1 / peak : 1
        try writeRepeatedPassage(
            passageURL, to: stagedOutput, frames: passageFrames,
            options: options, gain: gain, format: format)

        // Recheck after rendering: an existing destination must remain intact on failure.
        guard !sameFile(input, output) else { throw MusicPracticeError.inputIsOutput }
        if options.overwrite {
            guard rename(stagedOutput.path, output.path) == 0 else {
                throw MusicPracticeError.publicationFailed(errno)
            }
        } else {
            try manager.moveItem(at: stagedOutput, to: output)
        }
        return MusicPracticeResult(
            schemaVersion: 1, sourceDurationSeconds: sourceDuration,
            selectedStartSeconds: Double(start) / rate,
            selectedDurationSeconds: Double(selected) / rate,
            speed: options.speed, semitones: options.semitones, loops: options.loops,
            fadeMilliseconds: options.fadeMilliseconds, sampleRate: rate,
            channels: Int(format.channelCount), passageFrames: passageFrames,
            outputFrames: totalFrames, outputDurationSeconds: Double(totalFrames) / rate,
            appliedGain: gain, outputPath: output.path)
    }

    private static func sameFile(_ first: URL, _ second: URL) -> Bool {
        if first.standardizedFileURL.resolvingSymlinksInPath() == second.standardizedFileURL.resolvingSymlinksInPath() {
            return true
        }
        guard let a = try? FileManager.default.attributesOfItem(atPath: first.path),
              let b = try? FileManager.default.attributesOfItem(atPath: second.path),
              let aDevice = a[.systemNumber] as? NSNumber,
              let bDevice = b[.systemNumber] as? NSNumber,
              let aNode = a[.systemFileNumber] as? NSNumber,
              let bNode = b[.systemFileNumber] as? NSNumber else { return false }
        return aDevice == bDevice && aNode == bNode
    }

    private static func floatFile(at url: URL, format: AVAudioFormat) throws -> AVAudioFile {
        try AVAudioFile(forWriting: url, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
        ], commonFormat: .pcmFormatFloat32, interleaved: false)
    }

    private static func checkedPeak(_ buffer: AVAudioPCMBuffer) throws -> Double {
        guard let data = buffer.floatChannelData else { throw MusicPracticeError.unsupportedAudioFormat }
        var peak: Double = 0
        for channel in 0..<Int(buffer.format.channelCount) {
            for frame in 0..<Int(buffer.frameLength) {
                let value = data[channel][frame]
                guard value.isFinite else { throw MusicPracticeError.nonFiniteAudio }
                peak = max(peak, abs(Double(value)))
            }
        }
        return peak
    }

    private static func copyPassage(
        source: AVAudioFile, start: AVAudioFramePosition, frames: AVAudioFramePosition, to url: URL
    ) throws -> Double {
        let destination = try floatFile(at: url, format: source.processingFormat)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: source.processingFormat, frameCapacity: blockFrames) else {
            throw MusicPracticeError.unsupportedAudioFormat
        }
        source.framePosition = start
        var remaining = frames
        var peak: Double = 0
        while remaining > 0 {
            try source.read(into: buffer, frameCount: AVAudioFrameCount(min(remaining, AVAudioFramePosition(blockFrames))))
            guard buffer.frameLength > 0 else { throw MusicPracticeError.rendererStalled }
            peak = max(peak, try checkedPeak(buffer))
            try destination.write(from: buffer)
            remaining -= AVAudioFramePosition(buffer.frameLength)
        }
        return peak
    }

    private static func transformPassage(
        source: AVAudioFile, start: AVAudioFramePosition, inputFrames: AVAudioFramePosition,
        outputFrames: AVAudioFramePosition, options: MusicPracticeOptions, to url: URL
    ) throws -> Double {
        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let timePitch = AVAudioUnitTimePitch()
        let format = source.processingFormat
        timePitch.rate = Float(options.speed)
        timePitch.pitch = Float(options.semitones * 100)
        timePitch.overlap = 8
        engine.attach(player)
        engine.attach(timePitch)
        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: blockFrames)
        defer { player.stop(); engine.stop() }
        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)
        player.scheduleSegment(source, startingFrame: start, frameCount: AVAudioFrameCount(inputFrames), at: nil)
        try engine.start()
        player.play()

        let latency = timePitch.auAudioUnit.latency
        guard latency.isFinite, latency >= 0, latency <= 2 else { throw MusicPracticeError.renderingFailed }
        var discardFrames = AVAudioFramePosition(ceil(latency * format.sampleRate))
        let destination = try floatFile(at: url, format: format)
        guard let rendered = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: blockFrames),
              let kept = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: blockFrames) else {
            throw MusicPracticeError.unsupportedAudioFormat
        }
        var written: AVAudioFramePosition = 0
        var stalls = 0
        var peak: Double = 0
        while written < outputFrames {
            let requested = AVAudioFrameCount(min(AVAudioFramePosition(blockFrames), outputFrames - written + discardFrames))
            let status = try engine.renderOffline(requested, to: rendered)
            switch status {
            case .success:
                guard rendered.frameLength > 0 else {
                    stalls += 1
                    if stalls > 64 { throw MusicPracticeError.rendererStalled }
                    continue
                }
                stalls = 0
                _ = try checkedPeak(rendered)
                let skip = min(discardFrames, AVAudioFramePosition(rendered.frameLength))
                discardFrames -= skip
                let count = min(AVAudioFramePosition(rendered.frameLength) - skip, outputFrames - written)
                if count > 0 {
                    guard let incoming = rendered.floatChannelData, let outgoing = kept.floatChannelData else {
                        throw MusicPracticeError.unsupportedAudioFormat
                    }
                    kept.frameLength = AVAudioFrameCount(count)
                    for channel in 0..<Int(format.channelCount) {
                        memcpy(outgoing[channel], incoming[channel].advanced(by: Int(skip)), Int(count) * MemoryLayout<Float>.stride)
                    }
                    peak = max(peak, try checkedPeak(kept))
                    try destination.write(from: kept)
                    written += count
                }
            case .cannotDoInCurrentContext, .insufficientDataFromInputNode:
                stalls += 1
                guard stalls <= 64 else { throw MusicPracticeError.rendererStalled }
                Thread.sleep(forTimeInterval: 0.001)
            case .error:
                throw MusicPracticeError.renderingFailed
            @unknown default:
                throw MusicPracticeError.renderingFailed
            }
        }
        return peak
    }

    private static func writeRepeatedPassage(
        _ input: URL, to output: URL, frames: AVAudioFramePosition,
        options: MusicPracticeOptions, gain: Double, format: AVAudioFormat
    ) throws {
        let source = try AVAudioFile(forReading: input, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard source.length == frames else { throw MusicPracticeError.renderingFailed }
        let destination = try AVAudioFile(forWriting: output, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
        ], commonFormat: .pcmFormatFloat32, interleaved: false)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: blockFrames) else {
            throw MusicPracticeError.unsupportedAudioFormat
        }
        let fadeFrames = min(AVAudioFramePosition((options.fadeMilliseconds * format.sampleRate / 1000).rounded()), frames / 2)
        for _ in 0..<options.loops {
            source.framePosition = 0
            var offset: AVAudioFramePosition = 0
            while offset < frames {
                try source.read(into: buffer, frameCount: AVAudioFrameCount(min(AVAudioFramePosition(blockFrames), frames - offset)))
                guard buffer.frameLength > 0, let data = buffer.floatChannelData else { throw MusicPracticeError.rendererStalled }
                for frame in 0..<Int(buffer.frameLength) {
                    let absolute = offset + AVAudioFramePosition(frame)
                    var envelope: Double = 1
                    if fadeFrames > 0 {
                        envelope = min(1, min(Double(absolute) / Double(fadeFrames), Double(frames - 1 - absolute) / Double(fadeFrames)))
                    }
                    for channel in 0..<Int(format.channelCount) {
                        let sample = Double(data[channel][frame]) * gain * envelope
                        guard sample.isFinite else { throw MusicPracticeError.nonFiniteAudio }
                        data[channel][frame] = Float(max(-1, min(1, sample)))
                    }
                }
                try destination.write(from: buffer)
                offset += AVAudioFramePosition(buffer.frameLength)
            }
        }
    }
}
