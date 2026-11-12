import AVFoundation
import Foundation

struct DecodedAnalysisAudio {
    let channels: [[Float]]
    let analysisSampleRate: Double
    let source: MusicAudioMetadata
    let startSeconds: Double
    let selectedDuration: Double
    let isPartial: Bool
}

/// The source is decoded in small blocks. Only the selected, downsampled
/// analysis window is retained, independently of the complete file length.
enum AnalysisAudioReader {
    static let sampleRate = 11_025.0

    static func read(url: URL, options: MusicAnalysisOptions) throws -> DecodedAnalysisAudio {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        let format = file.processingFormat
        guard format.sampleRate.isFinite, (4_000.0...384_000.0).contains(format.sampleRate) else {
            throw MusicAnalysisError.invalidSampleRate
        }
        guard (1...2).contains(Int(format.channelCount)) else {
            throw MusicAnalysisError.unsupportedChannels
        }
        guard file.length > 0 else { throw MusicAnalysisError.emptyAudio }
        let duration = Double(file.length) / format.sampleRate
        guard options.startSeconds < duration else { throw MusicAnalysisError.rangeOutsideAudio }
        let startFrame = AVAudioFramePosition((options.startSeconds * format.sampleRate).rounded(.down))
        let wanted = AVAudioFramePosition((options.maxDurationSeconds * format.sampleRate).rounded(.down))
        let selected = min(wanted, file.length - startFrame)
        guard selected > 0 else { throw MusicAnalysisError.emptyAudio }
        file.framePosition = startFrame

        guard let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
            channels: format.channelCount, interleaved: false
        ), let converter = AVAudioConverter(from: format, to: target),
           let input = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8192),
           let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 4096) else {
            throw MusicAnalysisError.conversionFailed("Could not create the native sample-rate converter.")
        }
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        let expectedFrames = max(1, Int((Double(selected) * sampleRate / format.sampleRate).rounded()))
        let capacityLimit = expectedFrames + 8192
        var channels = [[Float]](repeating: [], count: Int(target.channelCount))
        for index in channels.indices { channels[index].reserveCapacity(expectedFrames) }
        var remaining = selected
        var readFrames: AVAudioFramePosition = 0
        var readError: Error?
        var stalledIterations = 0

        while true {
            let previousRead = readFrames
            let previousOutput = channels[0].count
            var conversionError: NSError?
            output.frameLength = 0
            let status = converter.convert(to: output, error: &conversionError) { requested, state in
                guard remaining > 0 else {
                    state.pointee = .endOfStream
                    return nil
                }
                do {
                    let count = AVAudioFrameCount(min(AVAudioFramePosition(min(requested, input.frameCapacity)), remaining))
                    try file.read(into: input, frameCount: count)
                    guard input.frameLength > 0 else {
                        remaining = 0
                        state.pointee = .endOfStream
                        return nil
                    }
                    remaining -= AVAudioFramePosition(input.frameLength)
                    readFrames += AVAudioFramePosition(input.frameLength)
                    state.pointee = .haveData
                    return input
                } catch {
                    readError = error
                    state.pointee = .endOfStream
                    return nil
                }
            }
            if let readError { throw readError }
            if let conversionError {
                throw MusicAnalysisError.conversionFailed(conversionError.localizedDescription)
            }
            if output.frameLength > 0 {
                guard let data = output.floatChannelData else {
                    throw MusicAnalysisError.conversionFailed("The converter did not return float PCM.")
                }
                guard channels[0].count + Int(output.frameLength) <= capacityLimit else {
                    throw MusicAnalysisError.conversionFailed("The converter exceeded the selected window.")
                }
                for channel in channels.indices {
                    channels[channel].append(contentsOf: UnsafeBufferPointer(
                        start: data[channel], count: Int(output.frameLength)
                    ))
                }
            }
            if status == .endOfStream { break }
            if status == .error {
                throw MusicAnalysisError.conversionFailed("The native converter reported an error.")
            }
            if readFrames == previousRead, channels[0].count == previousOutput {
                stalledIterations += 1
                guard stalledIterations <= 8 else {
                    throw MusicAnalysisError.conversionFailed("The native converter stopped making progress.")
                }
            } else {
                stalledIterations = 0
            }
        }
        for index in channels.indices {
            if channels[index].count > expectedFrames {
                channels[index].removeLast(channels[index].count - expectedFrames)
            }
        }
        guard !channels[0].isEmpty else { throw MusicAnalysisError.emptyAudio }
        return DecodedAnalysisAudio(
            channels: channels, analysisSampleRate: sampleRate,
            source: MusicAudioMetadata(
                fileName: url.lastPathComponent, sampleRate: format.sampleRate,
                channelCount: Int(format.channelCount), durationSeconds: duration
            ),
            startSeconds: Double(startFrame) / format.sampleRate,
            selectedDuration: Double(readFrames) / format.sampleRate,
            isPartial: startFrame > 0 || startFrame + readFrames < file.length
        )
    }
}
