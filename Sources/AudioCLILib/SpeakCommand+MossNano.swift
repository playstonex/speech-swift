import AudioCommon
import CoreML
import Foundation
import MossTTSNano

extension SpeakCommand {
    func runMossNano() throws {
        try runAsync {
            let units: MLComputeUnits = mossNanoComputeUnits == "gpu" ? .cpuAndGPU : .cpuOnly
            let model: MossTTSNanoModel
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: mossNanoModel, isDirectory: &isDirectory), isDirectory.boolValue {
                model = try MossTTSNanoModel(modelDirectory: URL(fileURLWithPath: mossNanoModel), computeUnits: units)
            } else {
                model = try await MossTTSNanoModel.fromPretrained(modelId: mossNanoModel, computeUnits: units)
            }
            let started = Date()
            var reference: MossTTSNanoReference?
            if let voiceSample {
                if cleanReference {
                    reference = try await model.encodeReference(channels: [loadReference(path: voiceSample, targetSampleRate: 48000)], sampleRate: 48000)
                } else {
                    reference = try await model.encodeReference(url: URL(fileURLWithPath: voiceSample))
                }
            }
            let options = MossTTSNanoOptions(maxFrames: mossNanoMaxFrames, seed: mossNanoSeed)
            let referenceSeconds = Date().timeIntervalSince(started)
            var left: [Float] = [], right: [Float] = []
            if stream {
                let player = play ? StreamingAudioPlayer() : nil
                player?.preBufferDuration = 0.24
                if let player { try player.start(sampleRate: 48000) }
                defer { player?.stop() }
                var first = true
                for try await chunk in model.synthesizeStream(text: text!, reference: reference, options: options) {
                    if !chunk.left.isEmpty && first {
                        print(String(format: "First audio: %.0f ms (includes reference preparation)", Date().timeIntervalSince(started) * 1000))
                        first = false
                    }
                    left += chunk.left; right += chunk.right
                    player?.scheduleChunk(chunk.mono)
                }
                player?.markGenerationComplete()
                if let player { await player.waitForCompletion() }
            } else {
                let result = try await model.synthesize(text: text!, reference: reference, options: options)
                left = result.left; right = result.right
                print(String(format: "First token: %.0f ms; first audio: %.0f ms (includes reference preparation; model loading excluded)",
                    (referenceSeconds + result.firstTokenSeconds) * 1000, (referenceSeconds + result.firstAudioSeconds) * 1000))
            }
            try WAVWriter.writeStereo(left: left, right: right, sampleRate: 48000, to: URL(fileURLWithPath: output))
            print(String(format: "Saved %.2f s of 48 kHz stereo audio to %@ in %.2f s", Double(left.count) / 48000, output, Date().timeIntervalSince(started)))
            if play && !stream {
                let player = StreamingAudioPlayer()
                try player.play(samples: zip(left, right).map { ($0 + $1) * 0.5 }, sampleRate: 48000)
                await player.waitForCompletion()
                player.stop()
            }
        }
    }
}
