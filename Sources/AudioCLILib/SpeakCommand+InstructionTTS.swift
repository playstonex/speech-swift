import ArgumentParser
import AudioCommon
import FireRedTTS3TTS
import Foundation
import MossVoiceGeneratorTTS

extension SpeakCommand {
    func validateInstructionTTS() throws {
        let fireRed = engine.lowercased() == "fireredtts3"
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let instruct, !instruct.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              batchFile == nil, !listSpeakers, speaker == nil, voiceSample == nil, !cleanReference else {
            throw ValidationError("\(engine) requires one text argument and --instruct; reference audio, fixed speakers and batch input are unsupported")
        }
        let directory = fireRed ? fireredtts3BundleDir : mossVoicegeneratorBundleDir
        let modelID = fireRed ? fireredtts3ModelId : mossVoicegeneratorModelId
        guard (directory == nil) != (modelID == nil),
              !(directory ?? modelID ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("Choose exactly one --\(engine)-bundle-dir or --\(engine)-model-id for an exported bundle")
        }
        if fireRed {
            if stream { throw ValidationError("fireredtts3 currently returns complete audio; omit --stream") }
            if let language, !["en", "english", "zh", "chinese", "zh-cn", "mandarin"].contains(language.lowercased()) {
                throw ValidationError("FireRedTTS3-Instruct voice design supports English and Chinese")
            }
            try fireRedTTS3Options().validate()
        } else {
            if language != nil { throw ValidationError("MOSS-VoiceGenerator uses the script and --instruct for language; omit --language") }
            try mossVoiceGeneratorOptions().validate()
        }
    }

    func fireRedTTS3Options() -> FireRedTTS3Options {
        var options = FireRedTTS3Options()
        options.seed = seed ?? 2
        options.flowSteps = fireredtts3FlowSteps
        options.maxAudioPatches = fireredtts3MaxPatches
        return options
    }

    func mossVoiceGeneratorOptions() -> MossVoiceGeneratorOptions {
        MossVoiceGeneratorOptions(maxFrames: mossVoicegeneratorMaxFrames, seed: seed ?? 42,
                                  chunkFrames: mossVoicegeneratorChunkFrames)
    }

    func runFireRedTTS3() throws {
        try runAsync {
            guard let text, let instruct else { throw ValidationError("Text and --instruct are required") }
            let loadStart = Date()
            let model: FireRedTTS3Model
            if let directory = fireredtts3BundleDir {
                model = try await .fromBundle(URL(fileURLWithPath: directory), progressHandler: reportProgress)
            } else if let modelID = fireredtts3ModelId {
                model = try await .fromPretrained(modelId: modelID, progressHandler: reportProgress)
            } else { throw ValidationError("An exported FireRedTTS3-Instruct bundle is required") }
            defer { model.unload() }
            print(String(format: "Model load: %.2fs", Date().timeIntervalSince(loadStart)))
            let started = Date()
            let result = try await model.generateWithPlan(text: text, instruction: instruct, language: language,
                                                         options: fireRedTTS3Options())
            let elapsed = Date().timeIntervalSince(started)
            let duration = Double(result.samples.count) / Double(model.sampleRate)
            print(String(format: "Complete PCM: %.2fs; duration: %.2fs; RTF: %.3f", elapsed, duration, elapsed / max(duration, 0.001)))
            print(String(format: "Planning: %.2fs; acoustic generation: %.2fs; decoding: %.2fs",
                         result.planningSeconds, result.acousticSeconds, result.decodingSeconds))
            try WAVWriter.write(samples: result.samples, sampleRate: model.sampleRate, to: URL(fileURLWithPath: output))
            print("Saved audio to \(output)")
            if play { try await playInstructionAudio(result.samples, sampleRate: model.sampleRate) }
        }
    }

    func runMossVoiceGenerator() throws {
        try runAsync {
            guard let text, let instruct else { throw ValidationError("Text and --instruct are required") }
            let loadStart = Date()
            let model: MossVoiceGeneratorModel
            if let directory = mossVoicegeneratorBundleDir {
                model = try await .fromBundle(URL(fileURLWithPath: directory), progressHandler: reportProgress)
            } else if let modelID = mossVoicegeneratorModelId {
                model = try await .fromPretrained(modelId: modelID, progressHandler: reportProgress)
            } else { throw ValidationError("An exported MOSS-VoiceGenerator bundle is required") }
            defer { model.unload() }
            print(String(format: "Model load: %.2fs", Date().timeIntervalSince(loadStart)))
            let player = stream && play ? StreamingAudioPlayer() : nil
            defer { player?.stop() }
            if let player {
                player.preBufferDuration = 0.24
                player.skipInitialSilence = false
                try player.start(sampleRate: Double(model.sampleRate))
            }
            let result: MossVoiceGeneratorResult
            if stream {
                result = try await model.generateStreaming(text: text, instruction: instruct,
                    options: mossVoiceGeneratorOptions()) { chunk in
                    if !chunk.samples.isEmpty { player?.scheduleChunk(chunk.samples) }
                }
            } else {
                result = try await model.generate(text: text, instruction: instruct, options: mossVoiceGeneratorOptions())
            }
            print(String(format: "First PCM: %.0fms; duration: %.2fs; generation: %.2fs; RTF: %.3f",
                         result.firstAudioSeconds * 1000, result.duration, result.generationSeconds,
                         result.generationSeconds / max(result.duration, 0.001)))
            try WAVWriter.write(samples: result.samples, sampleRate: result.sampleRate, to: URL(fileURLWithPath: output))
            print("Saved audio to \(output)")
            if let player {
                player.markGenerationComplete()
                await player.waitForCompletion()
            } else if play { try await playInstructionAudio(result.samples, sampleRate: result.sampleRate) }
        }
    }

    private func playInstructionAudio(_ samples: [Float], sampleRate: Int) async throws {
        let player = StreamingAudioPlayer()
        defer { player.stop() }
        try player.play(samples: samples, sampleRate: sampleRate)
        await player.waitForCompletion()
    }
}
