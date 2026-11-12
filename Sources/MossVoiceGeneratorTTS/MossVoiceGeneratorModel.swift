// Adapted from MOSS-VoiceGenerator and MOSS-Audio-Tokenizer.
// Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
// Licensed under Apache 2.0; see LICENSE.
// Native Swift adaptation uses request-local state and bounded causal caches.

import AudioCommon
import Foundation
import MLX
import MLXNN
import MLXCommon

/// Reference-free voice design using a free-form instruction and exact script.
/// One generation may run per instance. No Python runtime or reference audio is required.
public final class MossVoiceGeneratorModel:SpeechGenerationModel,ModelMemoryManageable,@unchecked Sendable {
    public let configuration:MossVoiceGeneratorConfiguration
    public let bundleDirectory:URL
    public var sampleRate:Int { configuration.sampleRate }
    let network:MVGLanguageModel
    let codec:MVGCodec
    private var tokenizer:MVGTokenizer?
    private let generationLock=NSLock()
    private var generating=false
    private var loaded=true
    private let weightBytes:Int
    private var cacheLease=MVGCachePolicy.acquire()

    private static let requiredFiles=["config.json","model.safetensors","codec_config.json",
        "codec.safetensors","tokenizer.json","tokenizer_config.json","mvg_export.json"]
    private init(configuration:MossVoiceGeneratorConfiguration,directory:URL,network:MVGLanguageModel,
                 codec:MVGCodec,tokenizer:MVGTokenizer) {
        self.configuration=configuration;bundleDirectory=directory;self.network=network
        self.codec=codec;self.tokenizer=tokenizer
        weightBytes=network.parameterMemoryBytes()+codec.parameterMemoryBytes()
    }

    public var isLoaded:Bool {generationLock.withLock {loaded}}
    public var memoryFootprint:Int {generationLock.withLock {loaded ? weightBytes : 0}}

    /// Marks the model unavailable immediately. If synthesis is active, it
    /// stops at its next boundary before weights and its cache lease are freed.
    public func unload() {
        let clearNow=generationLock.withLock { ()->Bool in
            guard loaded else {return false};loaded=false;return !generating
        }
        if clearNow {releaseResources()}
    }
    private func releaseResources() {
        // Clear only this model's parameters. Avoid the shared unload helper:
        // it also flushes spare buffers that other loaded models may reuse.
        network.apply(filter:Module.filterAll) {_ in MLXArray()}
        codec.apply(filter:Module.filterAll) {_ in MLXArray()}
        tokenizer=nil
        cacheLease?.release();cacheLease=nil
    }
    private func ensureLoaded()throws {
        guard generationLock.withLock({loaded}) else {throw MossVoiceGeneratorError.unloaded}
    }
    private func finishGeneration() {
        let clearNow=generationLock.withLock { ()->Bool in generating=false;return !loaded }
        if clearNow {releaseResources()}
    }

    /// Load a converted bundle. The upstream tokenizer checkpoint must first be
    /// reduced to its synthesis decoder; an unconverted upstream folder is rejected.
    public static func fromBundle(_ directory:URL,
        progressHandler:((Double,String)->Void)?=nil)async throws->MossVoiceGeneratorModel {
        try Task.checkCancellation()
        for name in requiredFiles where !FileManager.default.fileExists(atPath:directory.appendingPathComponent(name).path) {
            throw MossVoiceGeneratorError.invalidConfiguration("missing \(name)")
        }
        let manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("mvg_export.json"))) as? [String:Any]
        guard manifest?["format"] as? String=="moss_voice_generator_mlx_v1",
              let codecInfo=manifest?["codec"] as? [String:Any],codecInfo["decoder_only"] as? Bool==true,
              codecInfo["codebooks"] as? Int==16,codecInfo["precision"] as? String=="float32" else {
            throw MossVoiceGeneratorError.invalidConfiguration("expected BF16 language and lossless FP32 synthesis codec export")
        }
        let config=try JSONDecoder().decode(MossVoiceGeneratorConfiguration.self,
            from:Data(contentsOf:directory.appendingPathComponent("config.json")))
        try config.validate()
        let codecConfig=try JSONDecoder().decode(MVGCodecConfiguration.self,
            from:Data(contentsOf:directory.appendingPathComponent("codec_config.json")))
        try codecConfig.validate()
        let tokenizer=try await MVGTokenizer.load(directory)
        try Task.checkCancellation()
        let loadingLease=MVGCachePolicy.acquire()
        defer { loadingLease?.release() }
        progressHandler?(0.1,"Loading MOSS-VoiceGenerator language weights")
        let network=withThrowawayRandomState { MVGLanguageModel(config) }
        let weights=try MLX.loadArrays(url:directory.appendingPathComponent("model.safetensors"))
        guard !weights.isEmpty,weights.values.allSatisfy({$0.dtype == .bfloat16}) else {
            throw MossVoiceGeneratorError.invalidConfiguration("language weights must retain upstream BF16 precision")
        }
        try network.update(parameters:ModuleParameters.unflattened(weights),verify:.all)
        eval(network.parameters())
        try Task.checkCancellation()
        progressHandler?(0.6,"Loading MOSS-VoiceGenerator audio decoder")
        let codec=withThrowawayRandomState { MVGCodec(codecConfig) }
        try codec.load(from:directory.appendingPathComponent("codec.safetensors"))
        try Task.checkCancellation()
        progressHandler?(1,"MOSS-VoiceGenerator ready")
        return MossVoiceGeneratorModel(configuration:config,directory:directory,network:network,codec:codec,tokenizer:tokenizer)
    }

    /// Download a previously converted compatible bundle. No default repository
    /// is assumed; callers choose the exact reviewed artifact.
    public static func fromPretrained(modelId:String,cacheDir:URL?=nil,offlineMode:Bool=false,
        downloadOptions:DownloadOptions = .init(),progressHandler:((Double,String)->Void)?=nil
    )async throws->MossVoiceGeneratorModel {
        let directory=try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for:modelId)
        try await HuggingFaceDownloader.downloadFiles(modelId:modelId,to:directory,files:requiredFiles,
            offlineMode:offlineMode,downloadOptions:downloadOptions) {progress in
            progressHandler?(progress*0.8,"Downloading MOSS-VoiceGenerator")
        }
        return try await fromBundle(directory) {progress,message in progressHandler?(0.8+0.2*progress,message)}
    }

    public func generate(text:String,language:String?=nil)async throws->[Float] {
        try ensureLoaded()
        if let language,!language.isEmpty {
            throw MossVoiceGeneratorError.invalidInput("put language and delivery requirements in the voice instruction")
        }
        return try await generate(text:text,instruction:"Speak naturally and clearly.").samples
    }

    public func generateStream(text:String,language:String?=nil)->AsyncThrowingStream<AudioChunk,Error> {
        AsyncThrowingStream {continuation in
            let task=Task {
                do {
                    try ensureLoaded()
                    if let language,!language.isEmpty { throw MossVoiceGeneratorError.invalidInput("use generateStreaming with a voice instruction") }
                    _=try await generateStreaming(text:text,instruction:"Speak naturally and clearly.") {continuation.yield($0)}
                    continuation.finish()
                } catch {continuation.finish(throwing:error)}
            }
            continuation.onTermination={@Sendable _ in task.cancel()}
        }
    }

    public func generate(text:String,instruction:String,options:MossVoiceGeneratorOptions = .init())async throws->MossVoiceGeneratorResult {
        try await generateStreaming(text:text,instruction:instruction,options:options) {_ in}
    }

    /// Chunks use the codec's exact causal state. Cancellation discards request
    /// caches, and a generation limit throws instead of returning truncated speech.
    @discardableResult
    public func generateStreaming(text:String,instruction:String,options:MossVoiceGeneratorOptions = .init(),
        onChunk:(AudioChunk)async throws->Void)async throws->MossVoiceGeneratorResult {
        try ensureLoaded()
        try options.validate()
        let prompt=try MVGPrompt.render(text:text,instruction:instruction,normalize:options.normalizeInputs)
        let acquired=try generationLock.withLock { ()->Bool in
            guard loaded else {throw MossVoiceGeneratorError.unloaded}
            guard !generating else {return false};generating=true;return true
        }
        guard acquired else {throw MossVoiceGeneratorError.invalidInput("this model instance is already generating")}
        defer {finishGeneration()}
        try Task.checkCancellation()
        guard let tokenizer else {throw MossVoiceGeneratorError.unloaded}
        let rawIds=tokenizer.encode(prompt)
        try MVGPrompt.validateTokenCount(rawIds.count)
        let maxSteps=options.maxFrames+16+256
        guard rawIds.count+maxSteps<=configuration.language.maxPositionEmbeddings else {throw MossVoiceGeneratorError.contextExceeded}
        guard rawIds.allSatisfy({$0>=0 && $0<configuration.language.vocabSize}) else {
            throw MossVoiceGeneratorError.invalidInput("prompt tokenizer returned an invalid token")
        }
        let ids=rawIds.map(Int32.init)
        let began=Date.timeIntervalSinceReferenceDate
        let random=MLXRandom.RandomState(seed:options.seed)
        var state=network.initialState(),codecState=codec.initialState()
        let padding=Array(repeating:Int32(configuration.audioPadCode),count:16)
        var input=MLXArray(ids.flatMap {[$0]+padding}).reshaped(1,ids.count,17)
        var pattern=MVGDelayPattern(),histories=Array(repeating:Set<Int32>([1024]),count:16)
        var pending:[[Int32]]=[],samples:[Float]=[],textTokens:[Int32]=[]
        var firstToken:Double?,firstAudio:Double?,emittedFrames=0

        for _ in 0..<maxSteps {
            try Task.checkCancellation()
            try ensureLoaded()
            let (logits,next)=network.forward(input,state)
            state=next
            let nextText:Int
            if let forced=pattern.forcedText {nextText=forced}
            else if pattern.isAudio {
                let choices=MLXArray([Int32(MVGDelayPattern.gen),Int32(MVGDelayPattern.delay)])
                let selected=try MVGSampling.sample(logits[0].reshaped([-1])[choices],temperature:options.textTemperature,
                    topP:options.textTopP,topK:options.topK,random:random)
                nextText=[MVGDelayPattern.gen,MVGDelayPattern.delay][selected]
            } else {
                let values=logits[0].reshaped([-1])
                var excluded=[151643,MVGDelayPattern.gen,MVGDelayPattern.delay,MVGDelayPattern.end]
                if pattern.step<=16 {excluded.append(MVGDelayPattern.imEnd)}
                values[MLXArray(excluded.map(Int32.init))]=MLXArray(-Float.infinity).asType(values.dtype)
                nextText=try MVGSampling.sample(values,temperature:options.textTemperature,
                    topP:options.textTopP,topK:options.topK,random:random)
            }
            var audio=padding
            // The pinned upstream generator flattens heads 1...15 before its
            // 2-D penalty helper, so those heads share the union of past IDs.
            // Head zero is sampled separately with its own history.
            let remainingHistory=histories.dropFirst().reduce(into:Set<Int32>()) {$0.formUnion($1)}
            for index in pattern.activeCodebooks {
                let values=logits[index+1].reshaped([-1])
                values[1024]=MLXArray(-Float.infinity).asType(values.dtype)
                audio[index]=Int32(try MVGSampling.sample(values,temperature:options.audioTemperature,
                    topP:options.audioTopP,topK:options.topK,history:index==0 ? histories[0] : remainingHistory,
                    penalty:options.repetitionPenalty,random:random))
            }
            for index in audio.indices {histories[index].insert(audio[index])}
            if firstToken==nil {firstToken=Date.timeIntervalSinceReferenceDate-began}
            if let frame=try pattern.append(text:nextText,audio:audio) {pending.append(frame)}
            guard pattern.firstCodebookFrames<=options.maxFrames else {throw MossVoiceGeneratorError.generationLimit}
            textTokens.append(Int32(nextText))
            if pending.count>=options.chunkFrames || ((pattern.stopped || nextText==MVGDelayPattern.end) && !pending.isEmpty) {
                let chunk=try codec.decode(pending,state:&codecState)
                try Task.checkCancellation()
                try ensureLoaded()
                let elapsed=Date.timeIntervalSinceReferenceDate-began
                if firstAudio==nil {firstAudio=elapsed}
                samples+=chunk
                try await onChunk(AudioChunk(samples:chunk,sampleRate:sampleRate,frameIndex:emittedFrames,
                    isFinal:false,elapsedTime:elapsed))
                emittedFrames+=pending.count;pending=[]
            }
            if nextText==MVGDelayPattern.end {codecState=codec.initialState()}
            if pattern.stopped {break}
            input=MLXArray([Int32(nextText)]+audio).reshaped(1,1,17)
            // Materialize the cache before the next iteration to bound lazy graphs.
            for pair in state.layers.compactMap({$0}) {eval(pair.0,pair.1)}
        }
        try pattern.validateComplete()
        try Task.checkCancellation()
        try ensureLoaded()
        guard pending.isEmpty,!samples.isEmpty,samples.count==pattern.frames.count*1920,
              samples.allSatisfy(\.isFinite) else {throw MossVoiceGeneratorError.invalidOutput("audio sample count or values differ from complete codes")}
        let energy=samples.reduce(Double(0)) {$0+Double($1)*Double($1)}/Double(samples.count)
        guard energy>1e-12 else {throw MossVoiceGeneratorError.invalidOutput("synthesis produced silent audio")}
        // Exactly one terminal event, after the completed schedule and waveform
        // checks. PCM chunks never claim completion before those checks pass.
        try await onChunk(AudioChunk(samples:[],sampleRate:sampleRate,frameIndex:emittedFrames,
            isFinal:true,elapsedTime:Date.timeIntervalSinceReferenceDate-began,textTokens:textTokens))
        try Task.checkCancellation()
        try ensureLoaded()
        return MossVoiceGeneratorResult(samples:samples,codes:pattern.frames,sampleRate:sampleRate,
            generationSeconds:Date.timeIntervalSinceReferenceDate-began,firstTokenSeconds:firstToken ?? 0,
            firstAudioSeconds:firstAudio ?? 0)
    }
}
