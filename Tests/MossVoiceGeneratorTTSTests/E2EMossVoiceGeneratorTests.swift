import AudioCommon
import Foundation
import XCTest
@testable import MossVoiceGeneratorTTS

/// Runs only against an explicitly selected, locally prepared bundle.
/// Content accuracy is checked by transcribing the saved WAV after releasing
/// this model process, so the synthesizer and ASR never contend for memory.
final class E2EMossVoiceGeneratorTests:XCTestCase {
    private func bundle()throws->URL {
        guard let path=ProcessInfo.processInfo.environment["MOSS_VOICEGEN_BUNDLE"] else {
            throw XCTSkip("Set MOSS_VOICEGEN_BUNDLE to a converted local BF16 bundle")
        }
        return URL(fileURLWithPath:path)
    }
    func testTokenizerIDsMatchPinnedReference()async throws {
        let env=ProcessInfo.processInfo.environment
        guard let path=env["MOSS_VOICEGEN_TOKENIZER"] ?? env["MOSS_VOICEGEN_BUNDLE"] else {
            throw XCTSkip("Set MOSS_VOICEGEN_TOKENIZER or MOSS_VOICEGEN_BUNDLE")
        }
        let tokenizer=try await MVGTokenizer.load(URL(fileURLWithPath:path))
        let url=try XCTUnwrap(Bundle.module.url(forResource:"prompt-tokenizer",withExtension:"json",subdirectory:"Fixtures"))
        let root=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any])
        for entry in try XCTUnwrap(root["cases"] as? [[String:Any]]) {
            let prompt=try MVGPrompt.render(text:XCTUnwrap(entry["text"] as? String),
                instruction:XCTUnwrap(entry["instruction"] as? String),normalize:XCTUnwrap(entry["normalize"] as? Bool))
            XCTAssertEqual(tokenizer.encode(prompt),try XCTUnwrap(entry["expected_input_ids"] as? [Int]))
        }
    }
    func testSynthesisProducesCompleteFiniteAudioAndContentArtifact()async throws {
        let model=try await MossVoiceGeneratorModel.fromBundle(bundle())
        var rejectedChunks=0
        for (oversizedText,limit) in [(String(repeating:"a",count:32769),"32768"),
                                     (String(repeating:"hello ",count:5000),"4096")] {
            do {
                _=try await model.generateStreaming(text:oversizedText,instruction:"A calm adult voice.") {_ in
                    rejectedChunks+=1
                }
                XCTFail("oversized request was accepted")
            } catch let error as MossVoiceGeneratorError {
                guard case .invalidInput(let reason)=error else {throw error}
                XCTAssertTrue(reason.contains(limit),"expected the \(limit) limit, received \(reason)")
            }
        }
        XCTAssertEqual(rejectedChunks,0,"rejected requests must not emit any audio")
        XCTAssertTrue(model.isLoaded)
        let text="The little boat crossed the quiet lake before sunset."
        let options=MossVoiceGeneratorOptions(maxFrames:150,seed:42)
        var chunks:[AudioChunk]=[]
        let result=try await model.generateStreaming(text:text,
            instruction:"A warm adult female voice speaking clearly at a natural pace.",options:options) {chunks.append($0)}
        XCTAssertEqual(result.sampleRate,24000)
        XCTAssertEqual(result.samples.count,result.codes.count*1920)
        XCTAssertTrue(result.samples.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(result.duration,1)
        XCTAssertLessThan(result.duration,12)
        let rms=sqrt(result.samples.reduce(Double(0)) {$0+Double($1)*Double($1)}/Double(result.samples.count))
        XCTAssertGreaterThan(rms,0.001)
        XCTAssertTrue(chunks.last?.isFinal == true)
        XCTAssertEqual(chunks.filter(\.isFinal).count,1,"stream must contain exactly one terminal chunk")
        XCTAssertTrue(chunks.last?.samples.isEmpty == true)
        XCTAssertEqual(chunks.last?.frameIndex,result.codes.count)
        XCTAssertEqual(chunks.flatMap(\.samples),result.samples)
        XCTAssertGreaterThan(result.firstAudioSeconds,0)
        XCTAssertLessThanOrEqual(result.firstAudioSeconds,result.generationSeconds)
        if let output=ProcessInfo.processInfo.environment["MOSS_VOICEGEN_TEST_OUTPUT"] {
            let file=URL(fileURLWithPath:output)
            try WAVWriter.write(samples:result.samples,sampleRate:result.sampleRate,to:file)
            let report:[String:Any]=["expected_text":text,"audio_frames":result.codes.count,
                "sample_count":result.samples.count,"duration_seconds":result.duration,
                "generation_seconds":result.generationSeconds,"first_token_seconds":result.firstTokenSeconds,
                "first_audio_seconds":result.firstAudioSeconds,"seed":options.seed,"rms":rms]
            try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys])
                .write(to:file.deletingPathExtension().appendingPathExtension("json"))
        }
    }
    func testCancellationBeforeGenerationLeavesInstanceUsable()async throws {
        let model=try await MossVoiceGeneratorModel.fromBundle(bundle())
        let cancelled=Task {
            withUnsafeCurrentTask {$0?.cancel()}
            return try await model.generate(text:"Hello.",instruction:"A calm adult voice.")
        }
        do {_=try await cancelled.value;XCTFail("cancelled generation was accepted")}
        catch is CancellationError {}
        let output=try await model.generate(text:"Good morning.",instruction:"A calm adult voice.",
            options:.init(maxFrames:100,seed:42))
        XCTAssertFalse(output.samples.isEmpty)
    }
    func testCancellationAfterFirstAudioClearsRequestState()async throws {
        let model=try await MossVoiceGeneratorModel.fromBundle(bundle())
        let cancelled=Task {
            try await model.generateStreaming(text:"Good morning. Welcome home.",
                instruction:"A calm adult voice.",options:.init(maxFrames:100,seed:42)) {chunk in
                if !chunk.samples.isEmpty {withUnsafeCurrentTask {$0?.cancel()}}
            }
        }
        do {_=try await cancelled.value;XCTFail("active cancellation was ignored")}
        catch is CancellationError {}
        XCTAssertTrue(model.isLoaded)
        let recovered=try await model.generate(text:"Good morning.",instruction:"A calm adult voice.",
            options:.init(maxFrames:100,seed:42))
        XCTAssertFalse(recovered.samples.isEmpty)
        XCTAssertEqual(recovered.samples.count,recovered.codes.count*1920)
        model.unload()
    }
    func testConcurrentRequestIsRejectedDuringStreaming()async throws {
        let model=try await MossVoiceGeneratorModel.fromBundle(bundle())
        var checked=false
        _=try await model.generateStreaming(text:"Hello, welcome home.",instruction:"A relaxed adult male voice.",
            options:.init(maxFrames:100,seed:42)) {_ in
            if checked {return};checked=true
            do {
                _=try await model.generate(text:"Hello.",instruction:"A calm voice.")
                XCTFail("concurrent request accepted")
            } catch let error as MossVoiceGeneratorError {
                guard case .invalidInput(let reason)=error else {throw error}
                XCTAssertTrue(reason.contains("already generating"))
            }
        }
        XCTAssertTrue(checked)
    }
    func testExplicitUnloadIsIdempotentAndPreventsInference()async throws {
        let model=try await MossVoiceGeneratorModel.fromBundle(bundle())
        XCTAssertTrue(model.isLoaded);XCTAssertGreaterThan(model.memoryFootprint,7_000_000_000)
        model.unload();model.unload()
        XCTAssertFalse(model.isLoaded);XCTAssertEqual(model.memoryFootprint,0)
        do {_=try await model.generate(text:"Hello.",instruction:"A calm voice.");XCTFail("unloaded model accepted")}
        catch let error as MossVoiceGeneratorError {
            guard case .unloaded=error else {return XCTFail("wrong post-unload error")}
        }
    }
    func testUnloadDuringStreamingStopsBeforeWeightsAreReleased()async throws {
        let model=try await MossVoiceGeneratorModel.fromBundle(bundle())
        var unloaded=false
        do {
            _=try await model.generateStreaming(text:"The little boat crossed the quiet lake before sunset.",
                instruction:"A calm adult voice.",options:.init(maxFrames:150)) {_ in
                if !unloaded {model.unload();unloaded=true}
                XCTAssertFalse(model.isLoaded)
            }
            XCTFail("generation completed after unload")
        } catch let error as MossVoiceGeneratorError {
            guard case .unloaded=error else {return XCTFail("wrong unload interruption")}
        }
        XCTAssertTrue(unloaded);XCTAssertEqual(model.memoryFootprint,0)
    }
}
