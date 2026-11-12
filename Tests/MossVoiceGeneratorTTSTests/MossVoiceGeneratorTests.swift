import Foundation
import XCTest
@testable import MossVoiceGeneratorTTS

final class MossVoiceGeneratorPromptTests:XCTestCase {
    func testPinnedEnglishAndChineseRawAndNormalizedPrompts()throws {
        let url=try XCTUnwrap(Bundle.module.url(forResource:"prompt-tokenizer",withExtension:"json",subdirectory:"Fixtures"))
        let root=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any])
        let cases=try XCTUnwrap(root["cases"] as? [[String:Any]])
        XCTAssertEqual(cases.count,4)
        for entry in cases {
            let text=try XCTUnwrap(entry["text"] as? String),instruction=try XCTUnwrap(entry["instruction"] as? String)
            let normalize=try XCTUnwrap(entry["normalize"] as? Bool)
            XCTAssertEqual(try MVGPrompt.render(text:text,instruction:instruction,normalize:normalize),
                try XCTUnwrap(entry["expected_prompt"] as? String),entry["id"] as? String ?? "case")
        }
    }
    func testExactReferenceFreePrompt()throws {
        let prompt=try MVGPrompt.render(text:"Hello!",instruction:"A warm, quiet voice.")
        XCTAssertEqual(prompt,"<|im_start|>user\n<user_inst>\n- Reference(s):\nNone\n- Instruction:\nA warm, quiet voice.\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\nHello!\n</user_inst><|im_end|>\n<|im_start|>assistant\n")
    }
    func testRawInputIsPreservedByDefault()throws {
        let prompt=try MVGPrompt.render(text:"[laugh] \"Hi\" — bye.",instruction:"High-pitched\nthen_low")
        XCTAssertTrue(prompt.contains("[laugh] \"Hi\" — bye."))
        XCTAssertTrue(prompt.contains("High-pitched\nthen_low"))
    }
    func testOptionalNormalizerMatchesRepeatedPythonCapture() {
        XCTAssertEqual(MVGPrompt.cleanup(" Hi[laugh]!!?\n{meta}\"there\" ",instruction:false),"Hi? there")
        XCTAssertEqual(MVGPrompt.cleanup("女声, warm_low-pitch [tag]",instruction:true),"女声， warm，low，pitch")
        XCTAssertEqual(MVGPrompt.cleanup("a-b_c",instruction:false),"a-b_c")
    }
    func testRejectsReservedTokensAndEmptyFields() {
        XCTAssertThrowsError(try MVGPrompt.render(text:"<|audio_start|>",instruction:"quiet"))
        XCTAssertThrowsError(try MVGPrompt.render(text:"hello",instruction:"</user_inst>"))
        XCTAssertThrowsError(try MVGPrompt.render(text:" ",instruction:"quiet"))
        XCTAssertThrowsError(try MVGPrompt.render(text:"[tag]",instruction:"quiet",normalize:true))
    }
    func testFieldByteLimitPreservesBoundaryAndRejectsBothFields()throws {
        let boundary=String(repeating:"a",count:32768)
        XCTAssertTrue(try MVGPrompt.render(text:boundary,instruction:"quiet").contains(boundary))
        XCTAssertTrue(try MVGPrompt.render(text:"hello",instruction:boundary).contains(boundary))
        for value in [boundary+"a",String(repeating:" ",count:32769)] {
            XCTAssertThrowsError(try MVGPrompt.render(text:value,instruction:"quiet"))
            XCTAssertThrowsError(try MVGPrompt.render(text:"hello",instruction:value))
        }
    }
    func testUnicodeLimitCountsUTF8BytesRatherThanCharacters()throws {
        let boundary=String(repeating:"你",count:10922)+"ab"
        XCTAssertEqual(boundary.utf8.count,32768)
        XCTAssertTrue(try MVGPrompt.render(text:boundary,instruction:"quiet").contains(boundary))
        let oversized=String(repeating:"你",count:10923)
        XCTAssertThrowsError(try MVGPrompt.render(text:oversized,instruction:"quiet"))
        XCTAssertThrowsError(try MVGPrompt.render(text:"hello",instruction:oversized))
    }
    func testNormalizationCannotBypassRawFieldLimit() {
        let oversized=String(repeating:"[x]",count:11000)+"hello"
        XCTAssertThrowsError(try MVGPrompt.render(text:oversized,instruction:"quiet",normalize:true))
        XCTAssertThrowsError(try MVGPrompt.render(text:"hello",instruction:oversized,normalize:true))
    }
    func testRenderedPromptTokenLimitBeforeTensorAllocation()throws {
        try MVGPrompt.validateTokenCount(1)
        try MVGPrompt.validateTokenCount(4096)
        for count in [-1,0,4097,40960,Int.max] {
            XCTAssertThrowsError(try MVGPrompt.validateTokenCount(count))
        }
    }
}

final class MossVoiceGeneratorDelayPatternTests:XCTestCase {
    private func renderFrames(_ count:Int)throws->MVGDelayPattern {
        var state=MVGDelayPattern()
        let pad=Array(repeating:Int32(1024),count:16)
        XCTAssertNil(try state.append(text:MVGDelayPattern.start,audio:pad))
        for row in 0..<(count+15) {
            let active=Set(state.activeCodebooks)
            let values=(0..<16).map {book in active.contains(book) ? Int32((row-book)*16+book) : Int32(1024)}
            let text=row<count-1 ? MVGDelayPattern.gen : MVGDelayPattern.delay
            _=try state.append(text:text,audio:values)
        }
        XCTAssertEqual(state.forcedText,MVGDelayPattern.end)
        _=try state.append(text:MVGDelayPattern.end,audio:pad)
        _=try state.append(text:MVGDelayPattern.imEnd,audio:pad)
        try state.validateComplete()
        return state
    }
    func testSixteenCodebooksAreUndelayedWithoutTrailingPads()throws {
        let state=try renderFrames(3)
        XCTAssertEqual(state.frames,(0..<3).map {frame in (0..<16).map {Int32(frame*16+$0)}})
        XCTAssertEqual(state.firstCodebookFrames,3)
        XCTAssertEqual(state.segments,1)
    }
    func testShortestOneFrameSegmentDrainsAllSixteenBooks()throws {
        let state=try renderFrames(1)
        XCTAssertEqual(state.frames.count,1)
        XCTAssertEqual(state.frames[0],(0..<16).map(Int32.init))
        XCTAssertTrue(state.stopped)
    }
    func testPrematureEndAndInvalidPaddingAreRejected() {
        var state=MVGDelayPattern()
        XCTAssertThrowsError(try state.append(text:MVGDelayPattern.imEnd,audio:Array(repeating:1024,count:16)))
        XCTAssertThrowsError(try state.append(text:MVGDelayPattern.start,audio:Array(repeating:0,count:16)))
        XCTAssertThrowsError(try state.validateComplete())
    }
    func testIncompleteTailCannotBeAcceptedAsFinishedSpeech()throws {
        var state=MVGDelayPattern()
        _=try state.append(text:MVGDelayPattern.start,audio:Array(repeating:1024,count:16))
        _=try state.append(text:MVGDelayPattern.delay,audio:[0]+Array(repeating:1024,count:15))
        XCTAssertEqual(state.forcedText,MVGDelayPattern.delay)
        XCTAssertEqual(state.activeCodebooks,[1])
        XCTAssertThrowsError(try state.validateComplete())
        XCTAssertThrowsError(try state.append(text:MVGDelayPattern.end,audio:Array(repeating:1024,count:16)))
    }
}

final class MossVoiceGeneratorConfigurationTests:XCTestCase {
    private func json(_ name:String)throws->[String:Any] {
        let url=try XCTUnwrap(Bundle.module.url(forResource:name,withExtension:"json",subdirectory:"Fixtures"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any])
    }
    func testPinnedModelAndCodecConfigurations()throws {
        let config=try JSONDecoder().decode(MossVoiceGeneratorConfiguration.self,
            from:JSONSerialization.data(withJSONObject:json("model-config")))
        try config.validate()
        XCTAssertEqual(config.codebooks,16)
        XCTAssertEqual(config.language.numHiddenLayers,28)
        let codec=try JSONDecoder().decode(MVGCodecConfiguration.self,
            from:JSONSerialization.data(withJSONObject:json("codec-config")))
        try codec.validate()
        XCTAssertEqual(codec.stages.count,8)
    }
    func testWrongArchitectureAndControlIdsAreRejectedBeforeAllocation()throws {
        for (key,value) in [("audio_start_token_id",42),("n_vq",8)] {
            var raw=try json("model-config");raw[key]=value
            let config=try JSONDecoder().decode(MossVoiceGeneratorConfiguration.self,
                from:JSONSerialization.data(withJSONObject:raw))
            XCTAssertThrowsError(try config.validate())
        }
        for key in ["hidden_size","intermediate_size","vocab_size","max_position_embeddings"] {
            var raw=try json("model-config")
            var language=try XCTUnwrap(raw["language_config"] as? [String:Any]);language[key]=0;raw["language_config"]=language
            let config=try JSONDecoder().decode(MossVoiceGeneratorConfiguration.self,
                from:JSONSerialization.data(withJSONObject:raw))
            XCTAssertThrowsError(try config.validate())
        }
    }
    func testUnsupportedCodecGraphsAreRejectedBeforeAllocation()throws {
        for (key,value) in [("causal",false as Any),("norm","rms_norm" as Any),
                            ("gating","silu" as Any),("positional_embedding","sin" as Any)] {
            var raw=try json("codec-config")
            var stages=try XCTUnwrap(raw["decoder_kwargs"] as? [[String:Any]])
            stages[0][key]=value;raw["decoder_kwargs"]=stages
            let codec=try JSONDecoder().decode(MVGCodecConfiguration.self,
                from:JSONSerialization.data(withJSONObject:raw))
            XCTAssertThrowsError(try codec.validate())
        }
    }
    func testOptionsRejectUnsafeOrUndefinedSettings()throws {
        try MossVoiceGeneratorOptions().validate()
        var value=MossVoiceGeneratorOptions()
        value.maxFrames=0;XCTAssertThrowsError(try value.validate())
        value = .init();value.chunkFrames=0;XCTAssertThrowsError(try value.validate())
        value = .init();value.audioTopP=0;XCTAssertThrowsError(try value.validate())
        value = .init();value.audioTemperature = .nan;XCTAssertThrowsError(try value.validate())
        value = .init(textTemperature:0,audioTemperature:0);try value.validate()
    }
    func testMissingBundleFailsBeforeModelAllocation()async throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {_=try await MossVoiceGeneratorModel.fromBundle(directory);XCTFail("missing bundle accepted")}
        catch let error as MossVoiceGeneratorError {
            guard case .invalidConfiguration(let message)=error else {return XCTFail("wrong failure")}
            XCTAssertTrue(message.contains("config.json"))
        }
    }
}
