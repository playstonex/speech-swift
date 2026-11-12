import Foundation
import MLX
import MLXNN
import MLXCommon
import XCTest
@testable import MossVoiceGeneratorTTS

/// Numerical checks allocate only toy arrays, but require the MLX metallib.
final class E2EMossVoiceGeneratorMathTests:XCTestCase {
    private func fixture(_ name:String="codec-tiny")throws->[String:Any] {
        let url=try XCTUnwrap(Bundle.module.url(forResource:name,withExtension:"json",subdirectory:"Fixtures"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any])
    }
    private func array(_ item:Any)throws->MLXArray {
        func shape(_ value:Any)->[Int] {
            guard let values=value as? [Any] else {return []}
            return [values.count]+(values.first.map {shape($0)} ?? [])
        }
        func flatten(_ value:Any)throws->[Float] {
            if let values=value as? [Any] {return try values.flatMap {try flatten($0)}}
            return [try XCTUnwrap(value as? NSNumber).floatValue]
        }
        return try MLXArray(flatten(item)).reshaped(shape(item))
    }
    private func close(_ actual:MLXArray,_ expected:MLXArray,absolute:Float=2e-6,
                       file:StaticString=#filePath,line:UInt=#line) {
        XCTAssertEqual(actual.shape,expected.shape,file:file,line:line)
        let difference=max(abs(actual-expected)).item(Float.self)
        XCTAssertLessThanOrEqual(difference,absolute,file:file,line:line)
    }
    func testCodecMatchesPinnedUpstreamOfflineAndCachedChunks()throws {
        let f=try fixture(),t=try XCTUnwrap(f["transformer"] as? [String:Any])
        var raw=try XCTUnwrap(t["config"] as? [String:Any])
        raw["module_type"]="Transformer";raw["input_dimension"]=8;raw["output_dimension"]=8
        let config=try JSONDecoder().decode(MVGCodecConfiguration.Stage.self,
            from:JSONSerialization.data(withJSONObject:raw))
        let model=withThrowawayRandomState {MVGCodecTransformer(config,context:3)}
        let source=try XCTUnwrap(t["weights"] as? [String:Any])
        let weights=try source.mapValues {try array($0)}
        try model.update(parameters:ModuleParameters.unflattened(weights),verify:.all)
        let x=try array(XCTUnwrap(t["input"]))
        let expected=try array(XCTUnwrap(t["expected_offline"]))
        var full=MVGCodecCache(kv:[nil])
        close(try model(x,state:&full),expected)
        for sizes in [[1,1,1,1,1,1,1,1,1],[2,2,2,3]] {
            var state=MVGCodecCache(kv:[nil]),position=0,parts:[MLXArray]=[]
            for count in sizes {
                parts.append(try model(x[0...,position..<(position+count),0...],state:&state))
                position+=count
            }
            close(concatenated(parts,axis:1),expected)
            XCTAssertEqual(state.position,9)
            XCTAssertEqual(state.kv[0]?.0.dim(2),3)
        }
    }
    func testCodecUsesAdjacentPairRoPEAtAbsoluteOffset()throws {
        let f=try fixture(),r=try XCTUnwrap(f["rope"] as? [String:Any])
        let rope=RoPE(dimensions:4,traditional:true,base:10000)
        close(rope(try array(XCTUnwrap(r["input_q"])),offset:5),try array(XCTUnwrap(r["expected_q"])))
        close(rope(try array(XCTUnwrap(r["input_k"])),offset:5),try array(XCTUnwrap(r["expected_k"])))
    }
    func testPatchDecodePreservesSourceChannelOrdering()throws {
        let f=try fixture(),p=try XCTUnwrap(f["patch_decode"] as? [String:Any])
        let patch=MVGCodecPatch(3)
        var state=MVGCodecCache(kv:[])
        close(try patch.decode(array(XCTUnwrap(p["input"])),state:&state),
            try array(XCTUnwrap(p["expected_output"])),absolute:0)
    }
    func testSignedWeightNormalizedProjectionMatchesTorch()throws {
        let f=try fixture(),p=try XCTUnwrap(f["weight_normalized_conv1x1"] as? [String:Any])
        let g=try array(XCTUnwrap(p["g"])),v=try array(XCTUnwrap(p["v"]))
        let folded=(v*(g/sqrt(sum(v*v,axes:[1,2],keepDims:true)))).squeezed(axis:2)
        close(folded,try array(XCTUnwrap(p["expected_folded_linear_weight"])))
        let x=try array(XCTUnwrap(p["input"])),bias=try array(XCTUnwrap(p["bias"]))
        close(matmul(x,folded.T)+bias,try array(XCTUnwrap(p["expected_output"])))
    }
    func testSamplingUsesLocalRandomStateAndGreedyPenaltyOrder()throws {
        let logits=MLXArray([Float(4),3,-1,-2])
        let random=MLXRandom.RandomState(seed:42)
        XCTAssertEqual(try MVGSampling.sample(logits,temperature:0,topP:0.5,topK:1,
            history:[0],penalty:2,random:random),1)
        let a=MLXRandom.RandomState(seed:17),b=MLXRandom.RandomState(seed:17)
        let first=try (0..<20).map {_ in try MVGSampling.sample(logits,temperature:1,topP:0.9,topK:3,random:a)}
        let second=try (0..<20).map {_ in try MVGSampling.sample(logits,temperature:1,topP:0.9,topK:3,random:b)}
        XCTAssertEqual(first,second)
    }
    func testLanguageModelMatchesQwen3WithAllPadEmbeddingsAndUntiedHeads()throws {
        let f=try fixture("qwen-tiny")
        var raw=try XCTUnwrap(f["audio_config"] as? [String:Any])
        raw["language_config"]=f["config"];raw["model_type"]="moss_tts_delay";raw["sampling_rate"]=24000
        raw["audio_start_token_id"]=151652;raw["audio_end_token_id"]=151653
        raw["audio_assistant_gen_slot_token_id"]=151656;raw["audio_assistant_delay_slot_token_id"]=151662
        let config=try JSONDecoder().decode(MossVoiceGeneratorConfiguration.self,
            from:JSONSerialization.data(withJSONObject:raw))
        let input=try array(XCTUnwrap(f["input_ids"])).asType(.int32)
        let modes=try XCTUnwrap(f["modes"] as? [String:[String:Any]])
        for precision in ["float32","bfloat16"] {
            let mode=try XCTUnwrap(modes[precision])
            let tolerance:Float=precision=="float32" ? 2e-6 : 0.02
            let dtype:DType=precision=="float32" ? .float32 : .bfloat16
            let model=withThrowawayRandomState {MVGLanguageModel(config)}
            let source=try XCTUnwrap(mode["weights"] as? [String:Any])
            let weights=try source.mapValues {try array($0).asType(dtype)}
            try model.update(parameters:ModuleParameters.unflattened(weights),verify:.all)
            close(model.embedded(input).asType(.float32),try array(XCTUnwrap(mode["embedded_input"])),absolute:tolerance)
            let (whole,wholeState)=model.forward(input,model.initialState())
            let expected=try XCTUnwrap(mode["expected_full_last_logits"] as? [Any])
            XCTAssertEqual(whole.count,17)
            for i in whole.indices {close(whole[i].asType(.float32),try array(expected[i]),absolute:tolerance)}
            let (_,prefixState)=model.forward(input[0...,0..<4,0...],model.initialState())
            let (cached,cachedState)=model.forward(input[0...,4...,0...],prefixState)
            for i in whole.indices {close(cached[i].asType(.float32),try array(expected[i]),absolute:tolerance)}
            XCTAssertEqual(wholeState.position,6);XCTAssertEqual(cachedState.position,6)
            XCTAssertEqual(cachedState.layers[0]?.0.shape,[1,1,6,4])
        }
    }
}
