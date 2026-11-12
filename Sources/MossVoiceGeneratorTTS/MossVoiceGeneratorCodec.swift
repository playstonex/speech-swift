// Adapted from MOSS-VoiceGenerator and MOSS-Audio-Tokenizer.
// Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
// Licensed under Apache 2.0; see LICENSE.
// Native Swift adaptation uses request-local state and bounded causal caches.

import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon

struct MVGCodecConfiguration: Decodable {
    struct Stage: Decodable {
        let moduleType:String
        let patchSize:Int?
        let inputDimension:Int?
        let outputDimension:Int?
        let dimension:Int?
        let feedforward:Int?
        let heads:Int?
        let layers:Int?
        let maxPeriod:Float?
        let causal:Bool?
        let convolutionLayout:Bool?
        let normalization:String?
        let gating:String?
        let positionalEmbedding:String?
        let layerScale:Float?
        enum CodingKeys:String,CodingKey {
            case moduleType="module_type",patchSize="patch_size",inputDimension="input_dimension"
            case outputDimension="output_dimension",dimension="d_model",feedforward="dim_feedforward"
            case heads="num_heads",layers="num_layers",maxPeriod="max_period"
            case causal,convolutionLayout="conv_layout",normalization="norm",gating
            case positionalEmbedding="positional_embedding"
            case layerScale="layer_scale"
        }
    }
    let stages:[Stage]
    let sampleRate:Int
    let samplesPerFrame:Int
    let contextSeconds:Float
    enum CodingKeys:String,CodingKey {
        case stages="decoder_kwargs",sampleRate="sampling_rate",samplesPerFrame="downsample_rate"
        case contextSeconds="causal_transformer_context_duration"
    }
    func validate() throws {
        guard sampleRate==24000,samplesPerFrame==1920,stages.count==8,contextSeconds==10 else {
            throw MossVoiceGeneratorError.invalidConfiguration("unsupported audio tokenizer format")
        }
        var channels=768,product=1
        for (index,stage) in stages.enumerated() {
            if stage.moduleType=="PatchedPretransform" {
                let expected=index==7 ? 240 : 2
                guard index%2==1,let patch=stage.patchSize,patch==expected,channels%patch==0 else { throw MossVoiceGeneratorError.invalidConfiguration("invalid codec patch stage") }
                channels/=patch;product*=patch
            } else {
                guard stage.moduleType=="Transformer",let input=stage.inputDimension,input==channels,
                      let output=stage.outputDimension,let dim=stage.dimension,let heads=stage.heads,
                      dim>0,heads>0,dim%heads==0,let layers=stage.layers,layers>0,
                      let ff=stage.feedforward,ff>0,stage.causal==true,stage.convolutionLayout==true,
                      stage.normalization=="layer_norm",stage.gating=="none",stage.positionalEmbedding=="rope",
                      stage.layerScale?.isFinite==true,
                      stage.maxPeriod==10000,index%2==0,
                      dim==(index==0 ? 1280 : 768),heads==(index==0 ? 20 : 12),
                      layers==(index==0 ? 32 : 12),ff==4*dim,
                      output==(index==0 ? 1280 : index==6 ? 240 : 768) else { throw MossVoiceGeneratorError.invalidConfiguration("invalid codec transformer stage") }
                channels=output
            }
        }
        guard product==1920,channels==1 else { throw MossVoiceGeneratorError.invalidConfiguration("codec is not 24 kHz mono") }
    }
}

struct MVGCodecCache {
    var kv:[(MLXArray,MLXArray)?]
    var position:Int=0
}
struct MVGCodecState { var stages:[MVGCodecCache] }

final class MVGCodecScale:Module {
    @ParameterInfo var scale:MLXArray
    init(_ dim:Int) { _scale.wrappedValue=MLXArray.ones([dim]);super.init() }
    func callAsFunction(_ x:MLXArray)->MLXArray { x*scale }
}

final class MVGCodecAttention:Module {
    @ModuleInfo(key:"in_projs") var input:[Linear]
    @ModuleInfo(key:"out_projs") var output:[Linear]
    let heads,headDim,dimension,context:Int
    let rope:RoPE
    init(dimension:Int,heads:Int,context:Int,period:Float) {
        self.dimension=dimension;self.heads=heads;headDim=dimension/heads;self.context=context
        _input.wrappedValue=[Linear(dimension,3*dimension,bias:false)]
        _output.wrappedValue=[Linear(dimension,dimension,bias:false)]
        rope=RoPE(dimensions:headDim,traditional:true,base:period)
        super.init()
    }
    func callAsFunction(_ x:MLXArray,cache:(MLXArray,MLXArray)?,position:Int)->(MLXArray,(MLXArray,MLXArray)) {
        let b=x.dim(0),t=x.dim(1)
        let projected=input[0](x).reshaped(b,t,3,heads,headDim)
        let q=rope(projected[0...,0...,0,0...,0...].transposed(0,2,1,3),offset:position)
        var k=rope(projected[0...,0...,1,0...,0...].transposed(0,2,1,3),offset:position)
        var v=projected[0...,0...,2,0...,0...].transposed(0,2,1,3)
        let previous=cache?.0.dim(2) ?? 0
        if let cache { k=concatenated([cache.0,k],axis:2);v=concatenated([cache.1,v],axis:2) }
        let queryPosition=MLXArray(0..<Int32(t)).expandedDimensions(axis:1)+Int32(previous)
        let keyPosition=MLXArray(0..<Int32(k.dim(2))).expandedDimensions(axis:0)
        let delta=queryPosition-keyPosition
        let allowed=logicalAnd(delta .>= Int32(0),delta .< Int32(context))
        let additive=MLX.where(allowed,MLXArray(Float(0)),MLXArray(-Float.infinity))
        let attended=SDPA.attendAndMerge(qHeads:q,kHeads:k,vHeads:v,scale:1/sqrt(Float(headDim)),
            mask:MLXFast.ScaledDotProductAttentionMaskMode.array(additive.reshaped(1,1,t,k.dim(2)).asType(q.dtype)))
        let start=max(0,k.dim(2)-context)
        return(output[0](attended),(k[0...,0...,start...,0...],v[0...,0...,start...,0...]))
    }
}

final class MVGCodecLayer:Module {
    @ModuleInfo(key:"self_attn") var attention:MVGCodecAttention
    @ModuleInfo var norm1:LayerNorm
    @ModuleInfo var norm2:LayerNorm
    @ModuleInfo var linear1:Linear
    @ModuleInfo var linear2:Linear
    @ModuleInfo(key:"layer_scale_1") var scale1:MVGCodecScale
    @ModuleInfo(key:"layer_scale_2") var scale2:MVGCodecScale
    init(dimension:Int,heads:Int,feedforward:Int,context:Int,period:Float) {
        _attention.wrappedValue=MVGCodecAttention(dimension:dimension,heads:heads,context:context,period:period)
        _norm1.wrappedValue=LayerNorm(dimensions:dimension,eps:1e-5)
        _norm2.wrappedValue=LayerNorm(dimensions:dimension,eps:1e-5)
        _linear1.wrappedValue=Linear(dimension,feedforward,bias:false)
        _linear2.wrappedValue=Linear(feedforward,dimension,bias:false)
        _scale1.wrappedValue=MVGCodecScale(dimension);_scale2.wrappedValue=MVGCodecScale(dimension)
        super.init()
    }
    func callAsFunction(_ x:MLXArray,cache:(MLXArray,MLXArray)?,position:Int)->(MLXArray,(MLXArray,MLXArray)) {
        let (a,next)=attention(norm1(x),cache:cache,position:position)
        let h=x+scale1(a)
        return(h+scale2(linear2(gelu(linear1(norm2(h))))),next)
    }
}

final class MVGCodecTransformer:Module {
    @ModuleInfo var layers:[MVGCodecLayer]
    init(_ c:MVGCodecConfiguration.Stage,context:Int) {
        _layers.wrappedValue=(0..<c.layers!).map {_ in MVGCodecLayer(dimension:c.dimension!,heads:c.heads!,
            feedforward:c.feedforward!,context:context,period:c.maxPeriod ?? 10000)}
        super.init()
    }
    func callAsFunction(_ x:MLXArray,state:inout MVGCodecCache)throws->MLXArray {
        var h=x
        for i in layers.indices {
            try Task.checkCancellation()
            let (value,next)=layers[i](h,cache:state.kv[i],position:state.position)
            h=value;state.kv[i]=next
            eval(h,next.0,next.1)
        }
        state.position+=x.dim(1)
        return h
    }
}

class MVGCodecStage:Module {
    var layerCount:Int { 0 }
    func decode(_ x:MLXArray,state:inout MVGCodecCache)throws->MLXArray { x }
}

final class MVGCodecProjected:MVGCodecStage {
    @ModuleInfo(key:"input_proj") var input:Linear?
    @ModuleInfo var transformer:MVGCodecTransformer
    @ModuleInfo(key:"output_proj") var output:Linear?
    override var layerCount:Int { transformer.layers.count }
    init(_ c:MVGCodecConfiguration.Stage,context:Int) {
        _input.wrappedValue=c.inputDimension==c.dimension ? nil : Linear(c.inputDimension!,c.dimension!,bias:false)
        _output.wrappedValue=c.outputDimension==c.dimension ? nil : Linear(c.dimension!,c.outputDimension!,bias:false)
        _transformer.wrappedValue=MVGCodecTransformer(c,context:context)
        super.init()
    }
    override func decode(_ x:MLXArray,state:inout MVGCodecCache)throws->MLXArray {
        let h=try transformer(input?(x) ?? x,state:&state)
        return output?(h) ?? h
    }
}

final class MVGCodecPatch:MVGCodecStage {
    let patch:Int
    init(_ patch:Int) { self.patch=patch;super.init() }
    /// Upstream decodes NCL [B,D*patch,T] by separating D and patch before time.
    override func decode(_ x:MLXArray,state:inout MVGCodecCache)throws->MLXArray {
        let b=x.dim(0),t=x.dim(1),d=x.dim(2)/patch
        return x.reshaped(b,t,d,patch).transposed(0,1,3,2).reshaped(b,t*patch,d)
    }
}

final class MVGCodecLFQ:Module {
    @ModuleInfo var codebook:Embedding
    @ModuleInfo(key:"out_proj") var output:Linear
    override init() {
        _codebook.wrappedValue=Embedding(embeddingCount:1024,dimensions:8)
        _output.wrappedValue=Linear(8,512,bias:true);super.init()
    }
    func callAsFunction(_ ids:MLXArray)->MLXArray { output(codebook(ids)) }
}
final class MVGCodecQuantizer:Module {
    @ModuleInfo var quantizers:[MVGCodecLFQ]
    @ModuleInfo(key:"output_proj") var output:Linear
    override init() {
        _quantizers.wrappedValue=(0..<16).map {_ in MVGCodecLFQ()}
        _output.wrappedValue=Linear(512,768,bias:true);super.init()
    }
    func callAsFunction(_ codes:MLXArray)->MLXArray {
        var sum=quantizers[0](codes[0...,0...,0])
        for i in 1..<16 { sum=sum+quantizers[i](codes[0...,0...,i]) }
        return output(sum)
    }
}

final class MVGCodec:Module {
    @ModuleInfo var quantizer:MVGCodecQuantizer
    @ModuleInfo var decoder:[MVGCodecStage]
    init(_ config:MVGCodecConfiguration) {
        _quantizer.wrappedValue=MVGCodecQuantizer()
        var rate=Float(config.sampleRate)/Float(config.samplesPerFrame)
        _decoder.wrappedValue=config.stages.map { stage in
            if stage.moduleType=="PatchedPretransform" { rate*=Float(stage.patchSize!);return MVGCodecPatch(stage.patchSize!) }
            return MVGCodecProjected(stage,context:Int(rate*config.contextSeconds))
        }
        super.init()
    }
    func initialState()->MVGCodecState { MVGCodecState(stages:decoder.map { MVGCodecCache(kv:Array(repeating:nil,count:$0.layerCount)) }) }
    func decode(_ frames:[[Int32]],state:inout MVGCodecState)throws->[Float] {
        guard !frames.isEmpty,frames.allSatisfy({$0.count==16 && $0.allSatisfy({(0..<1024).contains($0)})}) else {
            throw MossVoiceGeneratorError.invalidOutput("codec requires nonempty complete sixteen-codebook frames")
        }
        let ids=MLXArray(frames.flatMap {$0}).reshaped(1,frames.count,16)
        var h=quantizer(ids)
        for i in decoder.indices { h=try decoder[i].decode(h,state:&state.stages[i]) }
        let values=h.reshaped([-1]).asArray(Float.self)
        guard values.count==frames.count*1920,values.allSatisfy(\.isFinite) else {
            throw MossVoiceGeneratorError.invalidOutput("codec output length or values invalid")
        }
        return values
    }
    func load(from file:URL)throws {
        let raw=try MLX.loadArrays(url:file)
        guard !raw.isEmpty,raw.values.allSatisfy({$0.dtype == .float32}) else {
            throw MossVoiceGeneratorError.invalidConfiguration("codec tensors must retain upstream FP32 precision")
        }
        var weights=raw.filter {!$0.key.contains(".parametrizations.weight.")}
        let prefixes=["quantizer.output_proj"]+(0..<16).map {"quantizer.quantizers.\($0).out_proj"}
        for prefix in prefixes {
            guard let g=raw[prefix+".parametrizations.weight.original0"],let v=raw[prefix+".parametrizations.weight.original1"],
                  g.ndim==3,v.ndim==3,v.dim(2)==1,g.shape==[v.dim(0),1,1] else {
                throw MossVoiceGeneratorError.invalidConfiguration("missing weight-normalized codec projection")
            }
            let norm=sqrt(sum(v*v,axes:[1,2],keepDims:true))
            guard g.asArray(Float.self).allSatisfy(\.isFinite),
                  norm.asArray(Float.self).allSatisfy({$0.isFinite && $0>0}) else {
                throw MossVoiceGeneratorError.invalidConfiguration("invalid codec weight-normalization values")
            }
            weights[prefix+".weight"]=(v*(g/norm)).squeezed(axis:2)
        }
        try update(parameters:ModuleParameters.unflattened(weights),verify:.all)
        eval(parameters())
    }
}
