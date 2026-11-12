// Adapted from MOSS-VoiceGenerator and MOSS-Audio-Tokenizer.
// Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
// Licensed under Apache 2.0; see LICENSE.
// Native Swift adaptation uses request-local state and bounded causal caches.

import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon

struct MVGKVState {
    var layers: [(MLXArray,MLXArray)?]
    var position: Int=0
}

final class MVGAttention: Module {
    @ModuleInfo(key:"q_proj") var q: Linear
    @ModuleInfo(key:"k_proj") var k: Linear
    @ModuleInfo(key:"v_proj") var v: Linear
    @ModuleInfo(key:"o_proj") var o: Linear
    @ModuleInfo(key:"q_norm") var qNorm: RMSNorm
    @ModuleInfo(key:"k_norm") var kNorm: RMSNorm
    let heads,kvHeads,headDim: Int
    let rope: RoPE
    init(_ c:MossVoiceGeneratorConfiguration.Language) {
        heads=c.numAttentionHeads;kvHeads=c.numKeyValueHeads;headDim=c.headDim
        _q.wrappedValue=Linear(c.hiddenSize,heads*headDim,bias:false)
        _k.wrappedValue=Linear(c.hiddenSize,kvHeads*headDim,bias:false)
        _v.wrappedValue=Linear(c.hiddenSize,kvHeads*headDim,bias:false)
        _o.wrappedValue=Linear(heads*headDim,c.hiddenSize,bias:false)
        _qNorm.wrappedValue=RMSNorm(dimensions:headDim,eps:c.rmsNormEps)
        _kNorm.wrappedValue=RMSNorm(dimensions:headDim,eps:c.rmsNormEps)
        rope=RoPE(dimensions:headDim,traditional:false,base:c.ropeTheta)
        super.init()
    }
    func callAsFunction(_ x:MLXArray,_ cache:(MLXArray,MLXArray)?,_ offset:Int)->(MLXArray,(MLXArray,MLXArray)) {
        let b=x.dim(0),t=x.dim(1)
        var queries=qNorm(q(x).reshaped(b,t,heads,headDim)).transposed(0,2,1,3)
        var keys=kNorm(k(x).reshaped(b,t,kvHeads,headDim)).transposed(0,2,1,3)
        var values=v(x).reshaped(b,t,kvHeads,headDim).transposed(0,2,1,3)
        queries=rope(queries,offset:offset);keys=rope(keys,offset:offset)
        if let cache { keys=concatenated([cache.0,keys],axis:2);values=concatenated([cache.1,values],axis:2) }
        let mask:MLXFast.ScaledDotProductAttentionMaskMode
        if t==1 { mask = .none }
        else {
            let qPos=MLXArray(0..<Int32(t)).expandedDimensions(axis:1)+Int32(offset)
            let kPos=MLXArray(0..<Int32(keys.dim(2))).expandedDimensions(axis:0)
            let additive=MLX.where(kPos .<= qPos,MLXArray(Float(0)),MLXArray(-Float.infinity))
            mask = .array(additive.reshaped(1,1,t,keys.dim(2)).asType(queries.dtype))
        }
        let merged=SDPA.attendAndMerge(qHeads:queries,kHeads:keys,vHeads:values,
            scale:1/sqrt(Float(headDim)),mask:mask)
        return (o(merged),(keys,values))
    }
}

final class MVGMLP:Module {
    @ModuleInfo(key:"gate_proj") var gate:Linear
    @ModuleInfo(key:"up_proj") var up:Linear
    @ModuleInfo(key:"down_proj") var down:Linear
    init(_ c:MossVoiceGeneratorConfiguration.Language) {
        _gate.wrappedValue=Linear(c.hiddenSize,c.intermediateSize,bias:false)
        _up.wrappedValue=Linear(c.hiddenSize,c.intermediateSize,bias:false)
        _down.wrappedValue=Linear(c.intermediateSize,c.hiddenSize,bias:false)
        super.init()
    }
    func callAsFunction(_ x:MLXArray)->MLXArray { down(silu(gate(x))*up(x)) }
}

final class MVGLayer:Module {
    @ModuleInfo(key:"self_attn") var attention:MVGAttention
    @ModuleInfo var mlp:MVGMLP
    @ModuleInfo(key:"input_layernorm") var norm1:RMSNorm
    @ModuleInfo(key:"post_attention_layernorm") var norm2:RMSNorm
    init(_ c:MossVoiceGeneratorConfiguration.Language) {
        _attention.wrappedValue=MVGAttention(c);_mlp.wrappedValue=MVGMLP(c)
        _norm1.wrappedValue=RMSNorm(dimensions:c.hiddenSize,eps:c.rmsNormEps)
        _norm2.wrappedValue=RMSNorm(dimensions:c.hiddenSize,eps:c.rmsNormEps)
        super.init()
    }
    func callAsFunction(_ x:MLXArray,_ cache:(MLXArray,MLXArray)?,_ offset:Int)->(MLXArray,(MLXArray,MLXArray)) {
        let (a,next)=attention(norm1(x),cache,offset);let h=x+a
        return(h+mlp(norm2(h)),next)
    }
}

final class MVGBackbone:Module {
    @ModuleInfo(key:"embed_tokens") var embedding:Embedding
    @ModuleInfo var layers:[MVGLayer]
    @ModuleInfo var norm:RMSNorm
    init(_ c:MossVoiceGeneratorConfiguration.Language) {
        _embedding.wrappedValue=Embedding(embeddingCount:c.vocabSize,dimensions:c.hiddenSize)
        _layers.wrappedValue=(0..<c.numHiddenLayers).map {_ in MVGLayer(c)}
        _norm.wrappedValue=RMSNorm(dimensions:c.hiddenSize,eps:c.rmsNormEps)
        super.init()
    }
    func callAsFunction(_ x:MLXArray,_ state:MVGKVState)->(MLXArray,MVGKVState) {
        var h=x,next=state
        for i in layers.indices { let (value,cache)=layers[i](h,state.layers[i],state.position);h=value;next.layers[i]=cache }
        next.position += x.dim(1)
        return(norm(h),next)
    }
}

final class MVGLanguageModel:Module {
    let configuration:MossVoiceGeneratorConfiguration
    @ModuleInfo(key:"language_model") var backbone:MVGBackbone
    @ModuleInfo(key:"emb_ext") var audioEmbeddings:[Embedding]
    @ModuleInfo(key:"lm_heads") var heads:[Linear]
    init(_ c:MossVoiceGeneratorConfiguration) {
        configuration=c;_backbone.wrappedValue=MVGBackbone(c.language)
        _audioEmbeddings.wrappedValue=(0..<c.codebooks).map {_ in Embedding(embeddingCount:c.audioVocabularySize+1,dimensions:c.language.hiddenSize)}
        _heads.wrappedValue=[Linear(c.language.hiddenSize,c.language.vocabSize,bias:false)] +
            (0..<c.codebooks).map {_ in Linear(c.language.hiddenSize,c.audioVocabularySize+1,bias:false)}
        super.init()
    }
    func initialState()->MVGKVState { MVGKVState(layers:Array(repeating:nil,count:configuration.language.numHiddenLayers)) }
    /// Every audio embedding, including the learned pad embedding, contributes at text positions.
    func embedded(_ ids:MLXArray)->MLXArray {
        var x=backbone.embedding(ids[0...,0...,0])
        for i in audioEmbeddings.indices { x=x+audioEmbeddings[i](ids[0...,0...,i+1]) }
        return x
    }
    func forward(_ ids:MLXArray,_ state:MVGKVState)->([MLXArray],MVGKVState) {
        let (h,next)=backbone(embedded(ids),state);let last=h[0...,-1,0...]
        return(heads.map {$0(last)},next)
    }
}
