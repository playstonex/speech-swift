// Adapted from MOSS-VoiceGenerator and MOSS-Audio-Tokenizer.
// Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
// Licensed under Apache 2.0; see LICENSE.
// Native Swift adaptation uses request-local state and bounded causal caches.

import Foundation
import MLX

enum MVGSampling {
    /// Temperature precedes sign-dependent repetition penalty, then top-k and
    /// nucleus filtering. This is the upstream inference_utils.py order.
    static func sample(_ logits:MLXArray,temperature:Float,topP:Float,topK:Int,
                       history:Set<Int32>=[],penalty:Float=1,
                       random:MLXRandom.RandomState)throws->Int {
        var values=logits.reshaped([-1]).asType(.float32)/(temperature>0 ? temperature : 1)
        if penalty != 1,!history.isEmpty {
            let ids=MLXArray(history.sorted())
            let previous=values[ids]
            values[ids]=MLX.where(previous .> Float(0),previous/penalty,previous*penalty)
        }
        if temperature==0 {
            guard max(values).item(Float.self).isFinite else { throw MossVoiceGeneratorError.invalidOutput("no finite sampling candidate") }
            return argMax(values).item(Int.self)
        }
        var indices:MLXArray?
        if topK>0,topK<values.size {
            let start=values.size-topK
            let selected=argPartition(values,kth:start)[start...]
            indices=selected;values=values[selected]
        }
        if topP<1 {
            let order=argSort(-values)
            let sorted=values[order]
            let probabilities=softmax(sorted)
            // Keep the token that first crosses the threshold.
            let keep=(cumsum(probabilities)-probabilities) .<= topP
            values=MLX.where(keep,sorted,MLXArray(-Float.infinity))
            indices=indices.map {$0[order]} ?? order
        }
        guard max(values).item(Float.self).isFinite else { throw MossVoiceGeneratorError.invalidOutput("no finite sampling candidate") }
        let selected=MLXRandom.categorical(values,key:random.next()).item(Int.self)
        return indices?[selected].item(Int.self) ?? selected
    }
}
