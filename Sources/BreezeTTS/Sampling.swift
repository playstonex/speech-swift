import MLX

/// Returns an integer tensor so dependent codebooks can stay on the device.
func breezeSampleCodebook(_ logits: MLXArray, codecVocabSize: Int, audioVocabSize: Int,
                          allowsEOS: Bool, options: BreezeTTSOptions,
                          random: MLXRandom.RandomState) -> MLXArray {
    var values = logits[0, ..<codecVocabSize].asType(.float32)
    if allowsEOS {
        values = concatenated([values, logits[0, audioVocabSize...audioVocabSize].asType(.float32)])
    }
    let index: MLXArray
    if options.temperature == 0 {
        index = argMax(values)
    } else {
        values = values / options.temperature
        if options.topK > 0 && options.topK < values.size {
            let cutoff = sorted(values)[values.size - options.topK]
            values = MLX.where(values .< cutoff, MLXArray(-Float.infinity), values)
        }
        if options.topP < 1 {
            let order = argSort(-values)
            let ordered = values[order]
            let probabilities = softmax(ordered)
            let keep = (cumsum(probabilities) - probabilities) .< options.topP
            let filtered = MLX.where(keep, ordered, MLXArray(-Float.infinity))
            index = order[MLXRandom.categorical(filtered, key: random.next())]
        } else {
            index = MLXRandom.categorical(values, key: random.next())
        }
    }
    let token = index.asType(.int32)
    return allowsEOS ? MLX.where(token .== codecVocabSize, MLXArray(Int32(audioVocabSize)), token) : token
}
