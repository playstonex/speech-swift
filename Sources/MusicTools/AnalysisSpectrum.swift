import Accelerate
import Foundation

struct AnalysisSpectralFeatures {
    let novelty: [Double]
    let frameRate: Double
    let chroma: [Double]
    let pitchedFrameCount: Int

    static func empty(sampleRate: Double) -> Self {
        Self(novelty: [], frameRate: sampleRate / max(1, (sampleRate / 43.06640625).rounded()),
             chroma: [Double](repeating: 0, count: 12), pitchedFrameCount: 0)
    }
}

enum AnalysisSpectrum {
    static func extract(channels: [[Float]], sampleRate: Double) throws -> AnalysisSpectralFeatures {
        let logSize = Int(log2(sampleRate * 0.18576).rounded())
        let size = 1 << min(16, max(10, logSize))
        let hop = max(1, Int((sampleRate / 43.06640625).rounded()))
        guard channels[0].count >= size else { return .empty(sampleRate: sampleRate) }
        guard let setup = vDSP_create_fftsetup(vDSP_Length(log2(Double(size))), FFTRadix(kFFTRadix2)) else {
            throw MusicAnalysisError.conversionFailed("Could not create the spectrum transform.")
        }
        defer { vDSP_destroy_fftsetup(setup) }
        let window = (0..<size).map { Float(0.5 - 0.5 * cos(2 * .pi * Double($0) / Double(size - 1))) }
        let binCount = size / 2 + 1
        let normalization = 1 / (Double(size) * Double(size) * Double(channels.count))
        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)
        var power = [Double](repeating: 0, count: binCount)
        var previous = [Double](repeating: 0, count: binCount)
        var novelty: [Double] = []
        novelty.reserveCapacity(channels[0].count / hop)
        var chroma = [Double](repeating: 0, count: 12)
        var pitchedFrames = 0
        let lower = max(1, Int(55 * Double(size) / sampleRate))
        let upper = min(binCount - 2, Int(5_000 * Double(size) / sampleRate))

        for offset in stride(from: 0, through: channels[0].count - size, by: hop) {
            for k in power.indices { power[k] = 0 }
            for channel in channels {
                for k in 0..<size {
                    real[k] = channel[offset + k] * window[k]
                    imag[k] = 0
                }
                real.withUnsafeMutableBufferPointer { re in
                    imag.withUnsafeMutableBufferPointer { im in
                        var split = DSPSplitComplex(realp: re.baseAddress!, imagp: im.baseAddress!)
                        vDSP_fft_zip(setup, &split, 1, vDSP_Length(log2(Double(size))), FFTDirection(FFT_FORWARD))
                    }
                }
                for k in 0..<binCount {
                    power[k] += (Double(real[k]) * Double(real[k]) + Double(imag[k]) * Double(imag[k])) * normalization
                }
            }
            var flux = 0.0
            for k in lower...upper {
                let magnitude = log1p(20 * sqrt(power[k]))
                flux += max(0, magnitude - previous[k])
                previous[k] = magnitude
            }
            novelty.append(flux)

            let band = power[lower...upper]
            let mean = band.reduce(0, +) / Double(band.count)
            guard mean > 1e-12 else { continue }
            let flatness = exp(band.reduce(0) { $0 + log(max($1, 1e-18)) } / Double(band.count)) / mean
            guard flatness < 0.30 else { continue }
            let maximum = band.max() ?? 0
            var frameChroma = [Double](repeating: 0, count: 12)
            let keyLower = max(lower, Int(110 * Double(size) / sampleRate))
            let keyUpper = min(upper, Int(2_000 * Double(size) / sampleRate))
            guard keyLower < keyUpper else { continue }
            for k in keyLower...keyUpper {
                guard power[k] > power[k - 1], power[k] >= power[k + 1],
                      power[k] > maximum * 0.0025 else { continue }
                let left = log(max(power[k - 1], 1e-18))
                let center = log(max(power[k], 1e-18))
                let right = log(max(power[k + 1], 1e-18))
                let denominator = left - 2 * center + right
                let correction = abs(denominator) > 1e-12 ? max(-0.5, min(0.5, 0.5 * (left - right) / denominator)) : 0
                let frequency = (Double(k) + correction) * sampleRate / Double(size)
                let midi = 69 + 12 * log2(frequency / 440)
                guard abs(midi - midi.rounded()) <= 0.40 else { continue }
                let pitchClass = (Int(midi.rounded()) % 12 + 12) % 12
                frameChroma[pitchClass] += pow(power[k], 0.35) / sqrt(frequency / 220)
            }
            let sum = frameChroma.reduce(0, +)
            if sum > 0 {
                pitchedFrames += 1
                for index in chroma.indices { chroma[index] += frameChroma[index] / sum * sqrt(mean) }
            }
        }
        return AnalysisSpectralFeatures(
            novelty: novelty, frameRate: sampleRate / Double(hop),
            chroma: chroma, pitchedFrameCount: pitchedFrames
        )
    }
}
