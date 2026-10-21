import XCTest
@testable import AudioCommon

final class PlaybackSampleBufferTests: XCTestCase {
    private func read(_ buffer: PlaybackSampleBuffer, count: Int) -> [Float] {
        var samples = [Float](repeating: 0, count: count)
        let read = samples.withUnsafeMutableBufferPointer { buffer.read(into: $0.baseAddress!, count: count) }
        return Array(samples.prefix(read))
    }

    func testOversizedChunkPreservesEverySampleAndFinalWordTail() {
        let buffer = PlaybackSampleBuffer(capacity: 8)
        let waveform = (0..<100).map(Float.init)
        buffer.write(waveform)
        XCTAssertEqual(buffer.availableToRead, waveform.count)
        var rendered: [Float] = []
        while buffer.availableToRead > 0 { rendered += read(buffer, count: 11) }
        XCTAssertEqual(rendered, waveform)
        XCTAssertEqual(buffer.availableToRead, 0)
    }

    func testFastProducerKeepsOrderingAcrossOverflowAndPartialDrain() {
        let buffer = PlaybackSampleBuffer(capacity: 5)
        buffer.write([1, 2, 3, 4, 5, 6])
        XCTAssertEqual(read(buffer, count: 3), [1, 2, 3])
        buffer.write([7, 8])
        XCTAssertEqual(read(buffer, count: 4), [4, 5, 6, 7])
        buffer.write([9, 10, 11, 12, 13])
        XCTAssertEqual(read(buffer, count: 20), [8, 9, 10, 11, 12, 13])
    }

    func testResetDiscardsOldGenerationIncludingOverflow() {
        let buffer = PlaybackSampleBuffer(capacity: 4)
        buffer.write([1, 2, 3, 4, 5, 6])
        buffer.reset()
        buffer.write([9, 10])
        XCTAssertEqual(read(buffer, count: 10), [9, 10])
    }

    func testReclaimingConsumedChunksPreservesUnplayedTail() {
        let buffer = PlaybackSampleBuffer(capacity: 5)
        buffer.write((1...9).map(Float.init))
        XCTAssertEqual(read(buffer, count: 6), (1...6).map(Float.init))
        buffer.reclaimConsumed()
        buffer.write([10, 11, 12])
        XCTAssertEqual(read(buffer, count: 20), (7...12).map(Float.init))
        buffer.reclaimConsumed()
        buffer.write([13])
        XCTAssertEqual(read(buffer, count: 2), [13])
    }
}
