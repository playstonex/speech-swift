import Foundation

public struct FishAudioModelInput: Sendable, Equatable {
    public let rows: [[Int32]]

    public var rowCount: Int { rows.count }
    public var tokenCount: Int { rows.first?.count ?? 0 }

    public init(rows: [[Int32]]) throws {
        guard !rows.isEmpty else {
            throw FishAudioError.invalidCodebookShape("model input must have at least one row")
        }
        let width = rows[0].count
        guard rows.allSatisfy({ $0.count == width }) else {
            throw FishAudioError.invalidCodebookShape("model input rows must have the same length")
        }
        self.rows = rows
    }

}
