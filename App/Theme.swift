import SwiftUI

/// Warm dark palette: ivory text on near-black, one clay accent.
enum Theme {
    static let ground = Color(hex: 0x1F1E1D)
    static let surface = Color(hex: 0x262624)
    static let raised = Color(hex: 0x30302E)
    static let high = Color(hex: 0x3D3D3A)
    static let track = Color(hex: 0x4D4C48)
    static let text = Color(hex: 0xFAF9F5)
    static let textBody = Color(hex: 0xE8E6DC)
    static let secondary = Color(hex: 0xC2C0B6)
    static let muted = Color(hex: 0x9C9A92)
    static let clay = Color(hex: 0xD97757)
    static let clayLight = Color(hex: 0xE8956F)
    static let onClay = Color(hex: 0x141413)
    static let hairline = Color(hex: 0xFAF9F5).opacity(0.08)
    static let border = Color(hex: 0xFAF9F5).opacity(0.16)
    static let danger = Color(hex: 0xE5735F)

    static func serif(_ size: CGFloat) -> Font {
        .system(size: size, weight: .regular, design: .serif)
    }
}

extension Color {
    init(hex: UInt32) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}
