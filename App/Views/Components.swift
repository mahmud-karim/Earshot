import SwiftUI

/// The clay orb with its two rings. It breathes while listening and pulses with the voice.
struct Orb: View {
    var size: CGFloat = 148
    var level: Float = 0
    var active = false
    @State private var breathe = false

    var body: some View {
        ZStack {
            Circle().stroke(Theme.clay.opacity(0.22), lineWidth: 1)
                .scaleEffect(active ? 1 + CGFloat(level) * 0.12 : (breathe ? 1.03 : 1))
            Circle().stroke(Theme.clay.opacity(0.45), lineWidth: 1)
                .frame(width: size * 0.73, height: size * 0.73)
                .scaleEffect(active ? 1 + CGFloat(level) * 0.08 : 1)
            Circle().fill(Theme.clay)
                .frame(width: size * 0.46, height: size * 0.46)
        }
        .frame(width: size, height: size)
        .animation(.easeOut(duration: 0.12), value: level)
        .onAppear {
            withAnimation(.easeInOut(duration: 2.4).repeatForever(autoreverses: true)) { breathe = true }
        }
        .accessibilityHidden(true)
    }
}

/// A round button with an outline, for the secondary actions beside the main one.
struct RoundButton: View {
    let systemImage: String
    let label: String
    var size: CGFloat = 56
    var filled = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: size * 0.36, weight: .medium))
                .frame(width: size, height: size)
                .foregroundStyle(filled ? Theme.onClay : Theme.secondary)
                .background(Circle().fill(filled ? Theme.clay : Theme.surface))
                .overlay(Circle().stroke(filled ? Color.clear : Theme.border, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

/// The clay pill: "Tap to talk".
struct PrimaryPill: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: systemImage).font(.system(size: 20, weight: .medium))
                Text(title).font(.system(size: 17, weight: .semibold))
            }
            .frame(maxWidth: .infinity)
            .frame(height: 64)
            .foregroundStyle(Theme.onClay)
            .background(Capsule().fill(Theme.clay))
        }
        .buttonStyle(.plain)
    }
}

/// A segmented control in the app's colours.
struct Segmented<Value: Hashable>: View {
    let options: [(Value, String)]
    @Binding var selection: Value
    var height: CGFloat = 36

    var body: some View {
        HStack(spacing: 4) {
            ForEach(options.indices, id: \.self) { index in
                let (value, title) = options[index]
                let selected = value == selection
                Button {
                    selection = value
                } label: {
                    Text(title)
                        .font(.system(size: 14, weight: selected ? .medium : .regular))
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                        .foregroundStyle(selected ? Theme.text : Theme.secondary)
                        .frame(maxWidth: .infinity)
                        .frame(height: height)
                        .background(RoundedRectangle(cornerRadius: 10).fill(selected ? Theme.high : Color.clear))
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(4)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.surface))
    }
}

/// Bars that follow the microphone.
struct LevelBars: View {
    let levels: [Float]
    var barWidth: CGFloat = 4
    var maxHeight: CGFloat = 56
    var spacing: CGFloat = 4

    var body: some View {
        HStack(alignment: .center, spacing: spacing) {
            ForEach(Array(levels.enumerated()), id: \.offset) { _, level in
                Capsule()
                    .fill(Theme.clay.opacity(0.45 + Double(level) * 0.55))
                    .frame(width: barWidth, height: max(barWidth + 2, CGFloat(level) * maxHeight))
            }
        }
        .frame(height: maxHeight)
        .animation(.linear(duration: 0.08), value: levels)
        .accessibilityHidden(true)
    }
}

/// Small grey caption above a grouped list.
struct GroupLabel: View {
    let text: String
    var body: some View {
        Text(text).font(.system(size: 13)).foregroundStyle(Theme.muted)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A rounded card holding rows separated by hairlines.
struct RowGroup<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        VStack(spacing: 0) {
            content
        }
        .background(RoundedRectangle(cornerRadius: 16).fill(Theme.surface))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Theme.hairline, lineWidth: 0.5))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

struct Hairline: View {
    var body: some View {
        Rectangle().fill(Theme.hairline).frame(height: 0.5)
    }
}

/// One row: a name, an optional line under it, and something on the right.
struct RowLabel<Trailing: View>: View {
    let title: String
    var subtitle: String?
    var titleColor: Color = Theme.text
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(size: 16)).foregroundStyle(titleColor)
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle).font(.system(size: 13)).foregroundStyle(Theme.muted)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
            trailing
        }
        .padding(.leading, 16)
        .padding(.trailing, 14)
        .padding(.vertical, 9)
        .frame(minHeight: 56)
        .contentShape(Rectangle())
    }
}

struct Chevron: View {
    var body: some View {
        Image(systemName: "chevron.right").font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.muted)
    }
}

struct Checkmark: View {
    var body: some View {
        Image(systemName: "checkmark").font(.system(size: 16, weight: .bold)).foregroundStyle(Theme.clay)
            .accessibilityLabel("Selected")
    }
}

/// A clay-on-dark switch like the system one.
struct ClayToggle: View {
    @Binding var isOn: Bool
    var body: some View {
        Toggle("", isOn: $isOn).labelsHidden().tint(Theme.clay)
    }
}

/// Header for full-screen pages: back chevron and a serif title.
struct PageHeader<Trailing: View>: View {
    let title: String
    var back: (() -> Void)?
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 4) {
            if let back {
                Button(action: back) {
                    Image(systemName: "chevron.left").font(.system(size: 19, weight: .medium))
                        .foregroundStyle(Theme.secondary).frame(width: 44, height: 44)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Back")
                .padding(.leading, -12)
            }
            Text(title).font(Theme.serif(32)).foregroundStyle(Theme.text)
            Spacer()
            trailing
        }
        .frame(height: 44)
    }
}

/// A banner for something that went wrong.
struct ErrorBanner: View {
    let text: String
    let dismiss: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.circle").foregroundStyle(Theme.danger)
            Text(text).font(.system(size: 14)).foregroundStyle(Theme.textBody)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Button(action: dismiss) {
                Image(systemName: "xmark").font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.muted)
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Dismiss")
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.raised))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Theme.danger.opacity(0.35), lineWidth: 0.5))
    }
}

/// The month's spend against the budget.
struct SpendCard: View {
    let title: String
    let spent: Double
    let budget: Double

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text(title).font(.system(size: 14)).foregroundStyle(Theme.text)
                Spacer()
                Text("\(Money.text(spent)) of \(Money.text(budget, cents: false))")
                    .font(.system(size: 14)).foregroundStyle(Theme.secondary)
            }
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.high)
                    Capsule().fill(Theme.clay)
                        .frame(width: max(6, proxy.size.width * CGFloat(min(1, spent / max(budget, 1)))))
                }
            }
            .frame(height: 6)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(RoundedRectangle(cornerRadius: 16).fill(Theme.surface))
    }
}

enum Money {
    static func text(_ value: Double, cents: Bool = true) -> String {
        if !cents { return "$" + String(format: "%.0f", value) }
        if value > 0 && value < 0.01 { return "$" + String(format: "%.4f", value) }
        return "$" + String(format: "%.2f", value)
    }
}
