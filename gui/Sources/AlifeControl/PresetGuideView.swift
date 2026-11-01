import AlifeCore
import AppKit
import SwiftUI

// MARK: - The snapshot

/// The snapshot JPEGs read from the repository, kept while the guide is on screen. The cache is keyed by the file's path and modification
/// date, so a snapshot made again while the app runs shows at the next redraw; a missing or unreadable file is nil and not cached.
@MainActor
enum PresetSnapshotImages {
    private static let cache = NSCache<NSString, NSImage>()

    static func key(for url: URL) -> NSString? {
        guard let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate else { return nil }
        return "\(url.path)|\(date.timeIntervalSinceReferenceDate)" as NSString
    }

    static func image(at url: URL) -> NSImage? {
        guard let key = key(for: url) else { return nil }
        if let hit = cache.object(forKey: key) { return hit }
        guard let image = NSImage(contentsOf: url), image.isValid else { return nil }
        cache.setObject(image, forKey: key)
        return image
    }
}

/// The snapshot of a preset at `width` points (the images are 640 x 400, a 8:5 picture), or a neutral placeholder when there is none.
struct PresetSnapshotView: View {
    let entry: PresetGuideEntry
    let width: CGFloat

    private var height: CGFloat { (width * 0.625).rounded() }

    var body: some View {
        Group {
            if let image = PresetSnapshotImages.image(at: entry.imageURL) {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fill)
                    .accessibilityLabel(entry.imageLabel)
            } else {
                ZStack {
                    Color.secondary.opacity(0.12)
                    VStack(spacing: 4) {
                        Image(systemName: "photo").font(.title3)
                        Text("No snapshot").font(.caption2)
                    }
                    .foregroundStyle(.secondary)
                }
                .accessibilityLabel(entry.imageLabel)
            }
        }
        .frame(width: width, height: height)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.secondary.opacity(0.35), lineWidth: 0.5))
    }
}

// MARK: - The words

/// The facts, the description and the captions of one preset: the card's text and a row of the compare sheet.
struct PresetTexts: View {
    let entry: PresetGuideEntry
    let machine: String
    var showsName = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if showsName {
                Text(entry.preset.rawValue).font(.headline)
            }
            if let facts = entry.facts {
                Text("\(facts.shapeName), \(facts.worldText)").font(.caption.weight(.semibold))
                Text(facts.populationText).font(.caption).foregroundStyle(.secondary)
            }
            if let note = entry.nameNote {
                Text(note).font(.caption).foregroundStyle(.secondary)
            }
            Text(entry.guide.what).font(.caption)
            labelled("Good for", entry.guide.goodFor)
            labelled("Cost", entry.guide.cost)
            if let caption = entry.snapshotCaption() {
                Text(caption).font(.caption2).foregroundStyle(.secondary)
            } else {
                Text("No snapshot has been made for this preset.").font(.caption2).foregroundStyle(.secondary)
            }
            if let speed = entry.speedCaption(machine: machine) {
                Text(speed).font(.caption2).foregroundStyle(.secondary)
            }
            if let memory = entry.memoryCaption() {
                Text(memory).font(.caption2).foregroundStyle(.secondary)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func labelled(_ label: String, _ text: String) -> some View {
        (Text("\(label): ").fontWeight(.semibold) + Text(text)).font(.caption)
    }
}

// MARK: - The card under the picker

/// The selected preset: its snapshot beside what it is, and a button for the comparison of all of them.
struct PresetCard: View {
    let entry: PresetGuideEntry
    let machine: String
    let compare: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                PresetSnapshotView(entry: entry, width: 132)
                PresetTexts(entry: entry, machine: machine)
            }
            HStack {
                Text("The picture is a frame of a real run of this preset, not a promise.")
                    .font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Button("Compare presets...", action: compare)
                    .controlSize(.small)
                    .help("All presets side by side, each with a snapshot and what it costs")
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.secondary.opacity(0.25), lineWidth: 0.5))
        .accessibilityElement(children: .contain)
    }
}

// MARK: - The compare sheet

/// A row of the compare sheet: a button that chooses the preset.
struct PresetCompareRow: View {
    let entry: PresetGuideEntry
    let machine: String
    let isSelected: Bool
    let choose: () -> Void

    var body: some View {
        Button(action: choose) {
            HStack(alignment: .top, spacing: 12) {
                PresetSnapshotView(entry: entry, width: 208)
                PresetTexts(entry: entry, machine: machine, showsName: true)
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 8).fill(isSelected ? Color.accentColor.opacity(0.14) : Color.secondary.opacity(0.06)))
            .overlay(RoundedRectangle(cornerRadius: 8)
                .strokeBorder(isSelected ? Color.accentColor : Color.secondary.opacity(0.25), lineWidth: isSelected ? 1.5 : 0.5))
            .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        // One element that says everything the row shows (the facts, the words, the captions), not only the name and the first sentence.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(entry.spokenText(machine: machine))
        .accessibilityHint("Uses this preset and closes the list")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .help("Use \(entry.preset.rawValue)")
    }
}

/// The ten presets in three groups. It has no scroll view of its own, so that a test can draw all of it.
struct PresetCompareList: View {
    let content: PresetGuideContent
    let selected: Preset
    let choose: (Preset) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(PresetGroup.allCases) { group in
                VStack(alignment: .leading, spacing: 6) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(group.title).font(.title3.weight(.semibold))
                        Text(group.blurb).font(.caption).foregroundStyle(.secondary)
                    }
                    ForEach(content.entries(in: group), id: \.preset) { entry in
                        PresetCompareRow(entry: entry, machine: content.machine, isSelected: entry.preset == selected) {
                            choose(entry.preset)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// "Compare presets": every preset with its snapshot and what it costs. A click chooses that preset (the caller sets the same value the
/// picker sets, so what the picker does on a change still happens) and closes the sheet.
struct PresetCompareSheet: View {
    let content: PresetGuideContent
    let selected: Preset
    let choose: (Preset) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Compare presets").font(.title2.weight(.semibold))
                Spacer()
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
            Divider()
            ScrollView {
                PresetCompareList(content: content, selected: selected) { preset in
                    choose(preset)
                    dismiss()
                }
                .padding(16)
            }
            Divider()
            Text("Every picture is a frame of a real run of that preset: seed 1, "
                 + "no extras switched on, the preset's own settings (only a checkpoint every 5\u{2009}000 steps instead of every 10\u{2009}000). "
                 + "Speed and memory were measured on a separate continuation of that run, on its own on "
                 + "\(content.machine.isEmpty ? "the Mac that made the snapshots" : content.machine), at the number of organisms shown. A run slows as its "
                 + "population grows, and another Mac or other settings change it. Click a preset to use it.")
                .font(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 16).padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minWidth: 620, idealWidth: 740, maxWidth: 1000, minHeight: 380, idealHeight: 580, maxHeight: 680)
    }
}
