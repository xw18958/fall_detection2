import SwiftUI

struct MainView: View {
    @ObservedObject var viewModel: CollectorViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            header
            saveLocationCard
            collectorCard
            lastRecordingCard

            if let error = viewModel.lastError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
        .padding(24)
        .frame(width: 560)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("M5 Data Collector")
                .font(.title2)
                .fontWeight(.semibold)
            Text("Record on the M5 and wait for the save confirmation before starting the next movement.")
                .font(.body)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var saveLocationCard: some View {
        card {
            VStack(alignment: .leading, spacing: 14) {
                Label("Save Location", systemImage: "folder")
                    .font(.headline)

                HStack(spacing: 12) {
                    Text(viewModel.displayDataPath)
                        .font(.body)
                        .foregroundStyle(viewModel.hasSaveFolder ? .primary : .secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)

                    Button("Choose…") { viewModel.chooseDataFolder() }
                        .buttonStyle(.bordered)
                        .disabled(!viewModel.canChangeSaveFolder)
                }

                HStack {
                    if !viewModel.hasSaveFolder {
                        Text("Choose a folder before collection starts.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else if !viewModel.canChangeSaveFolder {
                        Text("Finish the current recording before changing folders.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Show in Finder") { viewModel.openDataFolder() }
                        .buttonStyle(.link)
                        .disabled(!viewModel.hasSaveFolder)
                }
            }
        }
    }

    private var collectorCard: some View {
        card {
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(viewModel.connectionIndicatorColor)
                        .frame(width: 8, height: 8)
                    Text(viewModel.connectionText)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                HStack(alignment: .firstTextBaseline) {
                    Text(viewModel.stateText)
                        .font(.title)
                        .fontWeight(.semibold)
                    Spacer()
                    if viewModel.showRecordingTimer {
                        Text(viewModel.recordingTimeText)
                            .font(.title2)
                            .fontWeight(.medium)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }

                if let progress = viewModel.saveProgress {
                    VStack(alignment: .leading, spacing: 7) {
                        ProgressView(value: progress)
                        Text("Saving… \(Int((progress * 100).rounded()))%")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Text(viewModel.message)
                    .font(.body)
                    .foregroundStyle(viewModel.stateText == "COMPLETE" ? .primary : .secondary)
                    .fontWeight(viewModel.stateText == "COMPLETE" ? .medium : .regular)
            }
        }
    }

    private var lastRecordingCard: some View {
        card {
            VStack(alignment: .leading, spacing: 10) {
                Label("Last Recording", systemImage: "waveform")
                    .font(.headline)

                Text(viewModel.lastSaveText)
                    .font(.body)

                if viewModel.hasSavedRecording {
                    HStack(spacing: 7) {
                        Image(systemName: viewModel.dataQualityWarning ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                            .foregroundStyle(viewModel.dataQualityWarning ? .orange : .green)
                        Text(viewModel.dataQualityText)
                            .font(.subheadline)
                            .foregroundStyle(viewModel.dataQualityWarning ? .primary : .secondary)
                    }
                }
            }
        }
    }

    private func card<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(nsColor: .controlBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.45), lineWidth: 0.5)
            )
    }
}
