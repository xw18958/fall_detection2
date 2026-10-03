import SwiftUI

struct MainView: View {
    @ObservedObject var viewModel: CollectorViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("M5 Data Collector")
                    .font(.title2.weight(.semibold))
                Text("Connect the M5, record on the device, and wait for a safe-save confirmation before starting the next recording.")
                    .foregroundStyle(.secondary)
            }

            GroupBox("Save location") {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 10) {
                        Text(viewModel.displayDataPath)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                        Button("Choose Folder") { viewModel.chooseDataFolder() }
                            .disabled(!viewModel.canChangeSaveFolder)
                    }
                    HStack {
                        if !viewModel.hasSaveFolder {
                            Text("Choose where recordings should be saved before collection starts.")
                                .foregroundStyle(.secondary)
                        } else if !viewModel.canChangeSaveFolder {
                            Text("Finish the current recording before changing folders.")
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Open Data Folder") { viewModel.openDataFolder() }
                            .disabled(!viewModel.hasSaveFolder)
                    }
                }
                .padding(.vertical, 4)
            }

            GroupBox("Collector") {
                VStack(alignment: .leading, spacing: 14) {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(viewModel.connectionIndicatorColor)
                            .frame(width: 10, height: 10)
                        Text(viewModel.connectionText)
                            .fontWeight(.medium)
                    }

                    HStack(alignment: .firstTextBaseline) {
                        Text(viewModel.stateText)
                            .font(.system(size: 28, weight: .bold, design: .rounded))
                        Spacer()
                        if viewModel.showRecordingTimer {
                            Text(viewModel.recordingTimeText)
                                .font(.system(size: 24, weight: .semibold, design: .monospaced))
                        }
                    }

                    if let progress = viewModel.saveProgress {
                        VStack(alignment: .leading, spacing: 6) {
                            ProgressView(value: progress)
                            Text("Saving to Mac… \(Int((progress * 100).rounded()))%")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Text(viewModel.message)
                        .fontWeight(viewModel.stateText == "COMPLETE" ? .semibold : .regular)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
            }

            GroupBox("Last recording") {
                VStack(alignment: .leading, spacing: 7) {
                    Text(viewModel.lastSaveText)
                        .fontWeight(.medium)
                    if viewModel.hasSavedRecording {
                        Text("Data quality: \(viewModel.dataQualityText)")
                            .foregroundStyle(viewModel.dataQualityWarning ? .orange : .secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
            }

            if let error = viewModel.lastError {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
        .padding(22)
        .frame(width: 560)
    }
}
