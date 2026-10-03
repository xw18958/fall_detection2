import SwiftUI

struct MainView: View {
    @ObservedObject var viewModel: CollectorViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("M5 Data Collector")
                    .font(.title2.weight(.semibold))
                Text("Turn on the M5 in COLLECT mode. The app connects and saves kept trials automatically.")
                    .foregroundStyle(.secondary)
            }

            GroupBox("Session") {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
                    GridRow {
                        Text("Participant ID")
                        TextField("P001", text: $viewModel.participant)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 250)
                    }
                    GridRow {
                        Text("Placement")
                        TextField("chest_front_axes_up", text: $viewModel.placement)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 250)
                    }
                }
                .padding(.vertical, 4)
                .disabled(viewModel.profileLocked)
            }

            GroupBox("Collector status") {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 9) {
                    GridRow { Text("Connection"); Text(viewModel.connectionText).fontWeight(.medium) }
                    GridRow { Text("Device"); Text(viewModel.deviceName) }
                    GridRow { Text("M5 state"); Text(viewModel.stateText).fontWeight(.semibold) }
                    GridRow { Text("Samples saved"); Text("\(viewModel.samplesSaved)") }
                    GridRow { Text("Trials saved"); Text("\(viewModel.trialsSaved)") }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
            }

            Text(viewModel.message)
                .frame(maxWidth: .infinity, alignment: .leading)

            if let error = viewModel.lastError {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }

            HStack {
                Button("Open Data Folder") { viewModel.openDataFolder() }
                Spacer()
                Text(viewModel.displayDataPath)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(22)
        .frame(width: 520)
    }
}
