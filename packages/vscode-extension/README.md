# CodeVerify VS Code Extension

VS Code extension for CodeVerify - AI-powered code analysis with formal verification.

## Features

- **Real-time Analysis**: Automatically analyzes files on save
- **Inline Diagnostics**: See issues directly in your code with squiggly underlines
- **Quick Fixes**: Apply suggested fixes with one click
- **Findings Panel**: Browse all findings in the sidebar
- **Local Analysis**: Works offline using the CodeVerify CLI
- **API Integration**: Optionally sync with CodeVerify cloud for team features

## Installation

### From VS Code Marketplace

Search for "CodeVerify" in the Extensions view (`Ctrl+Shift+X`).

### From VSIX

1. Download the `.vsix` file from releases
2. In VS Code: Extensions → ⋯ → Install from VSIX

### Build from Source

```bash
cd packages/vscode-extension
npm install
npm run compile
npm run package
```

## Requirements

- VS Code 1.85.0 or higher
- CodeVerify CLI (for local analysis): `pip install codeverify-cli`

## Configuration

Open Settings (`Ctrl+,`) and search for "CodeVerify":

| Setting | Description | Default |
|---------|-------------|---------|
| `codeverify.enabled` | Enable CodeVerify | `true` |
| `codeverify.analyzeOnSave` | Auto-analyze on save | `true` |
| `codeverify.apiEndpoint` | API endpoint | `https://api.codeverify.io` |
| `codeverify.apiKey` | API key (optional) | `` |
| `codeverify.severityFilter` | Min severity to show | `all` |
| `codeverify.showInlineAnnotations` | Show inline annotations | `true` |
| `codeverify.localAnalysisEnabled` | Enable local CLI analysis | `true` |
| `codeverify.cliPath` | Path to CLI | `codeverify` |

## Commands

Access via Command Palette (`Ctrl+Shift+P`):

| Command | Description |
|---------|-------------|
| `CodeVerify: Analyze Current File` | Analyze the active file |
| `CodeVerify: Analyze Workspace` | Analyze entire workspace |
| `CodeVerify: Show Findings` | Open findings panel |
| `CodeVerify: Open Dashboard` | Open web dashboard |
| `CodeVerify: Refresh Findings` | Re-run analysis |

## Keyboard Shortcuts

| Shortcut | Command |
|----------|---------|
| `Ctrl+Shift+V` / `Cmd+Shift+V` | Analyze current file |

## Usage

### Automatic Analysis

With `analyzeOnSave` enabled, files are analyzed automatically when you save.

### Manual Analysis

1. Open a file
2. Press `Ctrl+Shift+V` or run "CodeVerify: Analyze Current File"
3. View findings in the Problems panel and inline

### Applying Fixes

1. Hover over a finding (squiggly underline)
2. Click the lightbulb or press `Ctrl+.`
3. Select "Fix: [issue name]"

### Dismissing Findings

1. Right-click on a finding in the tree view
2. Select "Dismiss Finding"
3. The finding will be hidden and pattern learned for future scans

## Supported Languages

- Python
- TypeScript / JavaScript
- Go
- Java

## Troubleshooting

### "CodeVerify CLI not found"

Install the CLI:
```bash
pip install codeverify-cli
```

Or specify the path in settings:
```json
{
  "codeverify.cliPath": "/path/to/codeverify"
}
```

### "Analysis not running"

1. Check that `codeverify.enabled` is `true`
2. Ensure the file language is supported
3. Check the Output panel (View → Output → CodeVerify)

### "API connection failed"

1. Verify your API key in settings
2. Check network connectivity
3. Try enabling local analysis as fallback

## Development

```bash
# Install dependencies
npm install

# Compile
npm run compile

# Watch mode
npm run watch

# Run tests
npm test

# Package
npm run package
```

## License

MIT
