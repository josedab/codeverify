# CodeVerify JetBrains Plugin

IntelliJ IDEA / PyCharm / WebStorm plugin for CodeVerify - AI-powered code verification with formal proofs.

## Features

- 🔍 **Real-time Verification**: Code is verified as you type or save
- 🛡️ **Formal Proofs**: Z3 SMT solver integration for mathematical proofs
- 🤖 **AI Analysis**: Semantic analysis powered by LLMs
- 🎯 **Trust Scoring**: Trust scores for code changes
- 🔧 **One-Click Fixes**: Apply suggested fixes instantly
- 📊 **Coverage Dashboard**: Visual proof coverage maps

## Supported Languages

- Python
- TypeScript / JavaScript
- Java
- Kotlin
- Go
- Rust

## Installation

### From JetBrains Marketplace

1. Open your JetBrains IDE
2. Go to `Settings` → `Plugins` → `Marketplace`
3. Search for "CodeVerify"
4. Click `Install`

### Manual Installation

1. Download the latest `.zip` from [Releases](https://github.com/codeverify/jetbrains-plugin/releases)
2. Go to `Settings` → `Plugins` → ⚙️ → `Install Plugin from Disk`
3. Select the downloaded `.zip` file

## Configuration

1. Go to `Settings` → `Tools` → `CodeVerify`
2. Enter your API key (get one at https://codeverify.dev/settings)
3. Configure verification options

### Offline Mode

For air-gapped environments:

1. Install [Ollama](https://ollama.ai)
2. Pull a code model: `ollama pull codellama:7b-instruct`
3. Enable "Offline Mode" in settings
4. Configure Ollama URL (default: `http://localhost:11434`)

## Usage

### Verify Current File

- Right-click → `CodeVerify` → `Verify This File`
- Or use keyboard shortcut: `Ctrl+Alt+V` (Windows/Linux), `⌘+Alt+V` (macOS)

### Verify Selection

1. Select code
2. Right-click → `CodeVerify` → `Verify Selection`

### Apply Fixes

When a finding has a suggested fix:
1. Place cursor on the highlighted line
2. Press `Alt+Enter` to show intentions
3. Select "Apply CodeVerify Fix"

### View Proofs

1. Click on a finding annotation
2. Select "Show Verification Proof"

## Building from Source

### Requirements

- JDK 17+
- Gradle 8+

### Build

```bash
./gradlew build
```

### Run in Development

```bash
./gradlew runIde
```

### Package

```bash
./gradlew buildPlugin
```

The plugin will be in `build/distributions/`.

## API Reference

The plugin communicates with the CodeVerify API:

```
POST /api/v1/verification/verify
Header: X-API-Key: <your-api-key>

{
    "code": "...",
    "language": "python",
    "includeProof": true,
    "includeFixes": true
}
```

## License

MIT License - see LICENSE file.

## Support

- Documentation: https://docs.codeverify.dev
- Issues: https://github.com/codeverify/jetbrains-plugin/issues
- Email: support@codeverify.dev
