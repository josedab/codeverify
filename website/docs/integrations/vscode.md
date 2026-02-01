---
sidebar_position: 4
---

# VS Code Extension

Real-time verification feedback directly in your editor.

## Installation

### From Marketplace

1. Open VS Code
2. Press `Ctrl+Shift+X` (Cmd+Shift+X on Mac)
3. Search for "CodeVerify"
4. Click **Install**

Or install via command line:

```bash
code --install-extension codeverify.codeverify-vscode
```

### From VSIX

Download the latest `.vsix` from [releases](https://github.com/codeverify/codeverify/releases):

```bash
code --install-extension codeverify-vscode-1.0.0.vsix
```

## Features

### Real-Time Analysis

As you type, CodeVerify analyzes your code and shows:

- **Squiggly underlines** for findings
- **Hover information** with details and fixes
- **Quick fixes** via lightbulb menu

### Inline Diagnostics

Findings appear inline in the editor:

```python
def get_user(user_id):
    users = fetch_users()
    return users[user_id]  # ⚠️ Array bounds: 'user_id' may exceed array length
```

Hover over the warning to see:
- Severity and category
- Detailed explanation
- Suggested fix
- Link to documentation

### Problems Panel

All findings appear in the Problems panel (`Ctrl+Shift+M`):

```
PROBLEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠ src/api.py
  ├─ Line 45: Potential null dereference [null_safety]
  └─ Line 78: Array index may be out of bounds [array_bounds]
⚠ src/utils.ts
  └─ Line 23: Possible integer overflow [integer_overflow]
```

### Quick Fixes

CodeVerify suggests fixes via the lightbulb menu:

1. Place cursor on a finding
2. Press `Ctrl+.` (Cmd+. on Mac)
3. Select a fix from the menu

Available quick fixes:
- Add null check
- Add bounds check
- Add type annotation
- Suppress finding (with comment)

### Status Bar

The status bar shows analysis status:

- ✅ **CodeVerify: OK** — No findings
- ⚠️ **CodeVerify: 3** — 3 findings found
- 🔄 **CodeVerify: Analyzing** — Analysis in progress
- ❌ **CodeVerify: Error** — Analysis failed

Click the status to open the output panel.

## Configuration

### Settings

Open VS Code settings (`Ctrl+,`) and search for "CodeVerify":

| Setting | Description | Default |
|---------|-------------|---------|
| `codeverify.enable` | Enable/disable extension | `true` |
| `codeverify.analyzeOnSave` | Analyze when file saved | `true` |
| `codeverify.analyzeOnType` | Analyze while typing | `true` |
| `codeverify.analyzeDelay` | Delay before analysis (ms) | `1000` |
| `codeverify.checks` | Checks to run | All |
| `codeverify.severityFilter` | Minimum severity to show | `low` |
| `codeverify.configFile` | Path to config file | `.codeverify.yml` |

### Settings JSON

```json
{
  "codeverify.enable": true,
  "codeverify.analyzeOnSave": true,
  "codeverify.analyzeOnType": true,
  "codeverify.analyzeDelay": 500,
  "codeverify.checks": [
    "null_safety",
    "array_bounds",
    "division_by_zero"
  ],
  "codeverify.severityFilter": "medium",
  "codeverify.ai.enabled": true,
  "codeverify.ai.provider": "openai"
}
```

### Workspace Settings

Create `.vscode/settings.json` for project-specific settings:

```json
{
  "codeverify.configFile": "./config/codeverify.yml",
  "codeverify.severityFilter": "high"
}
```

## Commands

Access via Command Palette (`Ctrl+Shift+P`):

| Command | Description |
|---------|-------------|
| `CodeVerify: Analyze File` | Analyze current file |
| `CodeVerify: Analyze Workspace` | Analyze entire workspace |
| `CodeVerify: Clear Diagnostics` | Clear all findings |
| `CodeVerify: Show Output` | Open output panel |
| `CodeVerify: Open Settings` | Open extension settings |
| `CodeVerify: Restart` | Restart the extension |

### Keyboard Shortcuts

Default shortcuts (customizable in keybindings):

| Shortcut | Command |
|----------|---------|
| `Ctrl+Shift+V` | Analyze current file |
| `Ctrl+Shift+Alt+V` | Analyze workspace |

## Authentication

### API Key

For cloud features, set your API key:

1. Open Command Palette
2. Run `CodeVerify: Set API Key`
3. Enter your API key

Or add to settings:

```json
{
  "codeverify.apiKey": "cv_..."
}
```

### AI Provider Keys

For AI features, configure provider keys:

```json
{
  "codeverify.ai.openaiApiKey": "sk-...",
  "codeverify.ai.anthropicApiKey": "sk-ant-..."
}
```

:::tip
Store sensitive keys in VS Code's secret storage using the command palette, not in settings files.
:::

## Copilot Trust Score

The extension shows Trust Score for AI-generated code:

1. When Copilot suggests code, CodeVerify analyzes it
2. A badge shows the Trust Score
3. Hover for detailed breakdown

Configure in settings:

```json
{
  "codeverify.copilot.showTrustScore": true,
  "codeverify.copilot.minimumScore": 70,
  "codeverify.copilot.warnBelow": 50
}
```

## Integration with Other Extensions

### GitHub Copilot

CodeVerify automatically integrates with GitHub Copilot to:
- Analyze suggested code before accepting
- Show Trust Scores for completions
- Flag potentially problematic suggestions

### GitLens

Findings include git blame information when GitLens is installed.

### Error Lens

Compatible with Error Lens for inline error display.

## Troubleshooting

### Extension Not Working

1. Check the output panel: `View > Output > CodeVerify`
2. Verify the CLI is installed: `codeverify --version`
3. Check settings are correct

### Slow Analysis

1. Increase `analyzeDelay` setting
2. Disable `analyzeOnType`, use `analyzeOnSave` instead
3. Reduce checks to essential ones
4. Exclude large files via `.codeverify.yml`

### High Memory Usage

```json
{
  "codeverify.advanced.maxFileSize": 500000,
  "codeverify.advanced.maxConcurrent": 2
}
```

### Debug Mode

Enable verbose logging:

```json
{
  "codeverify.debug": true,
  "codeverify.logLevel": "debug"
}
```

## Uninstall

1. Open Extensions panel
2. Find CodeVerify
3. Click **Uninstall**

Or via command line:

```bash
code --uninstall-extension codeverify.codeverify-vscode
```
