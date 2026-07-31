/**
 * Copilot Interceptor Provider - Verifies Copilot suggestions before acceptance
 *
 * Intercepts GitHub Copilot suggestions and runs verification checks
 * showing inline status before the user accepts the code.
 */

import * as vscode from "vscode";
import {
  getDefaultInterceptorConfig,
  InterceptorConfig,
  MockCodeVerifyClient,
  SuggestionContext,
  SuggestionVerification,
  SuggestionVerificationClient,
  VerificationIssue,
  VerificationStatus,
} from "../copilotVerification";

// Re-exported for backward compatibility with any external importers of
// this module; the canonical definitions now live in copilotVerification.ts
// so they can be unit tested without a `vscode` mock.
export {
  VerificationStatus,
  VerificationIssue,
  SuggestionVerification,
  InterceptorConfig,
  MockCodeVerifyClient,
};

/**
 * Decoration types for inline status display
 */
const decorationTypes = {
  verified: vscode.window.createTextEditorDecorationType({
    after: {
      contentText: " ✓",
      color: "#4CAF50",
      fontWeight: "bold",
    },
    backgroundColor: "rgba(76, 175, 80, 0.1)",
  }),
  warning: vscode.window.createTextEditorDecorationType({
    after: {
      contentText: " ⚠",
      color: "#FF9800",
      fontWeight: "bold",
    },
    backgroundColor: "rgba(255, 152, 0, 0.1)",
  }),
  error: vscode.window.createTextEditorDecorationType({
    after: {
      contentText: " ✗",
      color: "#F44336",
      fontWeight: "bold",
    },
    backgroundColor: "rgba(244, 67, 54, 0.1)",
  }),
  pending: vscode.window.createTextEditorDecorationType({
    after: {
      contentText: " ⏳",
      color: "#9E9E9E",
    },
    backgroundColor: "rgba(158, 158, 158, 0.05)",
  }),
};

/**
 * Copilot Interceptor Provider
 *
 * Monitors Copilot suggestions and runs verification before acceptance.
 */
export class CopilotInterceptorProvider implements vscode.Disposable {
  private client: SuggestionVerificationClient;
  private config: InterceptorConfig;
  private statusBarItem: vscode.StatusBarItem;
  private pendingVerifications: Map<string, SuggestionVerification> = new Map();
  private disposables: vscode.Disposable[] = [];
  private outputChannel: vscode.OutputChannel;

  constructor(client: SuggestionVerificationClient) {
    this.client = client;
    this.config = getDefaultInterceptorConfig();
    this.outputChannel = vscode.window.createOutputChannel(
      "CodeVerify Interceptor"
    );

    // Create status bar item
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.statusBarItem.command = "codeverify.toggleInterceptor";
    this.updateStatusBar();
    this.statusBarItem.show();

    this.initialize();
  }

  private async initialize(): Promise<void> {
    // Load config from server
    try {
      this.config = await this.client.getConfig();
    } catch {
      this.log("Using default config");
    }

    // Register completion provider wrapper
    this.registerCompletionInterceptor();

    // Register text change listener for inline completions
    this.disposables.push(
      vscode.workspace.onDidChangeTextDocument((event) => {
        if (this.config.enabled && this.config.autoVerify) {
          this.handleTextChange(event);
        }
      })
    );

    // Register commands
    this.registerCommands();

    this.log("Copilot Interceptor initialized");
  }

  private registerCompletionInterceptor(): void {
    // Intercept inline completion provider
    const interceptor = vscode.languages.registerInlineCompletionItemProvider(
      { pattern: "**" },
      {
        provideInlineCompletionItems: async (
          document: vscode.TextDocument,
          position: vscode.Position,
          context: vscode.InlineCompletionContext,
          token: vscode.CancellationToken
        ): Promise<
          vscode.InlineCompletionList | vscode.InlineCompletionItem[] | null
        > => {
          // We don't provide completions, just intercept for verification
          // Let Copilot provide completions, we'll verify them post-acceptance
          return null;
        },
      }
    );

    this.disposables.push(interceptor);
  }

  private registerCommands(): void {
    // Toggle interceptor
    this.disposables.push(
      vscode.commands.registerCommand("codeverify.toggleInterceptor", () => {
        this.config.enabled = !this.config.enabled;
        this.updateStatusBar();
        vscode.window.showInformationMessage(
          `CodeVerify Interceptor ${this.config.enabled ? "enabled" : "disabled"}`
        );
      })
    );

    // Verify current selection
    this.disposables.push(
      vscode.commands.registerCommand("codeverify.verifySelection", () => {
        this.verifySelection();
      })
    );

    // Show verification details
    this.disposables.push(
      vscode.commands.registerCommand(
        "codeverify.showVerificationDetails",
        (suggestionId: string) => {
          this.showVerificationDetails(suggestionId);
        }
      )
    );

    // Apply suggested fix
    this.disposables.push(
      vscode.commands.registerCommand(
        "codeverify.applySuggestedFix",
        (issue: VerificationIssue) => {
          this.applySuggestedFix(issue);
        }
      )
    );
  }

  private async handleTextChange(
    event: vscode.TextDocumentChangeEvent
  ): Promise<void> {
    // Detect if this looks like a Copilot insertion
    // Copilot typically inserts multi-line completions
    for (const change of event.contentChanges) {
      const insertedText = change.text;

      // Heuristics for detecting Copilot suggestions:
      // 1. Multi-line insertion
      // 2. Contains function/method definitions
      // 3. Inserted at cursor position
      if (this.looksLikeCopilotSuggestion(insertedText)) {
        await this.verifySuggestion(
          insertedText,
          event.document,
          change.range.start
        );
      }
    }
  }
  private looksLikeCopilotSuggestion(text: string): boolean {
    if (!text || text.length < 10) return false;

    // Check for code patterns
    const codePatterns = [
      /\bfunction\b/,
      /\bdef\b/,
      /\bclass\b/,
      /\bconst\b|\blet\b|\bvar\b/,
      /=>/,
      /\bif\b.*\{/,
      /\bfor\b.*\{/,
      /\bwhile\b.*\{/,
    ];

    return codePatterns.some((pattern) => pattern.test(text));
  }

  private async verifySuggestion(
    code: string,
    document: vscode.TextDocument,
    position: vscode.Position
  ): Promise<void> {
    const suggestionId = this.generateSuggestionId(code, position);

    // Show pending status
    if (this.config.showInlineStatus) {
      this.showInlineStatus(
        document,
        position,
        VerificationStatus.Pending,
        suggestionId
      );
    }

    try {
      const context: SuggestionContext = {
        filePath: document.uri.fsPath,
        language: document.languageId,
        surroundingCode: this.getSurroundingCode(document, position),
        cursorPosition: position,
        documentVersion: document.version,
      };

      const result = await this.client.verifySuggestion(code, context);
      this.pendingVerifications.set(suggestionId, result);

      // Update inline status
      if (this.config.showInlineStatus) {
        this.showInlineStatus(document, position, result.status, suggestionId);
      }

      // Show issues in problems panel
      if (result.issues.length > 0) {
        this.reportIssues(document, position, result.issues);
      }

      // Block if configured and errors found
      if (
        this.config.blockOnError &&
        result.status === VerificationStatus.Error
      ) {
        this.showBlockingWarning(result);
      }

      // Log result
      this.log(
        `Verified suggestion: ${result.status} (score: ${result.score}, issues: ${result.issues.length})`
      );
    } catch (error) {
      this.log(`Verification error: ${error}`);
      this.showInlineStatus(
        document,
        position,
        VerificationStatus.Timeout,
        suggestionId
      );
    }
  }

  private generateSuggestionId(
    code: string,
    position: vscode.Position
  ): string {
    const hash = this.simpleHash(code);
    return `${hash}-${position.line}-${position.character}`;
  }

  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  }

  private getSurroundingCode(
    document: vscode.TextDocument,
    position: vscode.Position,
    lines: number = 10
  ): string {
    const startLine = Math.max(0, position.line - lines);
    const endLine = Math.min(document.lineCount - 1, position.line + lines);

    const range = new vscode.Range(startLine, 0, endLine, Number.MAX_VALUE);
    return document.getText(range);
  }

  private showInlineStatus(
    document: vscode.TextDocument,
    position: vscode.Position,
    status: VerificationStatus,
    suggestionId: string
  ): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document !== document) return;

    // Clear existing decorations
    for (const type of Object.values(decorationTypes)) {
      editor.setDecorations(type, []);
    }

    // Get the appropriate decoration type
    let decorationType: vscode.TextEditorDecorationType;
    switch (status) {
      case VerificationStatus.Verified:
        decorationType = decorationTypes.verified;
        break;
      case VerificationStatus.Warning:
        decorationType = decorationTypes.warning;
        break;
      case VerificationStatus.Error:
        decorationType = decorationTypes.error;
        break;
      default:
        decorationType = decorationTypes.pending;
    }

    // Create decoration
    const range = new vscode.Range(position, position);
    const decoration: vscode.DecorationOptions = {
      range,
      hoverMessage: this.getHoverMessage(suggestionId),
    };

    editor.setDecorations(decorationType, [decoration]);
  }

  private getHoverMessage(suggestionId: string): vscode.MarkdownString {
    const verification = this.pendingVerifications.get(suggestionId);
    if (!verification) {
      return new vscode.MarkdownString("⏳ Verification in progress...");
    }

    const md = new vscode.MarkdownString();
    md.isTrusted = true;

    // Status header
    const statusIcon =
      {
        [VerificationStatus.Verified]: "✓",
        [VerificationStatus.Warning]: "⚠",
        [VerificationStatus.Error]: "✗",
        [VerificationStatus.Pending]: "⏳",
        [VerificationStatus.Timeout]: "⏱",
      }[verification.status] || "?";

    md.appendMarkdown(
      `### CodeVerify ${statusIcon} ${verification.status.toUpperCase()}\n\n`
    );
    md.appendMarkdown(`**Trust Score:** ${verification.score}/100\n\n`);

    // Issues
    if (verification.issues.length > 0) {
      md.appendMarkdown("**Issues Found:**\n\n");
      for (const issue of verification.issues) {
        const icon = issue.severity === "error" ? "🔴" : "🟡";
        md.appendMarkdown(`${icon} ${issue.message}\n\n`);
        if (issue.fix) {
          md.appendMarkdown(
            `[Apply Fix](command:codeverify.applySuggestedFix?${encodeURIComponent(JSON.stringify(issue))})\n\n`
          );
        }
      }
    } else {
      md.appendMarkdown("✅ No issues found\n\n");
    }

    md.appendMarkdown(
      `\n*Verified in ${verification.verificationTimeMs}ms*`
    );

    return md;
  }

  private reportIssues(
    document: vscode.TextDocument,
    startPosition: vscode.Position,
    issues: VerificationIssue[]
  ): void {
    const diagnostics: vscode.Diagnostic[] = issues.map((issue) => {
      const line = startPosition.line + (issue.line || 0);
      const range = new vscode.Range(line, 0, line, Number.MAX_VALUE);

      const severity =
        {
          error: vscode.DiagnosticSeverity.Error,
          warning: vscode.DiagnosticSeverity.Warning,
          info: vscode.DiagnosticSeverity.Information,
        }[issue.severity] || vscode.DiagnosticSeverity.Warning;

      const diagnostic = new vscode.Diagnostic(range, issue.message, severity);
      diagnostic.source = "CodeVerify";
      diagnostic.code = issue.category;

      return diagnostic;
    });

    // Get or create diagnostic collection
    const collection =
      vscode.languages.createDiagnosticCollection("codeverify-interceptor");
    collection.set(document.uri, diagnostics);

    this.disposables.push(collection);
  }

  private showBlockingWarning(verification: SuggestionVerification): void {
    const errorCount = verification.issues.filter(
      (i) => i.severity === "error"
    ).length;

    vscode.window
      .showWarningMessage(
        `CodeVerify found ${errorCount} potential issues in this suggestion. ` +
          `Trust score: ${verification.score}/100`,
        "Show Details",
        "Accept Anyway",
        "Reject"
      )
      .then((selection) => {
        if (selection === "Show Details") {
          this.showVerificationDetails(verification.suggestionId);
        } else if (selection === "Reject") {
          vscode.commands.executeCommand("undo");
        }
      });
  }

  private async verifySelection(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage("No active editor");
      return;
    }

    const selection = editor.selection;
    if (selection.isEmpty) {
      vscode.window.showWarningMessage("No text selected");
      return;
    }

    const selectedText = editor.document.getText(selection);
    await this.verifySuggestion(
      selectedText,
      editor.document,
      selection.start
    );
  }

  private showVerificationDetails(suggestionId: string): void {
    const verification = this.pendingVerifications.get(suggestionId);
    if (!verification) {
      vscode.window.showWarningMessage("Verification not found");
      return;
    }

    // Create webview with details
    const panel = vscode.window.createWebviewPanel(
      "codeverifyDetails",
      "Verification Details",
      vscode.ViewColumn.Beside,
      { enableScripts: true }
    );

    panel.webview.html = this.getDetailsHtml(verification);
  }

  private getDetailsHtml(verification: SuggestionVerification): string {
    const issuesHtml = verification.issues
      .map(
        (issue) => `
        <div class="issue ${issue.severity}">
          <h4>${issue.severity.toUpperCase()}: ${issue.category}</h4>
          <p>${issue.message}</p>
          ${issue.fix ? `<pre><code>${this.escapeHtml(issue.fix)}</code></pre>` : ""}
        </div>
      `
      )
      .join("");

    return `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: var(--vscode-font-family); padding: 20px; }
          .status { font-size: 24px; margin-bottom: 20px; }
          .score { font-size: 18px; color: var(--vscode-descriptionForeground); }
          .issue { padding: 10px; margin: 10px 0; border-radius: 4px; }
          .issue.error { background: rgba(244, 67, 54, 0.1); border-left: 3px solid #F44336; }
          .issue.warning { background: rgba(255, 152, 0, 0.1); border-left: 3px solid #FF9800; }
          .issue.info { background: rgba(33, 150, 243, 0.1); border-left: 3px solid #2196F3; }
          pre { background: var(--vscode-textBlockQuote-background); padding: 10px; overflow-x: auto; }
        </style>
      </head>
      <body>
        <h1>Verification Details</h1>
        <div class="status">Status: ${verification.status}</div>
        <div class="score">Trust Score: ${verification.score}/100</div>
        <div class="score">Verification Time: ${verification.verificationTimeMs}ms</div>
        
        <h2>Issues (${verification.issues.length})</h2>
        ${issuesHtml || "<p>No issues found</p>"}
      </body>
      </html>
    `;
  }

  private escapeHtml(text: string): string {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  private async applySuggestedFix(issue: VerificationIssue): Promise<void> {
    if (!issue.fix) {
      vscode.window.showWarningMessage("No fix available for this issue");
      return;
    }

    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    // Apply the fix at the issue location
    const position = new vscode.Position(issue.line, issue.column);
    const edit = new vscode.WorkspaceEdit();

    // This is a simplified fix application - in practice would need
    // more sophisticated diff application
    edit.insert(editor.document.uri, position, issue.fix);

    await vscode.workspace.applyEdit(edit);
    vscode.window.showInformationMessage("Fix applied");
  }

  private updateStatusBar(): void {
    if (this.config.enabled) {
      this.statusBarItem.text = "$(shield) CodeVerify";
      this.statusBarItem.tooltip = "CodeVerify Interceptor (Click to toggle)";
      this.statusBarItem.backgroundColor = undefined;
    } else {
      this.statusBarItem.text = "$(shield) CodeVerify (Off)";
      this.statusBarItem.tooltip =
        "CodeVerify Interceptor Disabled (Click to enable)";
      this.statusBarItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground"
      );
    }
  }

  private log(message: string): void {
    const timestamp = new Date().toISOString();
    this.outputChannel.appendLine(`[${timestamp}] ${message}`);
  }

  public dispose(): void {
    for (const decorationType of Object.values(decorationTypes)) {
      decorationType.dispose();
    }
    this.statusBarItem.dispose();
    this.outputChannel.dispose();
    for (const disposable of this.disposables) {
      disposable.dispose();
    }
  }
}
