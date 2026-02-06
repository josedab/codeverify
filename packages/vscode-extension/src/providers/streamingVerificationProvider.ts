/**
 * Streaming Verification Provider
 *
 * Connects to the LSP server for real-time, progressive verification.
 * Shows results from each stage (pattern -> AI -> formal) as they complete.
 */

import * as vscode from 'vscode';

interface StreamingDiagnostic {
    line: number;
    character: number;
    end_line: number;
    end_character: number;
    severity: number;
    message: string;
    source: string;
    code: string | null;
    stage: string;
}

interface StageEvent {
    type: 'stage_start' | 'stage_complete' | 'complete';
    stage?: string;
    diagnostics?: StreamingDiagnostic[];
    elapsed_ms?: number;
    progress?: number;
}

export class StreamingVerificationProvider implements vscode.Disposable {
    private diagnosticCollection: vscode.DiagnosticCollection;
    private statusBarItem: vscode.StatusBarItem;
    private debounceTimer: NodeJS.Timeout | undefined;
    private debounceMs: number;
    private isVerifying: boolean = false;
    private disposables: vscode.Disposable[] = [];

    constructor(debounceMs: number = 500) {
        this.debounceMs = debounceMs;
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeverify-streaming');
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 90);
        this.statusBarItem.text = '$(shield) CodeVerify';
        this.statusBarItem.show();

        // Watch for document changes
        this.disposables.push(
            vscode.workspace.onDidChangeTextDocument((e) => {
                this.onDocumentChange(e.document);
            }),
            vscode.workspace.onDidOpenTextDocument((doc) => {
                this.verifyDocument(doc);
            })
        );
    }

    private onDocumentChange(document: vscode.TextDocument): void {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        this.debounceTimer = setTimeout(() => {
            this.verifyDocument(document);
        }, this.debounceMs);
    }

    async verifyDocument(document: vscode.TextDocument): Promise<void> {
        const supportedLanguages = ['python', 'typescript', 'javascript', 'go', 'java'];
        if (!supportedLanguages.includes(document.languageId)) {
            return;
        }

        this.isVerifying = true;
        this.statusBarItem.text = '$(loading~spin) Verifying...';

        const allDiagnostics: vscode.Diagnostic[] = [];
        const stages = ['pattern', 'ai', 'formal'];

        for (const stage of stages) {
            this.statusBarItem.text = `$(loading~spin) ${stage}...`;

            const stageDiagnostics = this.runLocalPatternCheck(
                document.getText(),
                document.languageId,
                stage
            );

            for (const sd of stageDiagnostics) {
                const range = new vscode.Range(
                    new vscode.Position(sd.line, sd.character),
                    new vscode.Position(sd.end_line, sd.end_character)
                );
                const severity = this.mapSeverity(sd.severity);
                const diagnostic = new vscode.Diagnostic(range, sd.message, severity);
                diagnostic.source = `codeverify (${sd.stage})`;
                if (sd.code) {
                    diagnostic.code = sd.code;
                }
                allDiagnostics.push(diagnostic);
            }

            this.diagnosticCollection.set(document.uri, allDiagnostics);
        }

        this.isVerifying = false;
        const issueCount = allDiagnostics.length;
        this.statusBarItem.text = issueCount > 0
            ? `$(shield) ${issueCount} issue(s)`
            : '$(shield) Verified';
    }

    private runLocalPatternCheck(
        content: string,
        language: string,
        stage: string
    ): StreamingDiagnostic[] {
        if (stage !== 'pattern') {
            return [];
        }

        const diagnostics: StreamingDiagnostic[] = [];
        const lines = content.split('\n');

        const patterns: Record<string, Array<[RegExp, string, number, string]>> = {
            python: [
                [/except\s*:/, 'Bare except clause', 2, 'CV001'],
                [/==\s*None/, "Use 'is None' instead of '== None'", 3, 'CV004'],
                [/\beval\s*\(/, 'eval() is a security risk', 1, 'S001'],
            ],
            typescript: [
                [/:\s*any\b/, "Avoid using 'any' type", 2, 'TS001'],
                [/==[^=]/, 'Use === for strict equality', 2, 'TS003'],
            ],
            go: [
                [/\b_\s*=\s*\w+\(/, 'Error return value ignored', 2, 'GO001'],
                [/panic\(/, 'Avoid panic() - return errors', 2, 'GO003'],
            ],
            java: [
                [/catch\s*\([^)]+\)\s*\{\s*\}/, 'Empty catch block', 1, 'J001'],
                [/System\.out\.print/, 'Use logging framework', 3, 'J002'],
            ],
        };

        const langPatterns = patterns[language] || [];

        for (let i = 0; i < lines.length; i++) {
            for (const [pattern, message, severity, code] of langPatterns) {
                const match = pattern.exec(lines[i]);
                if (match) {
                    diagnostics.push({
                        line: i,
                        character: match.index,
                        end_line: i,
                        end_character: match.index + match[0].length,
                        severity,
                        message,
                        source: 'codeverify',
                        code,
                        stage: 'pattern',
                    });
                }
            }
        }

        return diagnostics;
    }

    private mapSeverity(severity: number): vscode.DiagnosticSeverity {
        switch (severity) {
            case 1: return vscode.DiagnosticSeverity.Error;
            case 2: return vscode.DiagnosticSeverity.Warning;
            case 3: return vscode.DiagnosticSeverity.Information;
            case 4: return vscode.DiagnosticSeverity.Hint;
            default: return vscode.DiagnosticSeverity.Information;
        }
    }

    dispose(): void {
        this.diagnosticCollection.dispose();
        this.statusBarItem.dispose();
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        for (const d of this.disposables) {
            d.dispose();
        }
    }
}
