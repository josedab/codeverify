/**
 * Copilot Native Plugin - Enhanced GitHub Copilot integration with verification
 * 
 * Features:
 * - Intercepts Copilot suggestions before insertion
 * - Real-time verification with inline badges
 * - One-click fix flow with verified alternatives
 * - Trust score display with risk assessment
 * - Smart throttling to avoid verification fatigue
 */

import * as vscode from 'vscode';

/**
 * Verification badge types for inline display
 */
export enum VerificationBadge {
    Verified = 'verified',
    Caution = 'caution',
    Warning = 'warning',
    Error = 'error',
    Pending = 'pending',
    Skipped = 'skipped',
}

/**
 * A verified fix suggestion
 */
export interface VerifiedFix {
    id: string;
    originalCode: string;
    fixedCode: string;
    description: string;
    confidence: number;
    proofSummary?: string;
    issues: FixedIssue[];
}

/**
 * Issue that was fixed
 */
export interface FixedIssue {
    title: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    category: string;
    lineNumber?: number;
}

/**
 * Trust assessment for code
 */
export interface TrustAssessment {
    score: number;
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    confidence: number;
    factors: TrustFactor[];
    recommendation: string;
}

/**
 * Factor contributing to trust score
 */
export interface TrustFactor {
    name: string;
    score: number;
    weight: number;
    description: string;
}

/**
 * Configuration for the native plugin
 */
export interface NativePluginConfig {
    enabled: boolean;
    verifyBeforeInsert: boolean;
    showBadges: boolean;
    showTrustScore: boolean;
    autoFixOnInsert: boolean;
    minTrustScoreForAutoAccept: number;
    throttleMs: number;
    maxConcurrentVerifications: number;
}

/**
 * Verification result from backend
 */
export interface PluginVerificationResult {
    code: string;
    badge: VerificationBadge;
    trustAssessment: TrustAssessment;
    issues: PluginIssue[];
    fixes: VerifiedFix[];
    verificationTimeMs: number;
    proofs: ProofSummary[];
}

/**
 * Issue found in code
 */
export interface PluginIssue {
    id: string;
    title: string;
    description: string;
    severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
    category: string;
    location: IssueLocation;
    fix?: VerifiedFix;
}

/**
 * Location of an issue
 */
export interface IssueLocation {
    line: number;
    column: number;
    endLine?: number;
    endColumn?: number;
}

/**
 * Summary of a formal proof
 */
export interface ProofSummary {
    checkType: string;
    status: 'proven' | 'failed' | 'timeout' | 'unknown';
    description: string;
    timeMs: number;
}

/**
 * Backend client interface
 */
export interface PluginBackendClient {
    verify(code: string, context: VerificationContext): Promise<PluginVerificationResult>;
    generateFixes(code: string, issues: PluginIssue[]): Promise<VerifiedFix[]>;
    getTrustScore(code: string, context: VerificationContext): Promise<TrustAssessment>;
}

/**
 * Context for verification
 */
export interface VerificationContext {
    filePath: string;
    language: string;
    surroundingCode: string;
    position: { line: number; column: number };
    projectContext?: string;
}

/**
 * Badge decoration configurations
 */
const badgeDecorations: Record<VerificationBadge, vscode.DecorationRenderOptions> = {
    [VerificationBadge.Verified]: {
        after: {
            contentText: ' ✓ Verified',
            color: '#4CAF50',
            backgroundColor: 'rgba(76, 175, 80, 0.15)',
            border: '1px solid rgba(76, 175, 80, 0.3)',
            borderRadius: '3px',
            margin: '0 0 0 8px',
            fontWeight: 'normal',
            fontSize: '11px',
        },
        isWholeLine: false,
    },
    [VerificationBadge.Caution]: {
        after: {
            contentText: ' ⚡ Caution',
            color: '#2196F3',
            backgroundColor: 'rgba(33, 150, 243, 0.15)',
            border: '1px solid rgba(33, 150, 243, 0.3)',
            borderRadius: '3px',
            margin: '0 0 0 8px',
            fontWeight: 'normal',
            fontSize: '11px',
        },
        isWholeLine: false,
    },
    [VerificationBadge.Warning]: {
        after: {
            contentText: ' ⚠ Warning',
            color: '#FF9800',
            backgroundColor: 'rgba(255, 152, 0, 0.15)',
            border: '1px solid rgba(255, 152, 0, 0.3)',
            borderRadius: '3px',
            margin: '0 0 0 8px',
            fontWeight: 'normal',
            fontSize: '11px',
        },
        isWholeLine: false,
    },
    [VerificationBadge.Error]: {
        after: {
            contentText: ' ✗ Issues Found',
            color: '#F44336',
            backgroundColor: 'rgba(244, 67, 54, 0.15)',
            border: '1px solid rgba(244, 67, 54, 0.3)',
            borderRadius: '3px',
            margin: '0 0 0 8px',
            fontWeight: 'normal',
            fontSize: '11px',
        },
        isWholeLine: false,
    },
    [VerificationBadge.Pending]: {
        after: {
            contentText: ' ⏳ Verifying...',
            color: '#9E9E9E',
            backgroundColor: 'rgba(158, 158, 158, 0.1)',
            borderRadius: '3px',
            margin: '0 0 0 8px',
            fontWeight: 'normal',
            fontSize: '11px',
        },
        isWholeLine: false,
    },
    [VerificationBadge.Skipped]: {
        after: {
            contentText: ' ○ Skipped',
            color: '#757575',
            margin: '0 0 0 8px',
            fontWeight: 'normal',
            fontSize: '11px',
        },
        isWholeLine: false,
    },
};

/**
 * Smart throttler for verification requests
 */
class VerificationThrottler {
    private pending: Map<string, NodeJS.Timeout> = new Map();
    private activeCount = 0;

    constructor(
        private throttleMs: number,
        private maxConcurrent: number
    ) {}

    async throttle<T>(
        key: string,
        fn: () => Promise<T>
    ): Promise<T | null> {
        // Cancel any pending request for this key
        const existing = this.pending.get(key);
        if (existing) {
            clearTimeout(existing);
        }

        // Wait if at max concurrent
        if (this.activeCount >= this.maxConcurrent) {
            return null;
        }

        return new Promise((resolve) => {
            const timeout = setTimeout(async () => {
                this.pending.delete(key);
                this.activeCount++;
                try {
                    const result = await fn();
                    resolve(result);
                } catch {
                    resolve(null);
                } finally {
                    this.activeCount--;
                }
            }, this.throttleMs);

            this.pending.set(key, timeout);
        });
    }

    clear(): void {
        for (const timeout of this.pending.values()) {
            clearTimeout(timeout);
        }
        this.pending.clear();
    }
}

/**
 * One-click fix provider
 */
class OneClickFixProvider implements vscode.CodeActionProvider {
    private fixes: Map<string, VerifiedFix[]> = new Map();

    provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
    ): vscode.CodeAction[] {
        const key = `${document.uri.toString()}:${range.start.line}`;
        const fixes = this.fixes.get(key);

        if (!fixes || fixes.length === 0) {
            return [];
        }

        return fixes.map((fix, index) => {
            const action = new vscode.CodeAction(
                `✨ Apply verified fix: ${fix.description}`,
                vscode.CodeActionKind.QuickFix
            );
            action.command = {
                command: 'codeverify.applyVerifiedFix',
                title: 'Apply Verified Fix',
                arguments: [document.uri, range, fix],
            };
            action.isPreferred = index === 0;
            action.diagnostics = context.diagnostics;
            return action;
        });
    }

    registerFixes(uri: vscode.Uri, line: number, fixes: VerifiedFix[]): void {
        const key = `${uri.toString()}:${line}`;
        this.fixes.set(key, fixes);
    }

    clearFixes(uri: vscode.Uri): void {
        for (const key of this.fixes.keys()) {
            if (key.startsWith(uri.toString())) {
                this.fixes.delete(key);
            }
        }
    }
}

/**
 * Copilot Native Plugin - Main class
 */
export class CopilotNativePlugin implements vscode.Disposable {
    private client: PluginBackendClient;
    private config: NativePluginConfig;
    private disposables: vscode.Disposable[] = [];
    private throttler: VerificationThrottler;
    private fixProvider: OneClickFixProvider;
    private decorationTypes: Map<VerificationBadge, vscode.TextEditorDecorationType> = new Map();
    private statusBarItem: vscode.StatusBarItem;
    private outputChannel: vscode.OutputChannel;
    private activeDecorations: Map<string, { type: vscode.TextEditorDecorationType; ranges: vscode.Range[] }[]> = new Map();
    private verificationResults: Map<string, PluginVerificationResult> = new Map();

    constructor(client: PluginBackendClient) {
        this.client = client;
        this.config = this.getDefaultConfig();
        this.outputChannel = vscode.window.createOutputChannel('CodeVerify Copilot Plugin');
        this.throttler = new VerificationThrottler(
            this.config.throttleMs,
            this.config.maxConcurrentVerifications
        );
        this.fixProvider = new OneClickFixProvider();

        // Create decoration types
        for (const [badge, options] of Object.entries(badgeDecorations)) {
            this.decorationTypes.set(
                badge as VerificationBadge,
                vscode.window.createTextEditorDecorationType(options)
            );
        }

        // Create status bar
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            99
        );
        this.statusBarItem.command = 'codeverify.showCopilotStatus';
        this.updateStatusBar(null);
        this.statusBarItem.show();

        this.initialize();
    }

    private getDefaultConfig(): NativePluginConfig {
        return {
            enabled: true,
            verifyBeforeInsert: true,
            showBadges: true,
            showTrustScore: true,
            autoFixOnInsert: false,
            minTrustScoreForAutoAccept: 80,
            throttleMs: 500,
            maxConcurrentVerifications: 3,
        };
    }

    private initialize(): void {
        // Register code action provider for fixes
        this.disposables.push(
            vscode.languages.registerCodeActionsProvider(
                { pattern: '**' },
                this.fixProvider,
                { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] }
            )
        );

        // Register text change listener
        this.disposables.push(
            vscode.workspace.onDidChangeTextDocument((event) => {
                if (this.config.enabled) {
                    this.handleDocumentChange(event);
                }
            })
        );

        // Register commands
        this.registerCommands();

        // Watch for editor changes
        this.disposables.push(
            vscode.window.onDidChangeActiveTextEditor((editor) => {
                if (editor) {
                    this.refreshDecorations(editor);
                }
            })
        );

        this.log('Copilot Native Plugin initialized');
    }

    private registerCommands(): void {
        // Toggle plugin
        this.disposables.push(
            vscode.commands.registerCommand('codeverify.toggleCopilotPlugin', () => {
                this.config.enabled = !this.config.enabled;
                this.updateStatusBar(null);
                vscode.window.showInformationMessage(
                    `CodeVerify Copilot Plugin ${this.config.enabled ? 'enabled' : 'disabled'}`
                );
            })
        );

        // Apply verified fix
        this.disposables.push(
            vscode.commands.registerCommand(
                'codeverify.applyVerifiedFix',
                async (uri: vscode.Uri, range: vscode.Range, fix: VerifiedFix) => {
                    await this.applyFix(uri, range, fix);
                }
            )
        );

        // Show Copilot status
        this.disposables.push(
            vscode.commands.registerCommand('codeverify.showCopilotStatus', () => {
                this.showStatusPanel();
            })
        );

        // Verify current suggestion
        this.disposables.push(
            vscode.commands.registerCommand('codeverify.verifyCurrentCode', async () => {
                await this.verifyCurrentSelection();
            })
        );

        // Show alternative fixes
        this.disposables.push(
            vscode.commands.registerCommand('codeverify.showAlternativeFixes', async () => {
                await this.showAlternativeFixes();
            })
        );

        // Accept with fix
        this.disposables.push(
            vscode.commands.registerCommand('codeverify.acceptWithFix', async () => {
                await this.acceptWithBestFix();
            })
        );
    }

    private async handleDocumentChange(event: vscode.TextDocumentChangeEvent): Promise<void> {
        for (const change of event.contentChanges) {
            // Detect potential Copilot insertion
            if (this.isPotentialCopilotInsertion(change)) {
                const key = this.getChangeKey(event.document, change.range.start);
                
                await this.throttler.throttle(key, async () => {
                    await this.verifyInsertion(
                        event.document,
                        change.range.start,
                        change.text
                    );
                });
            }
        }
    }

    private isPotentialCopilotInsertion(change: vscode.TextDocumentContentChangeEvent): boolean {
        const text = change.text;
        
        // Heuristics for Copilot suggestions:
        // - Multi-line or substantial single-line insertion
        // - Contains code patterns (functions, classes, control flow)
        // - Not just whitespace or single characters
        
        if (text.length < 15) return false;
        if (/^\s*$/.test(text)) return false;

        const codePatterns = [
            /\b(function|def|class|const|let|var|if|for|while|return|async|await)\b/,
            /[={}()[\];]/,
            /=>/,
        ];

        return codePatterns.some(pattern => pattern.test(text));
    }

    private getChangeKey(document: vscode.TextDocument, position: vscode.Position): string {
        return `${document.uri.toString()}:${position.line}:${position.character}`;
    }

    private async verifyInsertion(
        document: vscode.TextDocument,
        position: vscode.Position,
        code: string
    ): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document !== document) return;

        // Show pending badge
        this.showBadge(editor, position, VerificationBadge.Pending);

        try {
            const context = this.buildVerificationContext(document, position);
            const result = await this.client.verify(code, context);

            // Store result
            const key = this.getChangeKey(document, position);
            this.verificationResults.set(key, result);

            // Update badge
            this.showBadge(editor, position, result.badge);

            // Update status bar
            this.updateStatusBar(result.trustAssessment);

            // Register fixes if available
            if (result.fixes.length > 0) {
                this.fixProvider.registerFixes(document.uri, position.line, result.fixes);
            }

            // Show diagnostics for issues
            if (result.issues.length > 0) {
                this.showDiagnostics(document, position, result.issues);
            }

            // Handle auto-fix if configured
            if (
                this.config.autoFixOnInsert &&
                result.badge === VerificationBadge.Error &&
                result.fixes.length > 0
            ) {
                await this.promptForAutoFix(editor, position, result);
            }

            this.log(
                `Verified: ${result.badge} (trust: ${result.trustAssessment.score}, ` +
                `issues: ${result.issues.length}, fixes: ${result.fixes.length})`
            );
        } catch (error) {
            this.log(`Verification error: ${error}`);
            this.showBadge(editor, position, VerificationBadge.Skipped);
        }
    }

    private buildVerificationContext(
        document: vscode.TextDocument,
        position: vscode.Position
    ): VerificationContext {
        const surroundingLines = 15;
        const startLine = Math.max(0, position.line - surroundingLines);
        const endLine = Math.min(document.lineCount - 1, position.line + surroundingLines);
        
        const surroundingCode = document.getText(
            new vscode.Range(startLine, 0, endLine, Number.MAX_VALUE)
        );

        return {
            filePath: document.uri.fsPath,
            language: document.languageId,
            surroundingCode,
            position: { line: position.line, column: position.character },
        };
    }

    private showBadge(
        editor: vscode.TextEditor,
        position: vscode.Position,
        badge: VerificationBadge
    ): void {
        if (!this.config.showBadges) return;

        const key = editor.document.uri.toString();
        
        // Clear existing decorations for this document
        const existing = this.activeDecorations.get(key) || [];
        for (const dec of existing) {
            editor.setDecorations(dec.type, []);
        }

        // Add new decoration
        const decorationType = this.decorationTypes.get(badge);
        if (!decorationType) return;

        const range = new vscode.Range(position.line, 0, position.line, Number.MAX_VALUE);
        editor.setDecorations(decorationType, [range]);

        this.activeDecorations.set(key, [{ type: decorationType, ranges: [range] }]);
    }

    private refreshDecorations(editor: vscode.TextEditor): void {
        const key = editor.document.uri.toString();
        const decorations = this.activeDecorations.get(key);
        
        if (decorations) {
            for (const dec of decorations) {
                editor.setDecorations(dec.type, dec.ranges);
            }
        }
    }

    private updateStatusBar(trust: TrustAssessment | null): void {
        if (!this.config.enabled) {
            this.statusBarItem.text = '$(shield) CodeVerify (Off)';
            this.statusBarItem.tooltip = 'Click to enable CodeVerify Copilot Plugin';
            this.statusBarItem.backgroundColor = new vscode.ThemeColor(
                'statusBarItem.warningBackground'
            );
            return;
        }

        if (!trust) {
            this.statusBarItem.text = '$(shield) CodeVerify';
            this.statusBarItem.tooltip = 'CodeVerify Copilot Plugin active';
            this.statusBarItem.backgroundColor = undefined;
            return;
        }

        const icon = this.getTrustIcon(trust.riskLevel);
        this.statusBarItem.text = `${icon} Trust: ${trust.score}`;
        this.statusBarItem.tooltip = new vscode.MarkdownString(
            this.buildTrustTooltip(trust),
            true
        );
        this.statusBarItem.backgroundColor = this.getTrustBackground(trust.riskLevel);
    }

    private getTrustIcon(riskLevel: string): string {
        switch (riskLevel) {
            case 'low': return '$(pass)';
            case 'medium': return '$(warning)';
            case 'high': return '$(warning)';
            case 'critical': return '$(error)';
            default: return '$(shield)';
        }
    }

    private getTrustBackground(riskLevel: string): vscode.ThemeColor | undefined {
        switch (riskLevel) {
            case 'high':
            case 'critical':
                return new vscode.ThemeColor('statusBarItem.warningBackground');
            default:
                return undefined;
        }
    }

    private buildTrustTooltip(trust: TrustAssessment): string {
        const lines = [
            `## Trust Score: ${trust.score}/100`,
            `**Risk Level:** ${trust.riskLevel.toUpperCase()}`,
            `**Confidence:** ${Math.round(trust.confidence * 100)}%`,
            '',
            '### Factors:',
        ];

        for (const factor of trust.factors) {
            const bar = '█'.repeat(Math.round(factor.score / 10)) + 
                       '░'.repeat(10 - Math.round(factor.score / 10));
            lines.push(`- ${factor.name}: ${bar} ${factor.score}`);
        }

        if (trust.recommendation) {
            lines.push('', `💡 **Recommendation:** ${trust.recommendation}`);
        }

        return lines.join('\n');
    }

    private showDiagnostics(
        document: vscode.TextDocument,
        basePosition: vscode.Position,
        issues: PluginIssue[]
    ): void {
        const collection = vscode.languages.createDiagnosticCollection('codeverify-copilot');
        this.disposables.push(collection);

        const diagnostics: vscode.Diagnostic[] = issues.map(issue => {
            const line = basePosition.line + issue.location.line;
            const range = new vscode.Range(
                line,
                issue.location.column,
                issue.location.endLine ? basePosition.line + issue.location.endLine : line,
                issue.location.endColumn || Number.MAX_VALUE
            );

            const severity = this.mapSeverity(issue.severity);
            const diagnostic = new vscode.Diagnostic(range, issue.description, severity);
            diagnostic.source = 'CodeVerify';
            diagnostic.code = issue.category;

            return diagnostic;
        });

        collection.set(document.uri, diagnostics);
    }

    private mapSeverity(severity: string): vscode.DiagnosticSeverity {
        switch (severity) {
            case 'critical':
            case 'high':
                return vscode.DiagnosticSeverity.Error;
            case 'medium':
                return vscode.DiagnosticSeverity.Warning;
            case 'low':
            case 'info':
                return vscode.DiagnosticSeverity.Information;
            default:
                return vscode.DiagnosticSeverity.Warning;
        }
    }

    private async promptForAutoFix(
        editor: vscode.TextEditor,
        position: vscode.Position,
        result: PluginVerificationResult
    ): Promise<void> {
        const bestFix = result.fixes[0];
        
        const choice = await vscode.window.showWarningMessage(
            `CodeVerify found ${result.issues.length} issues. Apply verified fix?`,
            { modal: false },
            'Apply Fix',
            'Show Options',
            'Ignore'
        );

        if (choice === 'Apply Fix') {
            const range = new vscode.Range(position, position.translate(0, result.code.length));
            await this.applyFix(editor.document.uri, range, bestFix);
        } else if (choice === 'Show Options') {
            await this.showAlternativeFixes();
        }
    }

    private async applyFix(
        uri: vscode.Uri,
        range: vscode.Range,
        fix: VerifiedFix
    ): Promise<void> {
        const edit = new vscode.WorkspaceEdit();
        
        // Find the range containing the original code
        const document = await vscode.workspace.openTextDocument(uri);
        const originalText = document.getText();
        const startOffset = document.offsetAt(range.start);
        
        // Find where the original code exists
        const originalIndex = originalText.indexOf(fix.originalCode, startOffset);
        if (originalIndex === -1) {
            // Code may have been modified, try to apply at current position
            edit.replace(uri, range, fix.fixedCode);
        } else {
            const fixRange = new vscode.Range(
                document.positionAt(originalIndex),
                document.positionAt(originalIndex + fix.originalCode.length)
            );
            edit.replace(uri, fixRange, fix.fixedCode);
        }

        const success = await vscode.workspace.applyEdit(edit);
        if (success) {
            vscode.window.showInformationMessage(
                `✨ Applied verified fix: ${fix.description} (confidence: ${Math.round(fix.confidence * 100)}%)`
            );
            this.log(`Applied fix: ${fix.id}`);
        } else {
            vscode.window.showErrorMessage('Failed to apply fix');
        }
    }

    private async verifyCurrentSelection(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        const code = selection.isEmpty
            ? editor.document.lineAt(selection.active.line).text
            : editor.document.getText(selection);

        if (!code.trim()) {
            vscode.window.showWarningMessage('No code to verify');
            return;
        }

        await this.verifyInsertion(editor.document, selection.start, code);
    }

    private async showAlternativeFixes(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const key = this.getChangeKey(editor.document, editor.selection.start);
        const result = this.verificationResults.get(key);

        if (!result || result.fixes.length === 0) {
            vscode.window.showInformationMessage('No fixes available for current code');
            return;
        }

        const items = result.fixes.map(fix => ({
            label: fix.description,
            description: `${Math.round(fix.confidence * 100)}% confidence`,
            detail: fix.issues.map(i => `• ${i.title}`).join(', '),
            fix,
        }));

        const selected = await vscode.window.showQuickPick(items, {
            title: 'Select a Verified Fix',
            placeHolder: 'Choose a fix to apply',
        });

        if (selected) {
            const range = editor.selection;
            await this.applyFix(editor.document.uri, range, selected.fix);
        }
    }

    private async acceptWithBestFix(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const key = this.getChangeKey(editor.document, editor.selection.start);
        const result = this.verificationResults.get(key);

        if (!result || result.fixes.length === 0) {
            vscode.window.showInformationMessage('No fixes available');
            return;
        }

        // Apply best fix (highest confidence)
        const bestFix = result.fixes.reduce((best, current) => 
            current.confidence > best.confidence ? current : best
        );

        await this.applyFix(editor.document.uri, editor.selection, bestFix);
    }

    private showStatusPanel(): void {
        const panel = vscode.window.createWebviewPanel(
            'codeverifyStatus',
            'CodeVerify Copilot Status',
            vscode.ViewColumn.Beside,
            { enableScripts: true }
        );

        panel.webview.html = this.buildStatusPanelHtml();
    }

    private buildStatusPanelHtml(): string {
        const recentResults = Array.from(this.verificationResults.entries())
            .slice(-10)
            .reverse();

        const resultsHtml = recentResults.map(([key, result]) => `
            <div class="result ${result.badge}">
                <div class="badge">${this.getBadgeEmoji(result.badge)} ${result.badge}</div>
                <div class="score">Trust: ${result.trustAssessment.score}/100</div>
                <div class="issues">${result.issues.length} issues, ${result.fixes.length} fixes</div>
                <div class="time">${result.verificationTimeMs}ms</div>
            </div>
        `).join('');

        return `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { 
                        font-family: var(--vscode-font-family); 
                        padding: 20px;
                        color: var(--vscode-foreground);
                    }
                    h1 { border-bottom: 1px solid var(--vscode-panel-border); padding-bottom: 10px; }
                    .result { 
                        padding: 12px; 
                        margin: 10px 0; 
                        border-radius: 6px;
                        background: var(--vscode-editor-background);
                        border: 1px solid var(--vscode-panel-border);
                    }
                    .result.verified { border-left: 4px solid #4CAF50; }
                    .result.warning { border-left: 4px solid #FF9800; }
                    .result.error { border-left: 4px solid #F44336; }
                    .badge { font-weight: bold; margin-bottom: 5px; }
                    .score { color: var(--vscode-descriptionForeground); }
                    .issues { font-size: 0.9em; }
                    .time { font-size: 0.8em; color: var(--vscode-descriptionForeground); }
                    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
                    .stat { 
                        padding: 15px; 
                        background: var(--vscode-editor-background);
                        border-radius: 8px;
                        text-align: center;
                    }
                    .stat-value { font-size: 24px; font-weight: bold; }
                    .stat-label { font-size: 12px; color: var(--vscode-descriptionForeground); }
                </style>
            </head>
            <body>
                <h1>🛡️ CodeVerify Copilot Plugin</h1>
                
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">${recentResults.length}</div>
                        <div class="stat-label">Verifications</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${recentResults.filter(([, r]) => r.badge === VerificationBadge.Verified).length}</div>
                        <div class="stat-label">Verified</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${recentResults.reduce((sum, [, r]) => sum + r.fixes.length, 0)}</div>
                        <div class="stat-label">Fixes Available</div>
                    </div>
                </div>

                <h2>Recent Verifications</h2>
                ${resultsHtml || '<p>No recent verifications</p>'}
            </body>
            </html>
        `;
    }

    private getBadgeEmoji(badge: VerificationBadge): string {
        switch (badge) {
            case VerificationBadge.Verified: return '✓';
            case VerificationBadge.Caution: return '⚡';
            case VerificationBadge.Warning: return '⚠';
            case VerificationBadge.Error: return '✗';
            case VerificationBadge.Pending: return '⏳';
            case VerificationBadge.Skipped: return '○';
        }
    }

    private log(message: string): void {
        const timestamp = new Date().toISOString();
        this.outputChannel.appendLine(`[${timestamp}] ${message}`);
    }

    public dispose(): void {
        this.throttler.clear();
        for (const decorationType of this.decorationTypes.values()) {
            decorationType.dispose();
        }
        this.statusBarItem.dispose();
        this.outputChannel.dispose();
        for (const disposable of this.disposables) {
            disposable.dispose();
        }
    }
}

/**
 * Mock backend client for testing
 */
export class MockPluginBackendClient implements PluginBackendClient {
    async verify(code: string, context: VerificationContext): Promise<PluginVerificationResult> {
        await new Promise(resolve => setTimeout(resolve, 300));

        const issues: PluginIssue[] = [];
        const fixes: VerifiedFix[] = [];

        // Detect common issues
        if (code.includes('eval(')) {
            issues.push({
                id: 'security-eval',
                title: 'Unsafe eval() usage',
                description: 'eval() can execute arbitrary code and is a security risk',
                severity: 'critical',
                category: 'security',
                location: { line: 0, column: code.indexOf('eval(') },
            });
            fixes.push({
                id: 'fix-eval',
                originalCode: code,
                fixedCode: code.replace(/eval\(([^)]+)\)/g, 'JSON.parse($1)'),
                description: 'Replace eval() with JSON.parse()',
                confidence: 0.85,
                proofSummary: 'Verified: JSON.parse is safe for this context',
                issues: [{ title: 'Removed eval()', severity: 'critical', category: 'security' }],
            });
        }

        if (/\[.*\]/.test(code) && !code.includes('length') && !code.includes('?.')) {
            issues.push({
                id: 'bounds-array',
                title: 'Potential array bounds issue',
                description: 'Array access without bounds checking',
                severity: 'medium',
                category: 'bounds',
                location: { line: 0, column: 0 },
            });
        }

        const badge = issues.some(i => i.severity === 'critical')
            ? VerificationBadge.Error
            : issues.some(i => i.severity === 'high' || i.severity === 'medium')
                ? VerificationBadge.Warning
                : issues.length > 0
                    ? VerificationBadge.Caution
                    : VerificationBadge.Verified;

        const score = Math.max(0, 100 - issues.length * 15);

        return {
            code,
            badge,
            trustAssessment: {
                score,
                riskLevel: score >= 80 ? 'low' : score >= 60 ? 'medium' : score >= 40 ? 'high' : 'critical',
                confidence: 0.9,
                factors: [
                    { name: 'Security', score: issues.some(i => i.category === 'security') ? 30 : 90, weight: 0.3, description: '' },
                    { name: 'Correctness', score: issues.some(i => i.category === 'bounds') ? 60 : 95, weight: 0.3, description: '' },
                    { name: 'Quality', score: 85, weight: 0.2, description: '' },
                    { name: 'AI Patterns', score: 75, weight: 0.2, description: '' },
                ],
                recommendation: issues.length > 0 ? 'Review flagged issues before accepting' : 'Code appears safe to use',
            },
            issues,
            fixes,
            verificationTimeMs: 300,
            proofs: [
                { checkType: 'null_safety', status: 'proven', description: 'No null dereferences found', timeMs: 50 },
                { checkType: 'bounds', status: issues.some(i => i.category === 'bounds') ? 'failed' : 'proven', description: 'Array bounds check', timeMs: 80 },
            ],
        };
    }

    async generateFixes(code: string, issues: PluginIssue[]): Promise<VerifiedFix[]> {
        // Mock implementation
        return [];
    }

    async getTrustScore(code: string, context: VerificationContext): Promise<TrustAssessment> {
        return {
            score: 85,
            riskLevel: 'low',
            confidence: 0.9,
            factors: [],
            recommendation: '',
        };
    }
}
