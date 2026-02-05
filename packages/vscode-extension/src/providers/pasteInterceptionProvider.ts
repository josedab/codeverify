/**
 * Paste Interception Provider
 *
 * Intercepts paste events to detect and verify AI-generated code before it enters
 * the codebase. Shows trust score overlay with Accept/Reject actions.
 */

import * as vscode from 'vscode';
import * as crypto from 'crypto';
import { CodeVerifyClient, Finding, TrustScore } from '../client';
import { logger } from '../logger';

/**
 * Result of paste interception analysis
 */
export interface PasteAnalysisResult {
    id: string;
    code: string;
    isAiGenerated: boolean;
    aiConfidence: number;
    trustScore: number;
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    findings: Finding[];
    detectedModel: string;
    recommendations: string[];
    analysisTimeMs: number;
}

/**
 * User decision on intercepted paste
 */
export type PasteDecision = 'accept' | 'reject' | 'accept_with_review' | 'modify';

/**
 * Configuration for paste interception
 */
export interface PasteInterceptionConfig {
    enabled: boolean;
    minCodeLength: number;
    aiConfidenceThreshold: number;
    trustScoreThreshold: number;
    autoAcceptTrustedCode: boolean;
    showOverlayForAllPastes: boolean;
    blockUntrustedCode: boolean;
}

/**
 * Statistics for paste interception
 */
interface InterceptionStatistics {
    totalInterceptions: number;
    aiDetectedCount: number;
    acceptedCount: number;
    rejectedCount: number;
    modifiedCount: number;
    averageTrustScore: number;
    averageAnalysisTimeMs: number;
}

/**
 * Cache entry for analyzed code
 */
interface CacheEntry {
    result: PasteAnalysisResult;
    timestamp: number;
}

/**
 * Provider for intercepting paste events and verifying AI-generated code
 */
export class PasteInterceptionProvider implements vscode.Disposable {
    private client: CodeVerifyClient;
    private disposables: vscode.Disposable[] = [];
    private enabled = false;
    private config: PasteInterceptionConfig;
    private analysisCache: Map<string, CacheEntry> = new Map();
    private statistics: InterceptionStatistics;
    private statusBarItem: vscode.StatusBarItem;
    private currentOverlay: vscode.WebviewPanel | null = null;
    private pendingPaste: {
        editor: vscode.TextEditor;
        selection: vscode.Selection;
        result: PasteAnalysisResult;
    } | null = null;

    // Event emitters
    private readonly _onPasteIntercepted = new vscode.EventEmitter<PasteAnalysisResult>();
    readonly onPasteIntercepted = this._onPasteIntercepted.event;

    private readonly _onPasteDecision = new vscode.EventEmitter<{ result: PasteAnalysisResult; decision: PasteDecision }>();
    readonly onPasteDecision = this._onPasteDecision.event;

    // Cache settings
    private readonly CACHE_TTL_MS = 60000; // 1 minute
    private readonly MAX_CACHE_SIZE = 100;

    constructor(client: CodeVerifyClient) {
        this.client = client;
        this.config = this.loadConfig();
        this.statistics = this.initializeStatistics();

        // Create status bar item
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            99
        );
        this.statusBarItem.command = 'codeverify.togglePasteInterception';
        this.updateStatusBar();

        // Register configuration change listener
        this.disposables.push(
            vscode.workspace.onDidChangeConfiguration(e => {
                if (e.affectsConfiguration('codeverify.pasteInterception')) {
                    this.config = this.loadConfig();
                    this.updateStatusBar();
                }
            })
        );
    }

    /**
     * Enable paste interception
     */
    enable(): void {
        if (this.enabled) return;

        this.enabled = true;
        this.registerPasteHandler();
        this.updateStatusBar();

        logger.info('Paste interception enabled');
    }

    /**
     * Disable paste interception
     */
    disable(): void {
        if (!this.enabled) return;

        this.enabled = false;
        this.updateStatusBar();

        logger.info('Paste interception disabled');
    }

    /**
     * Toggle paste interception
     */
    toggle(): void {
        if (this.enabled) {
            this.disable();
        } else {
            this.enable();
        }
    }

    /**
     * Check if paste interception is enabled
     */
    isEnabled(): boolean {
        return this.enabled;
    }

    /**
     * Register paste command handler
     */
    private registerPasteHandler(): void {
        // Override the default paste command
        const pasteDisposable = vscode.commands.registerCommand(
            'codeverify.interceptPaste',
            async () => {
                await this.handlePaste();
            }
        );
        this.disposables.push(pasteDisposable);
    }

    /**
     * Handle paste event
     */
    async handlePaste(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            // Fall back to default paste
            await vscode.commands.executeCommand('editor.action.clipboardPasteAction');
            return;
        }

        // Check if interception is enabled
        if (!this.enabled || !this.config.enabled) {
            await vscode.commands.executeCommand('editor.action.clipboardPasteAction');
            return;
        }

        // Get clipboard content
        const clipboardText = await vscode.env.clipboard.readText();

        // Check minimum length
        if (clipboardText.length < this.config.minCodeLength) {
            await vscode.commands.executeCommand('editor.action.clipboardPasteAction');
            return;
        }

        // Check if it looks like code
        if (!this.looksLikeCode(clipboardText, editor.document.languageId)) {
            await vscode.commands.executeCommand('editor.action.clipboardPasteAction');
            return;
        }

        // Analyze the code
        const result = await this.analyzeCode(clipboardText, editor.document.languageId);

        // Update statistics
        this.updateStatistics(result);

        // Emit event
        this._onPasteIntercepted.fire(result);

        // Decision logic
        if (this.shouldAutoAccept(result)) {
            // Auto-accept trusted code
            await this.acceptPaste(editor, clipboardText, result, 'accept');
            return;
        }

        if (this.config.blockUntrustedCode && result.riskLevel === 'critical') {
            // Block critical risk code
            this.showBlockedNotification(result);
            return;
        }

        // Show overlay for user decision
        this.pendingPaste = {
            editor,
            selection: editor.selection,
            result,
        };
        this.showOverlay(result);
    }

    /**
     * Analyze code for AI generation and trust score
     */
    private async analyzeCode(code: string, language: string): Promise<PasteAnalysisResult> {
        const startTime = Date.now();
        const codeHash = this.computeHash(code);

        // Check cache
        const cached = this.getFromCache(codeHash);
        if (cached) {
            logger.debug('Using cached analysis result', { codeHash });
            return cached;
        }

        try {
            // Parallel analysis for speed
            const [trustScorePromise, findingsPromise] = await Promise.allSettled([
                this.client.getTrustScoreForCode(code, language),
                this.client.analyzeCodeQuick(code, language),
            ]);

            const trustScoreResult = trustScorePromise.status === 'fulfilled'
                ? trustScorePromise.value
                : this.getDefaultTrustScore();

            const findings = findingsPromise.status === 'fulfilled'
                ? findingsPromise.value
                : [];

            const analysisTimeMs = Date.now() - startTime;

            const result: PasteAnalysisResult = {
                id: crypto.randomUUID(),
                code,
                isAiGenerated: (trustScoreResult.ai_probability || 0) > 50,
                aiConfidence: (trustScoreResult.ai_probability || 0) / 100,
                trustScore: trustScoreResult.score || 0,
                riskLevel: this.determineRiskLevel(trustScoreResult),
                findings: findings.filter(f => f.severity === 'critical' || f.severity === 'high'),
                detectedModel: this.detectModel(code, trustScoreResult),
                recommendations: this.generateRecommendations(trustScoreResult, findings),
                analysisTimeMs,
            };

            // Cache result
            this.addToCache(codeHash, result);

            logger.info('Paste analysis completed', {
                analysisTimeMs,
                isAiGenerated: result.isAiGenerated,
                trustScore: result.trustScore,
                findingsCount: result.findings.length,
            });

            return result;

        } catch (error) {
            logger.error('Paste analysis failed', { error });

            // Return conservative result on error
            return {
                id: crypto.randomUUID(),
                code,
                isAiGenerated: false,
                aiConfidence: 0,
                trustScore: 50,
                riskLevel: 'medium',
                findings: [],
                detectedModel: 'unknown',
                recommendations: ['Analysis failed. Manual review recommended.'],
                analysisTimeMs: Date.now() - startTime,
            };
        }
    }

    /**
     * Check if text looks like code
     */
    private looksLikeCode(text: string, languageId: string): boolean {
        const lines = text.split('\n');

        // Too few lines
        if (lines.length < 2) return false;

        // Check for common code patterns
        const codePatterns = [
            /^\s*(def|function|class|const|let|var|import|from|async|export)\s/m,
            /[{}\[\]();]/,
            /^\s*(if|for|while|try|return)\s/m,
            /=>/,
            /:\s*(int|str|float|bool|string|number|void)\b/,
            /^\s*#\s*\w+/m,  // Comments
            /\/\//,  // JS comments
            /^\s*@\w+/m,  // Decorators
        ];

        let patternMatches = 0;
        for (const pattern of codePatterns) {
            if (pattern.test(text)) {
                patternMatches++;
            }
        }

        return patternMatches >= 2;
    }

    /**
     * Determine if code should be auto-accepted
     */
    private shouldAutoAccept(result: PasteAnalysisResult): boolean {
        if (!this.config.autoAcceptTrustedCode) return false;

        // Auto-accept if trust score is high and not detected as AI
        if (result.trustScore >= this.config.trustScoreThreshold && !result.isAiGenerated) {
            return true;
        }

        // Auto-accept if low AI confidence and no critical findings
        if (result.aiConfidence < 0.3 && result.findings.length === 0) {
            return true;
        }

        return false;
    }

    /**
     * Show overlay for paste decision
     */
    private showOverlay(result: PasteAnalysisResult): void {
        // Close existing overlay
        if (this.currentOverlay) {
            this.currentOverlay.dispose();
        }

        this.currentOverlay = vscode.window.createWebviewPanel(
            'codeverifyPasteInterception',
            'CodeVerify: AI Code Detected',
            {
                viewColumn: vscode.ViewColumn.Beside,
                preserveFocus: true,
            },
            {
                enableScripts: true,
                retainContextWhenHidden: false,
            }
        );

        this.currentOverlay.webview.html = this.getOverlayHtml(result);

        // Handle messages from webview
        this.currentOverlay.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.command) {
                    case 'accept':
                        await this.handleDecision('accept');
                        break;
                    case 'reject':
                        await this.handleDecision('reject');
                        break;
                    case 'acceptWithReview':
                        await this.handleDecision('accept_with_review');
                        break;
                    case 'modify':
                        await this.handleDecision('modify');
                        break;
                }
            },
            undefined,
            this.disposables
        );

        this.currentOverlay.onDidDispose(() => {
            this.currentOverlay = null;
        });
    }

    /**
     * Handle user's paste decision
     */
    private async handleDecision(decision: PasteDecision): Promise<void> {
        if (!this.pendingPaste) return;

        const { editor, result } = this.pendingPaste;

        // Emit decision event
        this._onPasteDecision.fire({ result, decision });

        switch (decision) {
            case 'accept':
                await this.acceptPaste(editor, result.code, result, decision);
                break;

            case 'accept_with_review':
                await this.acceptPaste(editor, result.code, result, decision);
                // Add TODO comment
                await this.addReviewComment(editor, result);
                break;

            case 'reject':
                vscode.window.showInformationMessage(
                    'CodeVerify: Paste rejected. Code was not inserted.'
                );
                break;

            case 'modify':
                // Show the code in a diff editor for modification
                await this.showModifyEditor(result);
                break;
        }

        // Update statistics
        this.updateDecisionStatistics(decision);

        // Cleanup
        this.pendingPaste = null;
        if (this.currentOverlay) {
            this.currentOverlay.dispose();
            this.currentOverlay = null;
        }
    }

    /**
     * Accept and insert the pasted code
     */
    private async acceptPaste(
        editor: vscode.TextEditor,
        code: string,
        result: PasteAnalysisResult,
        decision: PasteDecision
    ): Promise<void> {
        await editor.edit(editBuilder => {
            editBuilder.replace(editor.selection, code);
        });

        // Show notification based on risk level
        if (result.isAiGenerated && result.riskLevel !== 'low') {
            const icon = result.riskLevel === 'critical' ? '$(warning)' :
                         result.riskLevel === 'high' ? '$(alert)' : '$(info)';

            vscode.window.showInformationMessage(
                `${icon} CodeVerify: AI-generated code inserted (Trust: ${result.trustScore}%, ${result.findings.length} issues)`
            );
        }

        logger.info('Paste accepted', {
            decision,
            trustScore: result.trustScore,
            isAiGenerated: result.isAiGenerated,
        });
    }

    /**
     * Add review comment to inserted code
     */
    private async addReviewComment(
        editor: vscode.TextEditor,
        result: PasteAnalysisResult
    ): Promise<void> {
        const commentPrefix = this.getCommentPrefix(editor.document.languageId);
        const reviewComment = `${commentPrefix} CODEVERIFY: AI-generated code (${result.detectedModel}, trust: ${result.trustScore}%) - Review required\n`;

        const position = editor.selection.start;
        await editor.edit(editBuilder => {
            editBuilder.insert(position, reviewComment);
        });
    }

    /**
     * Show modify editor for code adjustment
     */
    private async showModifyEditor(result: PasteAnalysisResult): Promise<void> {
        // Create a temporary document with the code
        const doc = await vscode.workspace.openTextDocument({
            content: result.code,
            language: 'python', // Default, will be overridden by context
        });

        await vscode.window.showTextDocument(doc, {
            viewColumn: vscode.ViewColumn.Beside,
            preview: true,
        });

        vscode.window.showInformationMessage(
            'CodeVerify: Edit the code, then copy and paste the modified version.'
        );
    }

    /**
     * Show notification for blocked code
     */
    private showBlockedNotification(result: PasteAnalysisResult): void {
        const findings = result.findings.slice(0, 3).map(f => f.title).join(', ');

        vscode.window.showWarningMessage(
            `CodeVerify: Paste blocked due to critical issues. Trust: ${result.trustScore}%. Issues: ${findings}`,
            'View Details',
            'Allow Anyway'
        ).then(selection => {
            if (selection === 'View Details') {
                this.showOverlay(result);
            } else if (selection === 'Allow Anyway') {
                this.handleDecision('accept_with_review');
            }
        });
    }

    /**
     * Get overlay HTML
     */
    private getOverlayHtml(result: PasteAnalysisResult): string {
        const trustColor = result.trustScore >= 80 ? '#2ecc71' :
                          result.trustScore >= 60 ? '#f1c40f' :
                          result.trustScore >= 40 ? '#e67e22' : '#e74c3c';

        const riskBadgeColor = {
            'low': '#2ecc71',
            'medium': '#f1c40f',
            'high': '#e67e22',
            'critical': '#e74c3c',
        }[result.riskLevel];

        const aiIndicator = result.isAiGenerated
            ? `<div class="ai-badge">AI-Generated (${(result.aiConfidence * 100).toFixed(0)}% confidence)</div>`
            : '<div class="human-badge">Likely Human-Written</div>';

        const findingsHtml = result.findings.length > 0
            ? `<div class="findings">
                <h3>Issues Detected (${result.findings.length})</h3>
                <ul>
                    ${result.findings.slice(0, 5).map(f => `
                        <li class="finding ${f.severity}">
                            <span class="severity">${f.severity.toUpperCase()}</span>
                            <span class="title">${this.escapeHtml(f.title)}</span>
                        </li>
                    `).join('')}
                </ul>
               </div>`
            : '<div class="no-findings">No critical issues detected</div>';

        const recommendationsHtml = result.recommendations.length > 0
            ? `<div class="recommendations">
                <h3>Recommendations</h3>
                <ul>
                    ${result.recommendations.map(r => `<li>${this.escapeHtml(r)}</li>`).join('')}
                </ul>
               </div>`
            : '';

        const codePreview = result.code.length > 500
            ? result.code.substring(0, 500) + '\n... (truncated)'
            : result.code;

        return `<!DOCTYPE html>
<html>
<head>
    <style>
        :root {
            --bg: var(--vscode-editor-background);
            --fg: var(--vscode-editor-foreground);
            --border: var(--vscode-panel-border);
        }
        body {
            font-family: var(--vscode-font-family);
            padding: 20px;
            color: var(--fg);
            background: var(--bg);
        }
        h1 {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        .trust-circle {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: conic-gradient(${trustColor} ${result.trustScore * 3.6}deg, #333 0deg);
            margin: 20px auto;
        }
        .trust-inner {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--bg);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            color: ${trustColor};
        }
        .badges {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 15px 0;
        }
        .ai-badge, .human-badge, .risk-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .ai-badge {
            background: #9b59b6;
            color: white;
        }
        .human-badge {
            background: #27ae60;
            color: white;
        }
        .risk-badge {
            background: ${riskBadgeColor};
            color: white;
        }
        .model-badge {
            background: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
        }
        .findings {
            margin: 20px 0;
            padding: 15px;
            background: var(--vscode-inputValidation-errorBackground);
            border-radius: 8px;
        }
        .findings h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
        }
        .findings ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .finding {
            padding: 8px;
            margin: 4px 0;
            border-radius: 4px;
            background: rgba(0,0,0,0.2);
            display: flex;
            gap: 10px;
        }
        .finding .severity {
            font-weight: bold;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .finding.critical .severity { background: #e74c3c; color: white; }
        .finding.high .severity { background: #e67e22; color: white; }
        .finding.medium .severity { background: #f1c40f; color: black; }
        .finding.low .severity { background: #3498db; color: white; }
        .no-findings {
            padding: 15px;
            background: rgba(46, 204, 113, 0.1);
            border: 1px solid #2ecc71;
            border-radius: 8px;
            text-align: center;
            color: #2ecc71;
        }
        .recommendations {
            margin: 20px 0;
            padding: 15px;
            background: var(--vscode-inputValidation-infoBackground);
            border-radius: 8px;
        }
        .recommendations h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
        }
        .recommendations ul {
            margin: 0;
            padding-left: 20px;
        }
        .recommendations li {
            margin: 5px 0;
        }
        .code-preview {
            margin: 20px 0;
            padding: 15px;
            background: var(--vscode-textCodeBlock-background);
            border-radius: 8px;
            font-family: var(--vscode-editor-font-family);
            font-size: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 200px;
            overflow-y: auto;
        }
        .actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: opacity 0.2s;
        }
        button:hover {
            opacity: 0.9;
        }
        .btn-accept {
            background: #27ae60;
            color: white;
        }
        .btn-review {
            background: #f39c12;
            color: white;
        }
        .btn-reject {
            background: #c0392b;
            color: white;
        }
        .btn-modify {
            background: #3498db;
            color: white;
        }
        .stats {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <h1>
        <span style="font-size: 24px;">$(shield)</span>
        AI Code Interception
    </h1>

    <div style="display: flex; align-items: center; gap: 30px;">
        <div class="trust-circle">
            <div class="trust-inner">${result.trustScore}%</div>
        </div>
        <div>
            <div class="badges">
                ${aiIndicator}
                <span class="risk-badge">${result.riskLevel.toUpperCase()} RISK</span>
                ${result.detectedModel !== 'unknown' ? `<span class="model-badge">${result.detectedModel}</span>` : ''}
            </div>
        </div>
    </div>

    ${findingsHtml}
    ${recommendationsHtml}

    <details>
        <summary style="cursor: pointer; padding: 10px 0;">Code Preview</summary>
        <div class="code-preview">${this.escapeHtml(codePreview)}</div>
    </details>

    <div class="actions">
        <button class="btn-accept" onclick="decide('accept')">
            Accept Code
        </button>
        <button class="btn-review" onclick="decide('acceptWithReview')">
            Accept + Mark for Review
        </button>
        <button class="btn-modify" onclick="decide('modify')">
            Modify First
        </button>
        <button class="btn-reject" onclick="decide('reject')">
            Reject
        </button>
    </div>

    <div class="stats">
        Analysis completed in ${result.analysisTimeMs}ms
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        function decide(decision) {
            vscode.postMessage({ command: decision });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                decide('accept');
            } else if (e.key === 'Escape') {
                decide('reject');
            } else if (e.key === 'r' && e.ctrlKey) {
                decide('acceptWithReview');
            }
        });
    </script>
</body>
</html>`;
    }

    /**
     * Helper methods
     */

    private loadConfig(): PasteInterceptionConfig {
        const config = vscode.workspace.getConfiguration('codeverify.pasteInterception');
        return {
            enabled: config.get('enabled', true),
            minCodeLength: config.get('minCodeLength', 50),
            aiConfidenceThreshold: config.get('aiConfidenceThreshold', 0.6),
            trustScoreThreshold: config.get('trustScoreThreshold', 80),
            autoAcceptTrustedCode: config.get('autoAcceptTrustedCode', false),
            showOverlayForAllPastes: config.get('showOverlayForAllPastes', false),
            blockUntrustedCode: config.get('blockUntrustedCode', false),
        };
    }

    private initializeStatistics(): InterceptionStatistics {
        return {
            totalInterceptions: 0,
            aiDetectedCount: 0,
            acceptedCount: 0,
            rejectedCount: 0,
            modifiedCount: 0,
            averageTrustScore: 0,
            averageAnalysisTimeMs: 0,
        };
    }

    private updateStatistics(result: PasteAnalysisResult): void {
        this.statistics.totalInterceptions++;
        if (result.isAiGenerated) {
            this.statistics.aiDetectedCount++;
        }

        // Rolling average
        const n = this.statistics.totalInterceptions;
        this.statistics.averageTrustScore =
            (this.statistics.averageTrustScore * (n - 1) + result.trustScore) / n;
        this.statistics.averageAnalysisTimeMs =
            (this.statistics.averageAnalysisTimeMs * (n - 1) + result.analysisTimeMs) / n;
    }

    private updateDecisionStatistics(decision: PasteDecision): void {
        switch (decision) {
            case 'accept':
            case 'accept_with_review':
                this.statistics.acceptedCount++;
                break;
            case 'reject':
                this.statistics.rejectedCount++;
                break;
            case 'modify':
                this.statistics.modifiedCount++;
                break;
        }
    }

    private updateStatusBar(): void {
        if (this.enabled) {
            this.statusBarItem.text = '$(shield-check) Paste Guard';
            this.statusBarItem.tooltip = 'CodeVerify Paste Interception: Active\nClick to toggle';
            this.statusBarItem.backgroundColor = undefined;
        } else {
            this.statusBarItem.text = '$(shield) Paste Guard';
            this.statusBarItem.tooltip = 'CodeVerify Paste Interception: Disabled\nClick to enable';
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        }
        this.statusBarItem.show();
    }

    private computeHash(code: string): string {
        const normalized = code.replace(/\s+/g, ' ').trim();
        return crypto.createHash('sha256').update(normalized).digest('hex').substring(0, 16);
    }

    private getFromCache(hash: string): PasteAnalysisResult | null {
        const entry = this.analysisCache.get(hash);
        if (entry && Date.now() - entry.timestamp < this.CACHE_TTL_MS) {
            return entry.result;
        }
        this.analysisCache.delete(hash);
        return null;
    }

    private addToCache(hash: string, result: PasteAnalysisResult): void {
        // Evict oldest entries if cache is full
        if (this.analysisCache.size >= this.MAX_CACHE_SIZE) {
            const oldestKey = this.analysisCache.keys().next().value;
            if (oldestKey) {
                this.analysisCache.delete(oldestKey);
            }
        }
        this.analysisCache.set(hash, { result, timestamp: Date.now() });
    }

    private determineRiskLevel(trustScore: TrustScore): 'low' | 'medium' | 'high' | 'critical' {
        const score = trustScore.score || 0;
        const riskLevel = trustScore.risk_level;

        if (riskLevel) {
            return riskLevel as 'low' | 'medium' | 'high' | 'critical';
        }

        if (score >= 80) return 'low';
        if (score >= 60) return 'medium';
        if (score >= 40) return 'high';
        return 'critical';
    }

    private detectModel(code: string, trustScore: TrustScore): string {
        // Check for model-specific patterns
        const patterns: [RegExp, string][] = [
            [/# Copilot|GitHub Copilot/i, 'GitHub Copilot'],
            [/```python|Here's (a|an|the)/i, 'ChatGPT'],
            [/I'll (create|implement)/i, 'Claude'],
            [/Generated by|Created with/i, 'AI Assistant'],
        ];

        for (const [pattern, model] of patterns) {
            if (pattern.test(code)) {
                return model;
            }
        }

        return 'unknown';
    }

    private generateRecommendations(trustScore: TrustScore, findings: Finding[]): string[] {
        const recommendations: string[] = [];

        if (trustScore.ai_probability && trustScore.ai_probability > 70) {
            recommendations.push('Manual review recommended for AI-generated code');
        }

        if (findings.some(f => f.severity === 'critical')) {
            recommendations.push('Critical issues detected - fix before committing');
        }

        if (findings.some(f => f.category === 'security')) {
            recommendations.push('Security review required');
        }

        if ((trustScore.score || 0) < 60) {
            recommendations.push('Consider running formal verification');
        }

        if (recommendations.length === 0) {
            recommendations.push('Code looks good, but manual review is still recommended');
        }

        return recommendations;
    }

    private getDefaultTrustScore(): TrustScore {
        return {
            score: 50,
            ai_probability: 0,
            risk_level: 'medium',
            complexity_score: 0,
            pattern_score: 0,
            quality_score: 0,
            verification_score: 0,
            factors: {},
        };
    }

    private getCommentPrefix(languageId: string): string {
        const prefixes: Record<string, string> = {
            'python': '#',
            'typescript': '//',
            'javascript': '//',
            'go': '//',
            'java': '//',
            'rust': '//',
            'c': '//',
            'cpp': '//',
        };
        return prefixes[languageId] || '//';
    }

    private escapeHtml(text: string): string {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * Get statistics
     */
    getStatistics(): InterceptionStatistics {
        return { ...this.statistics };
    }

    /**
     * Dispose resources
     */
    dispose(): void {
        this.disposables.forEach(d => d.dispose());
        this.statusBarItem.dispose();
        this._onPasteIntercepted.dispose();
        this._onPasteDecision.dispose();
        if (this.currentOverlay) {
            this.currentOverlay.dispose();
        }
    }
}
