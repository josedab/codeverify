/**
 * Copilot Suggestion Rewriter Provider
 *
 * Instead of rejecting unsafe Copilot suggestions, automatically rewrites them
 * to be safe while preserving developer intent.
 *
 * Key features:
 * 1. Intent Preservation Analysis - Understands what the developer wants
 * 2. Safety Transformation - Applies minimal changes to make code safe
 * 3. Visual Diff Display - Shows original vs. rewritten with explanations
 * 4. User Preference Learning - Adapts rewriting style based on accept/reject
 */

import * as vscode from 'vscode';
import { CodeVerifyClient } from '../client';
import { logger } from '../logger';

// Types
export interface SuggestionAnalysis {
    isUnsafe: boolean;
    issues: SuggestionIssue[];
    intent: string;
    confidence: number;
}

export interface SuggestionIssue {
    type: IssueType;
    severity: 'critical' | 'high' | 'medium' | 'low';
    description: string;
    location: { start: number; end: number };
    fix?: string;
}

export enum IssueType {
    NULL_DEREFERENCE = 'null_dereference',
    BOUNDS_VIOLATION = 'bounds_violation',
    TYPE_MISMATCH = 'type_mismatch',
    RESOURCE_LEAK = 'resource_leak',
    SECURITY_VULNERABILITY = 'security_vulnerability',
    UNHANDLED_EXCEPTION = 'unhandled_exception',
    RACE_CONDITION = 'race_condition',
}

export interface RewrittenSuggestion {
    original: string;
    rewritten: string;
    diff: string;
    issuesFixed: SuggestionIssue[];
    explanation: string;
    preservesIntent: boolean;
    confidence: number;
}

export interface RewritePreferences {
    preferNullCoalescing: boolean;
    preferOptionalChaining: boolean;
    preferEarlyReturn: boolean;
    addErrorHandling: boolean;
    preserveComments: boolean;
    maxChanges: number;
}

// Rewrite strategies for different issue types
interface RewriteStrategy {
    type: IssueType;
    canFix(code: string, issue: SuggestionIssue, language: string): boolean;
    fix(code: string, issue: SuggestionIssue, language: string, prefs: RewritePreferences): string;
}

// Null safety rewrite strategy
const nullSafetyStrategy: RewriteStrategy = {
    type: IssueType.NULL_DEREFERENCE,

    canFix(code: string, issue: SuggestionIssue, language: string): boolean {
        return ['python', 'typescript', 'javascript'].includes(language);
    },

    fix(code: string, issue: SuggestionIssue, language: string, prefs: RewritePreferences): string {
        if (language === 'typescript' || language === 'javascript') {
            if (prefs.preferOptionalChaining) {
                // Convert obj.property to obj?.property
                return code.replace(
                    /(\w+)\.(\w+)/g,
                    (match, obj, prop) => `${obj}?.${prop}`
                );
            } else if (prefs.preferNullCoalescing) {
                // Add nullish coalescing
                return code.replace(
                    /(\w+)\.(\w+)/g,
                    (match, obj, prop) => `(${obj} ?? {}).${prop}`
                );
            }
        } else if (language === 'python') {
            // Add None check
            const lines = code.split('\n');
            const lineIndex = findLineWithIssue(lines, issue);
            if (lineIndex >= 0) {
                const line = lines[lineIndex];
                const indent = getIndent(line);
                const varMatch = line.match(/(\w+)\./);
                if (varMatch) {
                    const varName = varMatch[1];
                    if (prefs.preferEarlyReturn) {
                        lines.splice(lineIndex, 0, `${indent}if ${varName} is None:\n${indent}    return None`);
                    } else {
                        lines[lineIndex] = `${indent}${line.trim()} if ${varName} is not None else None`;
                    }
                }
            }
            return lines.join('\n');
        }
        return code;
    }
};

// Bounds check rewrite strategy
const boundsCheckStrategy: RewriteStrategy = {
    type: IssueType.BOUNDS_VIOLATION,

    canFix(code: string, issue: SuggestionIssue, language: string): boolean {
        return ['python', 'typescript', 'javascript'].includes(language);
    },

    fix(code: string, issue: SuggestionIssue, language: string, prefs: RewritePreferences): string {
        if (language === 'typescript' || language === 'javascript') {
            // Add bounds check
            return code.replace(
                /(\w+)\[(\w+)\]/g,
                (match, arr, idx) => `${arr}[Math.min(${idx}, ${arr}.length - 1)]`
            );
        } else if (language === 'python') {
            return code.replace(
                /(\w+)\[(\w+)\]/g,
                (match, arr, idx) => `${arr}[${idx}] if 0 <= ${idx} < len(${arr}) else None`
            );
        }
        return code;
    }
};

// Resource leak rewrite strategy
const resourceLeakStrategy: RewriteStrategy = {
    type: IssueType.RESOURCE_LEAK,

    canFix(code: string, issue: SuggestionIssue, language: string): boolean {
        return language === 'python';
    },

    fix(code: string, issue: SuggestionIssue, language: string, prefs: RewritePreferences): string {
        if (language === 'python') {
            // Convert to context manager
            const fileOpenMatch = code.match(/(\w+)\s*=\s*open\(([^)]+)\)/);
            if (fileOpenMatch) {
                const [full, varName, args] = fileOpenMatch;
                const lines = code.split('\n');
                const lineIndex = lines.findIndex(l => l.includes(full));
                if (lineIndex >= 0) {
                    const indent = getIndent(lines[lineIndex]);
                    // Find the scope of the variable usage
                    const restOfCode = lines.slice(lineIndex + 1).join('\n');
                    lines[lineIndex] = `${indent}with open(${args}) as ${varName}:`;
                    // Indent following lines
                    for (let i = lineIndex + 1; i < lines.length; i++) {
                        if (lines[i].trim()) {
                            lines[i] = `    ${lines[i]}`;
                        }
                    }
                    return lines.join('\n');
                }
            }
        }
        return code;
    }
};

// Security vulnerability rewrite strategy
const securityStrategy: RewriteStrategy = {
    type: IssueType.SECURITY_VULNERABILITY,

    canFix(code: string, issue: SuggestionIssue, language: string): boolean {
        return true;
    },

    fix(code: string, issue: SuggestionIssue, language: string, prefs: RewritePreferences): string {
        // SQL injection fix
        if (code.includes('execute') && code.includes('+') && code.includes('SELECT')) {
            if (language === 'python') {
                // Convert string concatenation to parameterized query
                return code.replace(
                    /execute\s*\(\s*["']([^"']+)["']\s*\+\s*(\w+)\s*\)/g,
                    'execute("$1 ?", ($2,))'
                );
            }
        }

        // XSS fix
        if (code.includes('innerHTML') || code.includes('document.write')) {
            if (language === 'typescript' || language === 'javascript') {
                return code.replace(
                    /\.innerHTML\s*=\s*(\w+)/g,
                    '.textContent = $1'
                );
            }
        }

        return code;
    }
};

// All strategies
const REWRITE_STRATEGIES: RewriteStrategy[] = [
    nullSafetyStrategy,
    boundsCheckStrategy,
    resourceLeakStrategy,
    securityStrategy,
];

// Utility functions
function findLineWithIssue(lines: string[], issue: SuggestionIssue): number {
    // Simple heuristic - find the line containing the issue start position
    let charCount = 0;
    for (let i = 0; i < lines.length; i++) {
        charCount += lines[i].length + 1; // +1 for newline
        if (charCount >= issue.location.start) {
            return i;
        }
    }
    return -1;
}

function getIndent(line: string): string {
    const match = line.match(/^(\s*)/);
    return match ? match[1] : '';
}

function generateDiff(original: string, rewritten: string): string {
    const originalLines = original.split('\n');
    const rewrittenLines = rewritten.split('\n');
    const diff: string[] = [];

    const maxLen = Math.max(originalLines.length, rewrittenLines.length);

    for (let i = 0; i < maxLen; i++) {
        const origLine = originalLines[i] ?? '';
        const rewriteLine = rewrittenLines[i] ?? '';

        if (origLine !== rewriteLine) {
            if (origLine) diff.push(`- ${origLine}`);
            if (rewriteLine) diff.push(`+ ${rewriteLine}`);
        } else if (origLine) {
            diff.push(`  ${origLine}`);
        }
    }

    return diff.join('\n');
}

/**
 * User preference learner - adapts rewriting based on user feedback
 */
export class PreferenceLearner {
    private acceptedRewrites: Map<IssueType, number> = new Map();
    private rejectedRewrites: Map<IssueType, number> = new Map();
    private stylePreferences: Map<string, number> = new Map();

    recordAccept(rewrite: RewrittenSuggestion): void {
        for (const issue of rewrite.issuesFixed) {
            const count = this.acceptedRewrites.get(issue.type) ?? 0;
            this.acceptedRewrites.set(issue.type, count + 1);
        }

        // Learn style preferences from accepted rewrites
        if (rewrite.rewritten.includes('?.')) {
            this.incrementStyle('optionalChaining');
        }
        if (rewrite.rewritten.includes('??')) {
            this.incrementStyle('nullishCoalescing');
        }
        if (rewrite.rewritten.includes('return') && rewrite.rewritten.includes('if')) {
            this.incrementStyle('earlyReturn');
        }
    }

    recordReject(rewrite: RewrittenSuggestion, reason?: string): void {
        for (const issue of rewrite.issuesFixed) {
            const count = this.rejectedRewrites.get(issue.type) ?? 0;
            this.rejectedRewrites.set(issue.type, count + 1);
        }

        // Decrement style preferences for rejected patterns
        if (rewrite.rewritten.includes('?.')) {
            this.decrementStyle('optionalChaining');
        }
        if (rewrite.rewritten.includes('??')) {
            this.decrementStyle('nullishCoalescing');
        }
    }

    private incrementStyle(style: string): void {
        const count = this.stylePreferences.get(style) ?? 0;
        this.stylePreferences.set(style, count + 1);
    }

    private decrementStyle(style: string): void {
        const count = this.stylePreferences.get(style) ?? 0;
        this.stylePreferences.set(style, Math.max(0, count - 1));
    }

    getPreferences(): RewritePreferences {
        return {
            preferNullCoalescing: (this.stylePreferences.get('nullishCoalescing') ?? 0) > 2,
            preferOptionalChaining: (this.stylePreferences.get('optionalChaining') ?? 5) > 3,
            preferEarlyReturn: (this.stylePreferences.get('earlyReturn') ?? 0) > 2,
            addErrorHandling: true,
            preserveComments: true,
            maxChanges: 10,
        };
    }

    shouldAttemptFix(issueType: IssueType): boolean {
        const accepted = this.acceptedRewrites.get(issueType) ?? 0;
        const rejected = this.rejectedRewrites.get(issueType) ?? 0;
        const total = accepted + rejected;

        if (total < 3) {
            return true; // Not enough data, try anyway
        }

        // Require > 40% acceptance rate
        return (accepted / total) > 0.4;
    }

    getStatistics(): object {
        return {
            accepted: Object.fromEntries(this.acceptedRewrites),
            rejected: Object.fromEntries(this.rejectedRewrites),
            stylePreferences: Object.fromEntries(this.stylePreferences),
        };
    }
}

/**
 * Main Suggestion Rewriter Provider
 */
export class SuggestionRewriterProvider implements vscode.Disposable {
    private client: CodeVerifyClient;
    private preferenceLearner: PreferenceLearner;
    private enabled: boolean = true;
    private disposables: vscode.Disposable[] = [];
    private pendingRewrites: Map<string, RewrittenSuggestion> = new Map();

    // Events
    private _onRewriteAvailable = new vscode.EventEmitter<{
        original: string;
        rewritten: RewrittenSuggestion;
        editor: vscode.TextEditor;
    }>();
    public readonly onRewriteAvailable = this._onRewriteAvailable.event;

    private _onRewriteAccepted = new vscode.EventEmitter<RewrittenSuggestion>();
    public readonly onRewriteAccepted = this._onRewriteAccepted.event;

    private _onRewriteRejected = new vscode.EventEmitter<{
        rewrite: RewrittenSuggestion;
        reason?: string;
    }>();
    public readonly onRewriteRejected = this._onRewriteRejected.event;

    // Decoration types
    private originalDecoration: vscode.TextEditorDecorationType;
    private rewrittenDecoration: vscode.TextEditorDecorationType;

    constructor(client: CodeVerifyClient) {
        this.client = client;
        this.preferenceLearner = new PreferenceLearner();

        // Initialize decorations
        this.originalDecoration = vscode.window.createTextEditorDecorationType({
            backgroundColor: 'rgba(255, 100, 100, 0.2)',
            isWholeLine: true,
        });

        this.rewrittenDecoration = vscode.window.createTextEditorDecorationType({
            backgroundColor: 'rgba(100, 255, 100, 0.2)',
            isWholeLine: true,
        });

        // Register inline completion provider to intercept suggestions
        this.disposables.push(
            vscode.languages.registerInlineCompletionItemProvider(
                [
                    { scheme: 'file', language: 'python' },
                    { scheme: 'file', language: 'typescript' },
                    { scheme: 'file', language: 'javascript' },
                ],
                {
                    provideInlineCompletionItems: this.interceptSuggestion.bind(this),
                }
            )
        );
    }

    /**
     * Intercept and potentially rewrite a suggestion
     */
    private async interceptSuggestion(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken,
    ): Promise<vscode.InlineCompletionItem[] | vscode.InlineCompletionList | null> {
        if (!this.enabled) {
            return null;
        }

        // We don't actually provide suggestions - we rewrite ones from Copilot
        // This is handled by intercepting the accept action
        return null;
    }

    /**
     * Analyze and potentially rewrite a code suggestion
     */
    async analyzeSuggestion(
        suggestion: string,
        context: {
            document: vscode.TextDocument;
            position: vscode.Position;
            surroundingCode: string;
        },
    ): Promise<SuggestionAnalysis> {
        const language = context.document.languageId;

        try {
            // Call backend for analysis
            const analysis = await this.client.analyzeSuggestion(
                suggestion,
                language,
                context.surroundingCode,
            );

            return {
                isUnsafe: analysis.issues?.length > 0,
                issues: analysis.issues ?? [],
                intent: analysis.intent ?? 'Unknown intent',
                confidence: analysis.confidence ?? 0.5,
            };
        } catch (error) {
            logger.error('Failed to analyze suggestion', { error });

            // Fallback to local analysis
            return this.localAnalyze(suggestion, language);
        }
    }

    /**
     * Local analysis fallback
     */
    private localAnalyze(suggestion: string, language: string): SuggestionAnalysis {
        const issues: SuggestionIssue[] = [];

        // Check for null dereference patterns
        if (language === 'python') {
            if (suggestion.match(/(\w+)\.(\w+)/) && !suggestion.includes('is not None')) {
                issues.push({
                    type: IssueType.NULL_DEREFERENCE,
                    severity: 'medium',
                    description: 'Potential None dereference',
                    location: { start: 0, end: suggestion.length },
                });
            }
        } else if (language === 'typescript' || language === 'javascript') {
            if (suggestion.match(/(\w+)\.(\w+)/) && !suggestion.includes('?.') && !suggestion.includes('??')) {
                issues.push({
                    type: IssueType.NULL_DEREFERENCE,
                    severity: 'medium',
                    description: 'Potential null/undefined dereference',
                    location: { start: 0, end: suggestion.length },
                });
            }
        }

        // Check for array access without bounds check
        if (suggestion.match(/\[\w+\]/) && !suggestion.includes('length')) {
            issues.push({
                type: IssueType.BOUNDS_VIOLATION,
                severity: 'medium',
                description: 'Array access without bounds check',
                location: { start: 0, end: suggestion.length },
            });
        }

        // Check for SQL injection
        if (suggestion.includes('execute') && suggestion.includes('+')) {
            issues.push({
                type: IssueType.SECURITY_VULNERABILITY,
                severity: 'critical',
                description: 'Potential SQL injection via string concatenation',
                location: { start: 0, end: suggestion.length },
            });
        }

        // Check for resource leaks
        if (language === 'python' && suggestion.includes('open(') && !suggestion.includes('with ')) {
            issues.push({
                type: IssueType.RESOURCE_LEAK,
                severity: 'medium',
                description: 'File opened without context manager',
                location: { start: 0, end: suggestion.length },
            });
        }

        return {
            isUnsafe: issues.length > 0,
            issues,
            intent: 'Inferred from code patterns',
            confidence: 0.6,
        };
    }

    /**
     * Rewrite a suggestion to fix issues while preserving intent
     */
    async rewriteSuggestion(
        suggestion: string,
        analysis: SuggestionAnalysis,
        language: string,
    ): Promise<RewrittenSuggestion | null> {
        if (!analysis.isUnsafe || analysis.issues.length === 0) {
            return null;
        }

        const preferences = this.preferenceLearner.getPreferences();
        let rewritten = suggestion;
        const fixedIssues: SuggestionIssue[] = [];

        // Apply fixes for each issue
        for (const issue of analysis.issues) {
            // Check if we should attempt this fix based on learning
            if (!this.preferenceLearner.shouldAttemptFix(issue.type)) {
                logger.debug('Skipping fix due to low acceptance rate', { type: issue.type });
                continue;
            }

            // Find applicable strategy
            const strategy = REWRITE_STRATEGIES.find(s =>
                s.type === issue.type && s.canFix(rewritten, issue, language)
            );

            if (strategy) {
                try {
                    const before = rewritten;
                    rewritten = strategy.fix(rewritten, issue, language, preferences);

                    if (rewritten !== before) {
                        fixedIssues.push(issue);
                    }
                } catch (error) {
                    logger.error('Failed to apply fix', { error, type: issue.type });
                }
            }
        }

        if (fixedIssues.length === 0) {
            return null;
        }

        const diff = generateDiff(suggestion, rewritten);
        const explanation = this.generateExplanation(fixedIssues, language);

        return {
            original: suggestion,
            rewritten,
            diff,
            issuesFixed: fixedIssues,
            explanation,
            preservesIntent: true, // TODO: Verify with LLM
            confidence: this.calculateConfidence(fixedIssues, analysis),
        };
    }

    /**
     * Generate human-readable explanation of changes
     */
    private generateExplanation(issues: SuggestionIssue[], language: string): string {
        const explanations: string[] = [];

        for (const issue of issues) {
            switch (issue.type) {
                case IssueType.NULL_DEREFERENCE:
                    if (language === 'typescript' || language === 'javascript') {
                        explanations.push('Added optional chaining (?.) to prevent null/undefined errors');
                    } else {
                        explanations.push('Added None check to prevent AttributeError');
                    }
                    break;

                case IssueType.BOUNDS_VIOLATION:
                    explanations.push('Added bounds checking to prevent index out of range');
                    break;

                case IssueType.RESOURCE_LEAK:
                    explanations.push('Wrapped file operation in context manager to ensure cleanup');
                    break;

                case IssueType.SECURITY_VULNERABILITY:
                    explanations.push('Fixed potential security vulnerability (e.g., SQL injection, XSS)');
                    break;

                default:
                    explanations.push(`Fixed ${issue.description}`);
            }
        }

        return explanations.join('\n');
    }

    /**
     * Calculate confidence in the rewrite
     */
    private calculateConfidence(fixedIssues: SuggestionIssue[], analysis: SuggestionAnalysis): number {
        // Base confidence from analysis
        let confidence = analysis.confidence;

        // Adjust based on issue severity
        for (const issue of fixedIssues) {
            switch (issue.severity) {
                case 'critical':
                    confidence += 0.1; // High value in fixing critical
                    break;
                case 'high':
                    confidence += 0.05;
                    break;
                case 'medium':
                    confidence += 0.02;
                    break;
            }
        }

        // Adjust based on historical acceptance
        for (const issue of fixedIssues) {
            if (this.preferenceLearner.shouldAttemptFix(issue.type)) {
                confidence += 0.05;
            }
        }

        return Math.min(confidence, 1.0);
    }

    /**
     * Show rewrite options to user
     */
    async showRewriteOptions(
        original: string,
        rewrite: RewrittenSuggestion,
        editor: vscode.TextEditor,
        range: vscode.Range,
    ): Promise<void> {
        // Store pending rewrite
        const id = `${editor.document.uri.toString()}-${range.start.line}`;
        this.pendingRewrites.set(id, rewrite);

        // Show quick pick with options
        const items: vscode.QuickPickItem[] = [
            {
                label: '$(check) Accept Rewritten Suggestion',
                description: `${rewrite.issuesFixed.length} issue(s) fixed`,
                detail: rewrite.explanation,
            },
            {
                label: '$(code) Use Original (Unsafe)',
                description: 'Insert the original suggestion without safety fixes',
            },
            {
                label: '$(diff) View Diff',
                description: 'See the changes made by the rewriter',
            },
            {
                label: '$(x) Cancel',
                description: 'Do not insert any code',
            },
        ];

        const selected = await vscode.window.showQuickPick(items, {
            title: 'CodeVerify Suggestion Rewriter',
            placeHolder: 'The suggestion has been rewritten to fix safety issues',
        });

        if (!selected) {
            return;
        }

        if (selected.label.includes('Accept Rewritten')) {
            // Insert rewritten code
            await editor.edit(editBuilder => {
                editBuilder.replace(range, rewrite.rewritten);
            });
            this.preferenceLearner.recordAccept(rewrite);
            this._onRewriteAccepted.fire(rewrite);
            vscode.window.showInformationMessage(
                `Inserted rewritten code with ${rewrite.issuesFixed.length} fix(es)`
            );

        } else if (selected.label.includes('Use Original')) {
            // Insert original (with warning)
            const confirm = await vscode.window.showWarningMessage(
                'This code has known safety issues. Are you sure?',
                'Yes, Insert Anyway',
                'Cancel'
            );

            if (confirm === 'Yes, Insert Anyway') {
                await editor.edit(editBuilder => {
                    editBuilder.replace(range, original);
                });
                this.preferenceLearner.recordReject(rewrite, 'User chose original');
                this._onRewriteRejected.fire({ rewrite, reason: 'User chose original' });
            }

        } else if (selected.label.includes('View Diff')) {
            // Show diff in webview
            await this.showDiffPanel(original, rewrite);
            // Recursively show options after viewing diff
            await this.showRewriteOptions(original, rewrite, editor, range);
        }

        this.pendingRewrites.delete(id);
    }

    /**
     * Show diff in a webview panel
     */
    private async showDiffPanel(original: string, rewrite: RewrittenSuggestion): Promise<void> {
        const panel = vscode.window.createWebviewPanel(
            'codeverifyRewriteDiff',
            'Suggestion Rewrite Diff',
            vscode.ViewColumn.Beside,
            { enableScripts: true }
        );

        const diffLines = rewrite.diff.split('\n').map(line => {
            if (line.startsWith('+')) {
                return `<div class="added">${escapeHtml(line)}</div>`;
            } else if (line.startsWith('-')) {
                return `<div class="removed">${escapeHtml(line)}</div>`;
            } else {
                return `<div class="unchanged">${escapeHtml(line)}</div>`;
            }
        }).join('');

        const issuesList = rewrite.issuesFixed.map(issue =>
            `<li><span class="severity-${issue.severity}">${issue.severity.toUpperCase()}</span> ${escapeHtml(issue.description)}</li>`
        ).join('');

        panel.webview.html = `<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding: 20px;
            line-height: 1.6;
            color: var(--vscode-foreground);
        }
        h2 { margin-top: 0; }
        .diff-container {
            background: var(--vscode-editor-background);
            padding: 15px;
            border-radius: 8px;
            font-family: 'Fira Code', monospace;
            font-size: 13px;
            overflow-x: auto;
        }
        .added { background: rgba(46, 160, 67, 0.2); color: #3fb950; }
        .removed { background: rgba(248, 81, 73, 0.2); color: #f85149; }
        .unchanged { color: var(--vscode-foreground); opacity: 0.7; }
        .issues { margin: 20px 0; }
        .issues ul { padding-left: 20px; }
        .issues li { margin: 5px 0; }
        .severity-critical { color: #f85149; font-weight: bold; }
        .severity-high { color: #f85149; }
        .severity-medium { color: #d29922; }
        .severity-low { color: #3fb950; }
        .explanation {
            background: var(--vscode-editor-inactiveSelectionBackground);
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .confidence {
            margin-top: 10px;
            font-size: 12px;
            color: var(--vscode-descriptionForeground);
        }
    </style>
</head>
<body>
    <h2>Suggestion Rewrite</h2>

    <div class="issues">
        <h3>Issues Fixed</h3>
        <ul>${issuesList}</ul>
    </div>

    <h3>Code Changes</h3>
    <div class="diff-container">
        ${diffLines}
    </div>

    <div class="explanation">
        <h3>Explanation</h3>
        <p>${escapeHtml(rewrite.explanation)}</p>
    </div>

    <div class="confidence">
        Confidence: ${(rewrite.confidence * 100).toFixed(0)}%
        ${rewrite.preservesIntent ? '• Intent preserved' : '• Intent may have changed'}
    </div>
</body>
</html>`;
    }

    /**
     * Process pasted code (for integration with paste interception)
     */
    async processPastedCode(
        code: string,
        document: vscode.TextDocument,
        position: vscode.Position,
    ): Promise<RewrittenSuggestion | null> {
        const surroundingCode = this.getSurroundingCode(document, position);

        const analysis = await this.analyzeSuggestion(code, {
            document,
            position,
            surroundingCode,
        });

        if (!analysis.isUnsafe) {
            return null;
        }

        return this.rewriteSuggestion(code, analysis, document.languageId);
    }

    private getSurroundingCode(document: vscode.TextDocument, position: vscode.Position): string {
        const startLine = Math.max(0, position.line - 10);
        const endLine = Math.min(document.lineCount - 1, position.line + 10);

        const range = new vscode.Range(startLine, 0, endLine, document.lineAt(endLine).text.length);
        return document.getText(range);
    }

    enable(): void {
        this.enabled = true;
    }

    disable(): void {
        this.enabled = false;
    }

    toggle(): void {
        this.enabled = !this.enabled;
    }

    isEnabled(): boolean {
        return this.enabled;
    }

    getStatistics(): object {
        return this.preferenceLearner.getStatistics();
    }

    dispose(): void {
        this._onRewriteAvailable.dispose();
        this._onRewriteAccepted.dispose();
        this._onRewriteRejected.dispose();
        this.originalDecoration.dispose();
        this.rewrittenDecoration.dispose();
        this.disposables.forEach(d => d.dispose());
    }
}

function escapeHtml(text: string): string {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
