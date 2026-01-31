/**
 * AI Pair Reviewer Provider
 * 
 * Real-time code review as developers type with:
 * - Sub-function granularity analysis
 * - Inline diagnostic provider with severity levels
 * - CodeLens for verification status per function
 * - Quick-fix actions for suggested remediations
 * - Smart throttling (verify on pause, not keystroke)
 * - Learning from user corrections
 */

import * as vscode from 'vscode';
import { CodeVerifyClient, Finding } from '../client';

// Review priority levels
export enum ReviewPriority {
    CRITICAL = 'critical',
    HIGH = 'high',
    MEDIUM = 'medium',
    LOW = 'low',
    SUGGESTION = 'suggestion',
}

// Review finding categories
export enum ReviewCategory {
    BUG = 'bug',
    SECURITY = 'security',
    PERFORMANCE = 'performance',
    LOGIC_ERROR = 'logic_error',
    TYPE_ERROR = 'type_error',
    NULL_SAFETY = 'null_safety',
    RESOURCE_LEAK = 'resource_leak',
    STYLE = 'style',
    BEST_PRACTICE = 'best_practice',
}

// Inline finding from pair reviewer
interface InlineFinding {
    id: string;
    category: ReviewCategory;
    priority: ReviewPriority;
    message: string;
    lineStart: number;
    lineEnd: number;
    colStart?: number;
    colEnd?: number;
    suggestion?: string;
    fixCode?: string;
    confidence: number;
    explanation?: string;
}

// Verification status for code regions
interface VerificationUnit {
    id: string;
    name: string;
    type: 'function' | 'class' | 'method' | 'block';
    range: vscode.Range;
    status: 'pending' | 'verifying' | 'verified' | 'error';
    findings: InlineFinding[];
    lastVerified?: Date;
    contentHash: string;
}

// User feedback on findings
interface ReviewFeedback {
    findingId: string;
    action: 'accepted' | 'dismissed' | 'modified';
    reason?: string;
    timestamp: Date;
}

/**
 * Smart throttler for real-time verification
 */
class SmartThrottler {
    private timers = new Map<string, NodeJS.Timeout>();
    private lastTrigger = new Map<string, number>();
    private typingVelocity = new Map<string, number[]>();
    
    constructor(
        private baseDelayMs: number = 300,
        private maxDelayMs: number = 2000
    ) {}
    
    throttle(key: string, content: string, callback: () => void): void {
        // Record keystroke timing
        const now = Date.now();
        const last = this.lastTrigger.get(key) || now;
        this.lastTrigger.set(key, now);
        
        // Track velocity
        const velocity = 1000 / Math.max(1, now - last);
        const velocities = this.typingVelocity.get(key) || [];
        velocities.push(velocity);
        if (velocities.length > 10) velocities.shift();
        this.typingVelocity.set(key, velocities);
        
        // Clear existing timer
        const existing = this.timers.get(key);
        if (existing) {
            clearTimeout(existing);
        }
        
        // Calculate adaptive delay
        const delay = this.calculateDelay(key, content);
        
        // Set new timer
        this.timers.set(key, setTimeout(() => {
            callback();
            this.timers.delete(key);
        }, delay));
    }
    
    private calculateDelay(key: string, content: string): number {
        let delay = this.baseDelayMs;
        
        // Adjust for content size
        const lines = content.split('\n').length;
        if (lines > 20) {
            delay *= Math.min(2.0, 1 + (lines - 20) / 50);
        }
        
        // Adjust for typing velocity
        const velocities = this.typingVelocity.get(key) || [];
        if (velocities.length >= 3) {
            const avgVelocity = velocities.slice(-3).reduce((a, b) => a + b, 0) / 3;
            if (avgVelocity > 5) {
                delay *= Math.min(1.5, 1 + (avgVelocity - 5) / 10);
            }
        }
        
        return Math.min(this.maxDelayMs, delay);
    }
    
    cancel(key: string): void {
        const timer = this.timers.get(key);
        if (timer) {
            clearTimeout(timer);
            this.timers.delete(key);
        }
    }
    
    cancelAll(): void {
        for (const timer of this.timers.values()) {
            clearTimeout(timer);
        }
        this.timers.clear();
    }
}

/**
 * Feedback learner for improving review quality
 */
class FeedbackLearner {
    private feedback: ReviewFeedback[] = [];
    private patternStats = new Map<string, { accepted: number; dismissed: number; modified: number }>();
    private thresholds = new Map<string, number>();
    
    recordFeedback(feedback: ReviewFeedback, finding: InlineFinding): void {
        this.feedback.push(feedback);
        
        // Update pattern statistics
        const patternKey = `${finding.category}:${finding.message.substring(0, 50)}`;
        const stats = this.patternStats.get(patternKey) || { accepted: 0, dismissed: 0, modified: 0 };
        stats[feedback.action]++;
        this.patternStats.set(patternKey, stats);
        
        this.updateThresholds();
    }
    
    private updateThresholds(): void {
        for (const [patternKey, stats] of this.patternStats.entries()) {
            const total = stats.accepted + stats.dismissed + stats.modified;
            if (total < 10) continue;
            
            const acceptanceRate = (stats.accepted + 0.5 * stats.modified) / total;
            const category = patternKey.split(':')[0];
            
            if (acceptanceRate < 0.3) {
                this.thresholds.set(category, Math.min(0.95, 0.7 + (0.3 - acceptanceRate)));
            } else if (acceptanceRate > 0.8) {
                this.thresholds.set(category, Math.max(0.5, 0.7 - (acceptanceRate - 0.8) * 0.5));
            }
        }
    }
    
    getConfidenceThreshold(category: ReviewCategory): number {
        return this.thresholds.get(category) || 0.7;
    }
    
    shouldShowFinding(finding: InlineFinding): boolean {
        const threshold = this.getConfidenceThreshold(finding.category);
        return finding.confidence >= threshold;
    }
    
    getStatistics(): object {
        return {
            totalFeedback: this.feedback.length,
            patterns: Object.fromEntries(this.patternStats),
            thresholds: Object.fromEntries(this.thresholds),
        };
    }
}

/**
 * Main Pair Reviewer Provider
 */
export class PairReviewerProvider implements vscode.Disposable {
    private client: CodeVerifyClient;
    private enabled = false;
    
    // Throttling and caching
    private throttler = new SmartThrottler();
    private feedbackLearner = new FeedbackLearner();
    private cache = new Map<string, { findings: InlineFinding[]; timestamp: Date }>();
    private cacheTTL = 300000; // 5 minutes
    
    // Document tracking
    private documentUnits = new Map<string, Map<string, VerificationUnit>>();
    private activeReviews = new Set<string>();
    
    // Diagnostics
    private diagnosticCollection: vscode.DiagnosticCollection;
    
    // Decorations
    private verifiedDecoration: vscode.TextEditorDecorationType;
    private pendingDecoration: vscode.TextEditorDecorationType;
    private reviewingDecoration: vscode.TextEditorDecorationType;
    private errorDecoration: vscode.TextEditorDecorationType;
    private suggestionDecoration: vscode.TextEditorDecorationType;
    
    // CodeLens
    private codeLensProvider: PairReviewerCodeLensProvider;
    private codeLensDisposable: vscode.Disposable | undefined;
    
    // Status bar
    private statusBarItem: vscode.StatusBarItem;
    
    // Events
    private _onReviewComplete = new vscode.EventEmitter<{
        uri: vscode.Uri;
        findings: InlineFinding[];
    }>();
    readonly onReviewComplete = this._onReviewComplete.event;
    
    private disposables: vscode.Disposable[] = [];
    
    constructor(client: CodeVerifyClient) {
        this.client = client;
        
        // Initialize diagnostics
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeverify-pair-review');
        this.disposables.push(this.diagnosticCollection);
        
        // Initialize decorations
        this.verifiedDecoration = this.createDecoration('#2ecc71', '✓', 'Verified');
        this.pendingDecoration = this.createDecoration('#95a5a6', '○', 'Pending verification');
        this.reviewingDecoration = this.createDecoration('#3498db', '⟳', 'Reviewing...');
        this.errorDecoration = this.createDecoration('#e74c3c', '✗', 'Issues found');
        this.suggestionDecoration = this.createDecoration('#f39c12', '💡', 'Suggestions available');
        
        // Initialize CodeLens provider
        this.codeLensProvider = new PairReviewerCodeLensProvider(this);
        
        // Initialize status bar
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 95);
        this.statusBarItem.command = 'codeverify.togglePairReviewer';
        this.updateStatusBar();
        this.statusBarItem.show();
        this.disposables.push(this.statusBarItem);
    }
    
    /**
     * Enable pair reviewer
     */
    enable(): void {
        this.enabled = true;
        
        // Register CodeLens provider
        this.codeLensDisposable = vscode.languages.registerCodeLensProvider(
            [
                { scheme: 'file', language: 'python' },
                { scheme: 'file', language: 'typescript' },
                { scheme: 'file', language: 'javascript' },
            ],
            this.codeLensProvider
        );
        this.disposables.push(this.codeLensDisposable);
        
        // Initialize active editor
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            this.initializeDocument(editor.document);
        }
        
        this.updateStatusBar();
    }
    
    /**
     * Disable pair reviewer
     */
    disable(): void {
        this.enabled = false;
        this.throttler.cancelAll();
        
        // Dispose CodeLens provider
        if (this.codeLensDisposable) {
            const idx = this.disposables.indexOf(this.codeLensDisposable);
            if (idx >= 0) this.disposables.splice(idx, 1);
            this.codeLensDisposable.dispose();
            this.codeLensDisposable = undefined;
        }
        
        // Clear diagnostics
        this.diagnosticCollection.clear();
        
        this.updateStatusBar();
    }
    
    /**
     * Toggle pair reviewer
     */
    toggle(): void {
        if (this.enabled) {
            this.disable();
        } else {
            this.enable();
        }
    }
    
    /**
     * Initialize document tracking
     */
    async initializeDocument(document: vscode.TextDocument): Promise<void> {
        if (!this.enabled || !this.isSupportedLanguage(document.languageId)) {
            return;
        }
        
        const uri = document.uri.toString();
        const units = await this.parseDocumentUnits(document);
        this.documentUnits.set(uri, units);
        
        // Queue initial review for all units
        for (const unit of units.values()) {
            this.queueReview(document, unit);
        }
    }
    
    /**
     * Handle document change
     */
    async handleChange(event: vscode.TextDocumentChangeEvent): Promise<void> {
        if (!this.enabled || !this.isSupportedLanguage(event.document.languageId)) {
            return;
        }
        
        const uri = event.document.uri.toString();
        let units = this.documentUnits.get(uri);
        
        if (!units) {
            await this.initializeDocument(event.document);
            return;
        }
        
        // Find affected units
        const affectedUnits = this.findAffectedUnits(event, units);
        
        // Mark as pending
        for (const unitId of affectedUnits) {
            const unit = units.get(unitId);
            if (unit) {
                unit.status = 'pending';
            }
        }
        
        // Re-parse document
        units = await this.parseDocumentUnits(event.document);
        this.documentUnits.set(uri, units);
        
        // Queue reviews for affected units
        for (const unitId of affectedUnits) {
            const unit = units.get(unitId);
            if (unit) {
                this.queueReview(event.document, unit);
            }
        }
        
        this.updateDecorations(event.document);
    }
    
    /**
     * Queue a review with smart throttling
     */
    private queueReview(document: vscode.TextDocument, unit: VerificationUnit): void {
        const key = `${document.uri.toString()}:${unit.id}`;
        
        // Check cache
        const cached = this.cache.get(unit.contentHash);
        if (cached && Date.now() - cached.timestamp.getTime() < this.cacheTTL) {
            unit.findings = cached.findings;
            unit.status = cached.findings.length > 0 ? 'error' : 'verified';
            unit.lastVerified = cached.timestamp;
            this.updateDiagnostics(document, [unit]);
            this.updateDecorations(document);
            return;
        }
        
        // Throttle the review
        const code = document.getText(unit.range);
        this.throttler.throttle(key, code, async () => {
            await this.performReview(document, unit);
        });
    }
    
    /**
     * Perform the actual review
     */
    private async performReview(document: vscode.TextDocument, unit: VerificationUnit): Promise<void> {
        const key = `${document.uri.toString()}:${unit.id}`;
        
        // Mark as reviewing
        unit.status = 'verifying';
        this.activeReviews.add(key);
        this.updateDecorations(document);
        this.updateStatusBar();
        
        try {
            const code = document.getText(unit.range);
            
            // Get surrounding context
            const contextStart = Math.max(0, unit.range.start.line - 10);
            const contextEnd = Math.min(document.lineCount - 1, unit.range.end.line + 5);
            const surroundingContext = document.getText(
                new vscode.Range(contextStart, 0, contextEnd, document.lineAt(contextEnd).text.length)
            );
            
            // Call the API for review
            const rawFindings = await this.client.pairReview(code, {
                language: document.languageId,
                filePath: document.uri.fsPath,
                lineStart: unit.range.start.line + 1,
                surroundingContext,
                unitType: unit.type,
                unitName: unit.name,
            });
            
            // Convert to InlineFinding objects
            const findings: InlineFinding[] = rawFindings.map((f: any, i: number) => ({
                id: `${unit.contentHash}:${i}`,
                category: f.category || ReviewCategory.BUG,
                priority: f.priority || ReviewPriority.MEDIUM,
                message: f.message || 'Unknown issue',
                lineStart: unit.range.start.line + (f.line_start || 1) - 1,
                lineEnd: unit.range.start.line + (f.line_end || f.line_start || 1) - 1,
                colStart: f.col_start,
                colEnd: f.col_end,
                suggestion: f.suggestion,
                fixCode: f.fix_code,
                confidence: f.confidence || 0.8,
                explanation: f.explanation,
            }));
            
            // Filter based on learned patterns
            const filteredFindings = findings.filter(f => this.feedbackLearner.shouldShowFinding(f));
            
            // Update unit
            unit.findings = filteredFindings;
            unit.status = filteredFindings.length > 0 ? 'error' : 'verified';
            unit.lastVerified = new Date();
            
            // Cache results
            this.cache.set(unit.contentHash, {
                findings: filteredFindings,
                timestamp: new Date(),
            });
            
            // Enforce cache size
            if (this.cache.size > 1000) {
                const firstKey = this.cache.keys().next().value;
                if (firstKey) this.cache.delete(firstKey);
            }
            
            // Update diagnostics
            this.updateDiagnostics(document, [unit]);
            
            // Fire completion event
            this._onReviewComplete.fire({
                uri: document.uri,
                findings: filteredFindings,
            });
            
        } catch (error) {
            console.error(`Pair review failed for ${unit.id}:`, error);
            unit.status = 'error';
        } finally {
            this.activeReviews.delete(key);
            this.updateDecorations(document);
            this.updateStatusBar();
        }
    }
    
    /**
     * Record user feedback on a finding
     */
    recordFeedback(finding: InlineFinding, action: 'accepted' | 'dismissed' | 'modified', reason?: string): void {
        this.feedbackLearner.recordFeedback({
            findingId: finding.id,
            action,
            reason,
            timestamp: new Date(),
        }, finding);
    }
    
    /**
     * Get units for a document
     */
    getUnits(uri: vscode.Uri): Map<string, VerificationUnit> | undefined {
        return this.documentUnits.get(uri.toString());
    }
    
    /**
     * Parse document into verification units
     */
    private async parseDocumentUnits(document: vscode.TextDocument): Promise<Map<string, VerificationUnit>> {
        const units = new Map<string, VerificationUnit>();
        const text = document.getText();
        const languageId = document.languageId;
        
        if (languageId === 'python') {
            // Parse Python functions and classes
            const functionRegex = /^(\s*)(async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->[^:]+)?:/gm;
            const classRegex = /^(\s*)class\s+(\w+)(?:\([^)]*\))?\s*:/gm;
            
            let match;
            while ((match = functionRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findPythonBlockEnd(document, startPos);
                const code = document.getText(new vscode.Range(startPos, endPos));
                const id = `func:${match[3]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    name: match[3],
                    type: 'function',
                    range: new vscode.Range(startPos, endPos),
                    status: 'pending',
                    findings: [],
                    contentHash: this.hashCode(code),
                });
            }
            
            while ((match = classRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findPythonBlockEnd(document, startPos);
                const code = document.getText(new vscode.Range(startPos, endPos));
                const id = `class:${match[2]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    name: match[2],
                    type: 'class',
                    range: new vscode.Range(startPos, endPos),
                    status: 'pending',
                    findings: [],
                    contentHash: this.hashCode(code),
                });
            }
        } else if (['typescript', 'javascript'].includes(languageId)) {
            // Parse TypeScript/JavaScript
            const functionRegex = /(?:export\s+)?(?:async\s+)?function\s+(\w+)/gm;
            const classRegex = /(?:export\s+)?(?:abstract\s+)?class\s+(\w+)/gm;
            const arrowRegex = /(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>/gm;
            
            let match;
            while ((match = functionRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findBraceBlockEnd(document, startPos);
                const code = document.getText(new vscode.Range(startPos, endPos));
                const id = `func:${match[1]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    name: match[1],
                    type: 'function',
                    range: new vscode.Range(startPos, endPos),
                    status: 'pending',
                    findings: [],
                    contentHash: this.hashCode(code),
                });
            }
            
            while ((match = classRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findBraceBlockEnd(document, startPos);
                const code = document.getText(new vscode.Range(startPos, endPos));
                const id = `class:${match[1]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    name: match[1],
                    type: 'class',
                    range: new vscode.Range(startPos, endPos),
                    status: 'pending',
                    findings: [],
                    contentHash: this.hashCode(code),
                });
            }
            
            while ((match = arrowRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findArrowFunctionEnd(document, startPos);
                const code = document.getText(new vscode.Range(startPos, endPos));
                const id = `arrow:${match[1]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    name: match[1],
                    type: 'function',
                    range: new vscode.Range(startPos, endPos),
                    status: 'pending',
                    findings: [],
                    contentHash: this.hashCode(code),
                });
            }
        }
        
        return units;
    }
    
    private findPythonBlockEnd(document: vscode.TextDocument, start: vscode.Position): vscode.Position {
        const startIndent = this.getIndentLevel(document.lineAt(start.line).text);
        let endLine = start.line;
        
        for (let i = start.line + 1; i < document.lineCount; i++) {
            const line = document.lineAt(i);
            if (line.isEmptyOrWhitespace) continue;
            
            const indent = this.getIndentLevel(line.text);
            if (indent <= startIndent && !line.text.trimStart().startsWith('#')) {
                break;
            }
            endLine = i;
        }
        
        return new vscode.Position(endLine, document.lineAt(endLine).text.length);
    }
    
    private findBraceBlockEnd(document: vscode.TextDocument, start: vscode.Position): vscode.Position {
        const text = document.getText();
        const startOffset = document.offsetAt(start);
        
        let braceCount = 0;
        let foundFirst = false;
        
        for (let i = startOffset; i < text.length; i++) {
            if (text[i] === '{') {
                braceCount++;
                foundFirst = true;
            } else if (text[i] === '}') {
                braceCount--;
                if (foundFirst && braceCount === 0) {
                    return document.positionAt(i + 1);
                }
            }
        }
        
        return document.positionAt(text.length);
    }
    
    private findArrowFunctionEnd(document: vscode.TextDocument, start: vscode.Position): vscode.Position {
        const text = document.getText();
        const startOffset = document.offsetAt(start);
        
        const arrowIndex = text.indexOf('=>', startOffset);
        if (arrowIndex === -1) return start;
        
        let afterArrow = arrowIndex + 2;
        while (afterArrow < text.length && /\s/.test(text[afterArrow])) {
            afterArrow++;
        }
        
        if (text[afterArrow] === '{') {
            return this.findBraceBlockEnd(document, document.positionAt(afterArrow));
        }
        
        // Expression body
        let end = afterArrow;
        let parenCount = 0;
        while (end < text.length) {
            if (text[end] === '(') parenCount++;
            else if (text[end] === ')') parenCount--;
            else if ((text[end] === ';' || text[end] === '\n') && parenCount === 0) {
                break;
            }
            end++;
        }
        
        return document.positionAt(end);
    }
    
    private getIndentLevel(line: string): number {
        const match = line.match(/^(\s*)/);
        return match ? match[1].length : 0;
    }
    
    private findAffectedUnits(
        event: vscode.TextDocumentChangeEvent,
        units: Map<string, VerificationUnit>
    ): string[] {
        const affected = new Set<string>();
        
        for (const change of event.contentChanges) {
            const changeRange = 'range' in change ? change.range : null;
            if (!changeRange) continue;
            
            for (const [id, unit] of units) {
                if (this.rangesOverlap(changeRange, unit.range)) {
                    affected.add(id);
                }
            }
        }
        
        return Array.from(affected);
    }
    
    private rangesOverlap(a: vscode.Range, b: vscode.Range): boolean {
        return !(a.end.isBefore(b.start) || b.end.isBefore(a.start));
    }
    
    /**
     * Update diagnostics for units
     */
    private updateDiagnostics(document: vscode.TextDocument, units: VerificationUnit[]): void {
        const diagnostics: vscode.Diagnostic[] = [];
        
        for (const unit of units) {
            for (const finding of unit.findings) {
                const range = new vscode.Range(
                    finding.lineStart,
                    finding.colStart || 0,
                    finding.lineEnd,
                    finding.colEnd || document.lineAt(finding.lineEnd).text.length
                );
                
                const severity = this.toVscodeSeverity(finding.priority);
                const diagnostic = new vscode.Diagnostic(range, finding.message, severity);
                diagnostic.source = 'CodeVerify Pair Review';
                diagnostic.code = finding.category;
                
                if (finding.suggestion) {
                    diagnostic.relatedInformation = [
                        new vscode.DiagnosticRelatedInformation(
                            new vscode.Location(document.uri, range),
                            `Suggestion: ${finding.suggestion}`
                        )
                    ];
                }
                
                diagnostics.push(diagnostic);
            }
        }
        
        this.diagnosticCollection.set(document.uri, diagnostics);
    }
    
    private toVscodeSeverity(priority: ReviewPriority): vscode.DiagnosticSeverity {
        switch (priority) {
            case ReviewPriority.CRITICAL:
            case ReviewPriority.HIGH:
                return vscode.DiagnosticSeverity.Error;
            case ReviewPriority.MEDIUM:
                return vscode.DiagnosticSeverity.Warning;
            case ReviewPriority.LOW:
                return vscode.DiagnosticSeverity.Information;
            case ReviewPriority.SUGGESTION:
                return vscode.DiagnosticSeverity.Hint;
            default:
                return vscode.DiagnosticSeverity.Warning;
        }
    }
    
    /**
     * Update decorations
     */
    private updateDecorations(document: vscode.TextDocument): void {
        const editor = vscode.window.visibleTextEditors.find(
            e => e.document.uri.toString() === document.uri.toString()
        );
        if (!editor) return;
        
        const units = this.documentUnits.get(document.uri.toString());
        if (!units) return;
        
        const verified: vscode.Range[] = [];
        const pending: vscode.Range[] = [];
        const reviewing: vscode.Range[] = [];
        const error: vscode.Range[] = [];
        
        for (const unit of units.values()) {
            const lineRange = new vscode.Range(unit.range.start.line, 0, unit.range.start.line, 0);
            
            switch (unit.status) {
                case 'verified':
                    verified.push(lineRange);
                    break;
                case 'pending':
                    pending.push(lineRange);
                    break;
                case 'verifying':
                    reviewing.push(lineRange);
                    break;
                case 'error':
                    error.push(lineRange);
                    break;
            }
        }
        
        editor.setDecorations(this.verifiedDecoration, verified);
        editor.setDecorations(this.pendingDecoration, pending);
        editor.setDecorations(this.reviewingDecoration, reviewing);
        editor.setDecorations(this.errorDecoration, error);
    }
    
    private createDecoration(color: string, icon: string, tooltip: string): vscode.TextEditorDecorationType {
        return vscode.window.createTextEditorDecorationType({
            gutterIconPath: this.createIconUri(color, icon),
            gutterIconSize: 'contain',
            overviewRulerColor: color,
            overviewRulerLane: vscode.OverviewRulerLane.Right,
        });
    }
    
    private createIconUri(color: string, text: string): vscode.Uri {
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="${color}">
            <text x="8" y="12" font-size="10" text-anchor="middle">${text}</text>
        </svg>`;
        return vscode.Uri.parse(`data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`);
    }
    
    private hashCode(str: string): string {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return hash.toString(16);
    }
    
    private updateStatusBar(): void {
        if (!this.enabled) {
            this.statusBarItem.text = '$(circle-slash) Pair Review: Off';
            this.statusBarItem.tooltip = 'AI Pair Reviewer: Disabled\nClick to enable';
            this.statusBarItem.backgroundColor = undefined;
        } else if (this.activeReviews.size > 0) {
            this.statusBarItem.text = `$(sync~spin) Reviewing: ${this.activeReviews.size}`;
            this.statusBarItem.tooltip = `AI Pair Reviewer: ${this.activeReviews.size} active reviews`;
            this.statusBarItem.backgroundColor = undefined;
        } else {
            let totalFindings = 0;
            let totalUnits = 0;
            let verifiedUnits = 0;
            
            for (const units of this.documentUnits.values()) {
                for (const unit of units.values()) {
                    totalUnits++;
                    totalFindings += unit.findings.length;
                    if (unit.status === 'verified') verifiedUnits++;
                }
            }
            
            const icon = totalFindings > 0 ? '$(warning)' : '$(pass)';
            this.statusBarItem.text = `${icon} PR: ${verifiedUnits}/${totalUnits}`;
            this.statusBarItem.tooltip = `AI Pair Reviewer: ${verifiedUnits} verified, ${totalFindings} findings`;
            this.statusBarItem.backgroundColor = totalFindings > 0 
                ? new vscode.ThemeColor('statusBarItem.warningBackground')
                : undefined;
        }
    }
    
    private isSupportedLanguage(languageId: string): boolean {
        return ['python', 'typescript', 'javascript', 'typescriptreact', 'javascriptreact'].includes(languageId);
    }
    
    /**
     * Get statistics
     */
    getStatistics(): object {
        let totalUnits = 0;
        let verified = 0;
        let pending = 0;
        let totalFindings = 0;
        
        for (const units of this.documentUnits.values()) {
            for (const unit of units.values()) {
                totalUnits++;
                totalFindings += unit.findings.length;
                if (unit.status === 'verified') verified++;
                else if (unit.status === 'pending' || unit.status === 'verifying') pending++;
            }
        }
        
        return {
            totalUnits,
            verified,
            pending,
            totalFindings,
            cacheSize: this.cache.size,
            activeReviews: this.activeReviews.size,
            feedbackStats: this.feedbackLearner.getStatistics(),
        };
    }
    
    dispose(): void {
        this.throttler.cancelAll();
        this.diagnosticCollection.dispose();
        this.verifiedDecoration.dispose();
        this.pendingDecoration.dispose();
        this.reviewingDecoration.dispose();
        this.errorDecoration.dispose();
        this.suggestionDecoration.dispose();
        this.statusBarItem.dispose();
        this._onReviewComplete.dispose();
        
        for (const disposable of this.disposables) {
            disposable.dispose();
        }
    }
}

/**
 * CodeLens provider for pair review status
 */
class PairReviewerCodeLensProvider implements vscode.CodeLensProvider {
    private _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
    readonly onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;
    
    constructor(private provider: PairReviewerProvider) {
        // Refresh CodeLenses when review completes
        provider.onReviewComplete(() => {
            this._onDidChangeCodeLenses.fire();
        });
    }
    
    provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
        const codeLenses: vscode.CodeLens[] = [];
        const units = this.provider.getUnits(document.uri);
        
        if (!units) return codeLenses;
        
        for (const unit of units.values()) {
            const range = new vscode.Range(unit.range.start.line, 0, unit.range.start.line, 0);
            
            // Status lens
            let statusText: string;
            let command: string;
            
            switch (unit.status) {
                case 'verified':
                    statusText = unit.findings.length > 0 
                        ? `$(warning) ${unit.findings.length} issue(s)`
                        : '$(pass) Verified';
                    command = 'codeverify.showFindings';
                    break;
                case 'verifying':
                    statusText = '$(sync~spin) Reviewing...';
                    command = 'codeverify.showFindings';
                    break;
                case 'pending':
                    statusText = '$(clock) Pending review';
                    command = 'codeverify.reviewUnit';
                    break;
                default:
                    statusText = '$(question) Unknown';
                    command = 'codeverify.reviewUnit';
            }
            
            codeLenses.push(new vscode.CodeLens(range, {
                title: statusText,
                command,
                arguments: [unit],
            }));
            
            // Quick fix lens if there are findings with fixes
            const fixableFindings = unit.findings.filter(f => f.fixCode);
            if (fixableFindings.length > 0) {
                codeLenses.push(new vscode.CodeLens(range, {
                    title: `$(lightbulb) ${fixableFindings.length} fix available`,
                    command: 'codeverify.applyFixes',
                    arguments: [unit, fixableFindings],
                }));
            }
        }
        
        return codeLenses;
    }
}
