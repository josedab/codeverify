/**
 * Continuous Verification Provider
 * 
 * Provides real-time streaming verification as developers type,
 * with incremental analysis, debouncing, and constraint caching.
 */

import * as vscode from 'vscode';
import { CodeVerifyClient, Finding } from '../client';

// Verification modes with different latency/depth tradeoffs
export enum VerificationMode {
    QUICK = 'quick',      // Fast syntax/pattern checks (~100ms)
    STANDARD = 'standard', // Standard analysis (~300ms)
    DEEP = 'deep',        // Full formal verification (~500ms+)
}

// Verification status for code regions
export enum VerificationStatus {
    PENDING = 'pending',
    VERIFYING = 'verifying',
    VERIFIED = 'verified',
    FAILED = 'failed',
    UNKNOWN = 'unknown',
}

// Represents a verifiable unit of code
interface VerifiableUnit {
    id: string;
    type: 'function' | 'class' | 'method' | 'block' | 'statement';
    name: string;
    range: vscode.Range;
    status: VerificationStatus;
    lastVerified?: Date;
    findings: Finding[];
    constraints?: string[];  // Z3 constraints
    dependencies: string[];  // IDs of units this depends on
}

// Cached verification result
interface CachedResult {
    hash: string;
    findings: Finding[];
    constraints: string[];
    timestamp: Date;
    ttl: number;
}

// Change event for incremental analysis
interface CodeChange {
    range: vscode.Range;
    text: string;
    affectedUnits: string[];
}

export class ContinuousVerificationProvider implements vscode.Disposable {
    private client: CodeVerifyClient;
    private enabled = false;
    private mode: VerificationMode = VerificationMode.STANDARD;
    
    // Caching
    private cache = new Map<string, CachedResult>();
    private maxCacheSize = 1000;
    private cacheTTL = 300000; // 5 minutes
    
    // Debouncing
    private debounceTimers = new Map<string, NodeJS.Timeout>();
    private debounceDelays: Record<VerificationMode, number> = {
        [VerificationMode.QUICK]: 100,
        [VerificationMode.STANDARD]: 300,
        [VerificationMode.DEEP]: 500,
    };
    
    // Tracked documents and units
    private documentUnits = new Map<string, Map<string, VerifiableUnit>>();
    private pendingVerifications = new Set<string>();
    
    // Decoration types
    private verifiedDecoration: vscode.TextEditorDecorationType;
    private pendingDecoration: vscode.TextEditorDecorationType;
    private verifyingDecoration: vscode.TextEditorDecorationType;
    private failedDecoration: vscode.TextEditorDecorationType;
    private heatmapDecorations: vscode.TextEditorDecorationType[] = [];
    
    // Status bar
    private statusBarItem: vscode.StatusBarItem;
    
    // Event emitters
    private _onVerificationComplete = new vscode.EventEmitter<{
        uri: vscode.Uri;
        units: VerifiableUnit[];
    }>();
    readonly onVerificationComplete = this._onVerificationComplete.event;
    
    private _onStatusChange = new vscode.EventEmitter<{
        unitId: string;
        status: VerificationStatus;
    }>();
    readonly onStatusChange = this._onStatusChange.event;
    
    private disposables: vscode.Disposable[] = [];

    constructor(client: CodeVerifyClient) {
        this.client = client;
        
        // Initialize decorations
        this.verifiedDecoration = this.createDecorationType('#2ecc71', '✓');
        this.pendingDecoration = this.createDecorationType('#95a5a6', '○');
        this.verifyingDecoration = this.createDecorationType('#3498db', '⟳');
        this.failedDecoration = this.createDecorationType('#e74c3c', '✗');
        
        // Initialize heat map colors (green -> yellow -> red)
        for (let i = 0; i <= 10; i++) {
            const r = Math.round(Math.min(255, (i / 5) * 255));
            const g = Math.round(Math.max(0, 255 - ((i - 5) / 5) * 255));
            const color = `rgba(${r}, ${g}, 0, 0.15)`;
            this.heatmapDecorations.push(
                vscode.window.createTextEditorDecorationType({
                    backgroundColor: color,
                    isWholeLine: true,
                })
            );
        }
        
        // Status bar
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            90
        );
        this.statusBarItem.command = 'codeverify.toggleContinuousMode';
        this.updateStatusBar();
        this.statusBarItem.show();
        
        this.disposables.push(this.statusBarItem);
    }

    /**
     * Enable continuous verification
     */
    enable(): void {
        this.enabled = true;
        this.updateStatusBar();
        
        // Start monitoring active editor
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            this.initializeDocument(editor.document);
        }
    }

    /**
     * Disable continuous verification
     */
    disable(): void {
        this.enabled = false;
        this.clearAllTimers();
        this.updateStatusBar();
    }

    /**
     * Toggle continuous verification
     */
    toggle(): void {
        if (this.enabled) {
            this.disable();
        } else {
            this.enable();
        }
    }

    /**
     * Set verification mode
     */
    setMode(mode: VerificationMode): void {
        this.mode = mode;
        this.updateStatusBar();
    }

    /**
     * Initialize document tracking
     */
    async initializeDocument(document: vscode.TextDocument): Promise<void> {
        if (!this.enabled || !this.isSupportedLanguage(document.languageId)) {
            return;
        }

        const uri = document.uri.toString();
        
        // Parse document structure
        const units = await this.parseDocumentUnits(document);
        this.documentUnits.set(uri, units);
        
        // Queue initial verification
        for (const unit of units.values()) {
            this.queueVerification(document, unit);
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

        // Determine affected units
        const affectedUnits = this.findAffectedUnits(event, units);
        
        // Mark affected units as pending
        for (const unitId of affectedUnits) {
            const unit = units.get(unitId);
            if (unit) {
                unit.status = VerificationStatus.PENDING;
                this._onStatusChange.fire({ unitId, status: VerificationStatus.PENDING });
            }
        }
        
        // Re-parse document structure
        units = await this.parseDocumentUnits(event.document);
        this.documentUnits.set(uri, units);
        
        // Queue verification for affected units
        for (const unitId of affectedUnits) {
            const unit = units.get(unitId);
            if (unit) {
                this.queueVerification(event.document, unit);
            }
        }
        
        // Update decorations
        this.updateDecorations(event.document);
    }

    /**
     * Queue verification with debouncing
     */
    private queueVerification(document: vscode.TextDocument, unit: VerifiableUnit): void {
        const key = `${document.uri.toString()}:${unit.id}`;
        
        // Clear existing timer
        const existingTimer = this.debounceTimers.get(key);
        if (existingTimer) {
            clearTimeout(existingTimer);
        }
        
        // Check cache
        const cacheKey = this.getCacheKey(document, unit);
        const cached = this.cache.get(cacheKey);
        if (cached && Date.now() - cached.timestamp.getTime() < cached.ttl) {
            // Use cached result
            unit.findings = cached.findings;
            unit.constraints = cached.constraints;
            unit.status = cached.findings.length > 0 ? 
                VerificationStatus.FAILED : VerificationStatus.VERIFIED;
            unit.lastVerified = cached.timestamp;
            this.updateDecorations(document);
            return;
        }
        
        // Set debounce timer
        const delay = this.debounceDelays[this.mode];
        this.debounceTimers.set(key, setTimeout(async () => {
            await this.verifyUnit(document, unit);
            this.debounceTimers.delete(key);
        }, delay));
    }

    /**
     * Verify a single unit
     */
    private async verifyUnit(document: vscode.TextDocument, unit: VerifiableUnit): Promise<void> {
        const uri = document.uri.toString();
        const key = `${uri}:${unit.id}`;
        
        // Mark as verifying
        unit.status = VerificationStatus.VERIFYING;
        this._onStatusChange.fire({ unitId: unit.id, status: VerificationStatus.VERIFYING });
        this.pendingVerifications.add(key);
        this.updateDecorations(document);
        this.updateStatusBar();
        
        try {
            // Get code for unit
            const code = document.getText(unit.range);
            
            // Call verification
            const findings = await this.client.analyzeCode(code, document.languageId);
            
            // Adjust line numbers to unit offset
            const adjustedFindings = findings.map(f => ({
                ...f,
                line_start: (f.line_start || 1) + unit.range.start.line,
                line_end: (f.line_end || f.line_start || 1) + unit.range.start.line,
            }));
            
            // Update unit
            unit.findings = adjustedFindings;
            unit.status = adjustedFindings.length > 0 ? 
                VerificationStatus.FAILED : VerificationStatus.VERIFIED;
            unit.lastVerified = new Date();
            
            // Cache result
            const cacheKey = this.getCacheKey(document, unit);
            this.cache.set(cacheKey, {
                hash: this.hashCode(code),
                findings: adjustedFindings,
                constraints: unit.constraints || [],
                timestamp: new Date(),
                ttl: this.cacheTTL,
            });
            
            // Enforce cache size limit
            if (this.cache.size > this.maxCacheSize) {
                const oldestKey = this.cache.keys().next().value;
                if (oldestKey) {
                    this.cache.delete(oldestKey);
                }
            }
            
            // Fire completion event
            this._onVerificationComplete.fire({
                uri: document.uri,
                units: [unit],
            });
            
        } catch (error) {
            unit.status = VerificationStatus.UNKNOWN;
            console.error(`Verification failed for ${unit.id}:`, error);
        } finally {
            this.pendingVerifications.delete(key);
            this._onStatusChange.fire({ unitId: unit.id, status: unit.status });
            this.updateDecorations(document);
            this.updateStatusBar();
        }
    }

    /**
     * Parse document into verifiable units
     */
    private async parseDocumentUnits(
        document: vscode.TextDocument
    ): Promise<Map<string, VerifiableUnit>> {
        const units = new Map<string, VerifiableUnit>();
        const text = document.getText();
        const languageId = document.languageId;
        
        // Simple regex-based parsing (would use tree-sitter in production)
        if (languageId === 'python') {
            // Match Python functions and classes
            const functionRegex = /^(async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->[^:]+)?:/gm;
            const classRegex = /^class\s+(\w+)(?:\([^)]*\))?\s*:/gm;
            
            let match;
            while ((match = functionRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findBlockEnd(document, startPos);
                const id = `func:${match[2]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    type: 'function',
                    name: match[2],
                    range: new vscode.Range(startPos, endPos),
                    status: VerificationStatus.PENDING,
                    findings: [],
                    dependencies: [],
                });
            }
            
            while ((match = classRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findBlockEnd(document, startPos);
                const id = `class:${match[1]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    type: 'class',
                    name: match[1],
                    range: new vscode.Range(startPos, endPos),
                    status: VerificationStatus.PENDING,
                    findings: [],
                    dependencies: [],
                });
            }
        } else if (['typescript', 'javascript', 'typescriptreact', 'javascriptreact'].includes(languageId)) {
            // Match TypeScript/JavaScript functions and classes
            const functionRegex = /(?:export\s+)?(?:async\s+)?function\s+(\w+)/gm;
            const classRegex = /(?:export\s+)?class\s+(\w+)/gm;
            const arrowRegex = /(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>/gm;
            
            let match;
            while ((match = functionRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findBraceBlockEnd(document, startPos);
                const id = `func:${match[1]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    type: 'function',
                    name: match[1],
                    range: new vscode.Range(startPos, endPos),
                    status: VerificationStatus.PENDING,
                    findings: [],
                    dependencies: [],
                });
            }
            
            while ((match = classRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findBraceBlockEnd(document, startPos);
                const id = `class:${match[1]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    type: 'class',
                    name: match[1],
                    range: new vscode.Range(startPos, endPos),
                    status: VerificationStatus.PENDING,
                    findings: [],
                    dependencies: [],
                });
            }
            
            while ((match = arrowRegex.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = this.findArrowFunctionEnd(document, startPos);
                const id = `arrow:${match[1]}:${startPos.line}`;
                
                units.set(id, {
                    id,
                    type: 'function',
                    name: match[1],
                    range: new vscode.Range(startPos, endPos),
                    status: VerificationStatus.PENDING,
                    findings: [],
                    dependencies: [],
                });
            }
        }
        
        return units;
    }

    /**
     * Find Python block end by indentation
     */
    private findBlockEnd(document: vscode.TextDocument, start: vscode.Position): vscode.Position {
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

    /**
     * Find brace-delimited block end
     */
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

    /**
     * Find arrow function end
     */
    private findArrowFunctionEnd(document: vscode.TextDocument, start: vscode.Position): vscode.Position {
        const text = document.getText();
        const startOffset = document.offsetAt(start);
        
        // Find the => first
        const arrowIndex = text.indexOf('=>', startOffset);
        if (arrowIndex === -1) return start;
        
        // Check if it's a block or expression
        let afterArrow = arrowIndex + 2;
        while (afterArrow < text.length && /\s/.test(text[afterArrow])) {
            afterArrow++;
        }
        
        if (text[afterArrow] === '{') {
            return this.findBraceBlockEnd(document, document.positionAt(afterArrow));
        }
        
        // Expression body - find the end (semicolon or newline)
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

    /**
     * Get indentation level
     */
    private getIndentLevel(line: string): number {
        const match = line.match(/^(\s*)/);
        return match ? match[1].length : 0;
    }

    /**
     * Find units affected by changes
     */
    private findAffectedUnits(
        event: vscode.TextDocumentChangeEvent,
        units: Map<string, VerifiableUnit>
    ): string[] {
        const affected = new Set<string>();
        
        for (const change of event.contentChanges) {
            const changeRange = 'range' in change ? change.range : null;
            if (!changeRange) continue;
            
            for (const [id, unit] of units) {
                // Check if change overlaps with unit
                if (this.rangesOverlap(changeRange, unit.range)) {
                    affected.add(id);
                    
                    // Also add dependent units
                    for (const depId of unit.dependencies) {
                        affected.add(depId);
                    }
                }
            }
        }
        
        return Array.from(affected);
    }

    /**
     * Check if two ranges overlap
     */
    private rangesOverlap(a: vscode.Range, b: vscode.Range): boolean {
        return !(a.end.isBefore(b.start) || b.end.isBefore(a.start));
    }

    /**
     * Update decorations for document
     */
    private updateDecorations(document: vscode.TextDocument): void {
        const editor = vscode.window.visibleTextEditors.find(
            e => e.document.uri.toString() === document.uri.toString()
        );
        if (!editor) return;
        
        const uri = document.uri.toString();
        const units = this.documentUnits.get(uri);
        if (!units) return;
        
        const verified: vscode.Range[] = [];
        const pending: vscode.Range[] = [];
        const verifying: vscode.Range[] = [];
        const failed: vscode.Range[] = [];
        
        for (const unit of units.values()) {
            switch (unit.status) {
                case VerificationStatus.VERIFIED:
                    verified.push(new vscode.Range(
                        unit.range.start.line, 0,
                        unit.range.start.line, 0
                    ));
                    break;
                case VerificationStatus.PENDING:
                    pending.push(new vscode.Range(
                        unit.range.start.line, 0,
                        unit.range.start.line, 0
                    ));
                    break;
                case VerificationStatus.VERIFYING:
                    verifying.push(new vscode.Range(
                        unit.range.start.line, 0,
                        unit.range.start.line, 0
                    ));
                    break;
                case VerificationStatus.FAILED:
                    failed.push(new vscode.Range(
                        unit.range.start.line, 0,
                        unit.range.start.line, 0
                    ));
                    break;
            }
        }
        
        editor.setDecorations(this.verifiedDecoration, verified);
        editor.setDecorations(this.pendingDecoration, pending);
        editor.setDecorations(this.verifyingDecoration, verifying);
        editor.setDecorations(this.failedDecoration, failed);
    }

    /**
     * Show verification heat map
     */
    showHeatMap(editor: vscode.TextEditor): void {
        const uri = editor.document.uri.toString();
        const units = this.documentUnits.get(uri);
        if (!units) return;
        
        // Clear all heat map decorations
        for (const decoration of this.heatmapDecorations) {
            editor.setDecorations(decoration, []);
        }
        
        // Calculate heat scores based on findings and age
        const heatRanges: vscode.Range[][] = Array.from({ length: 11 }, () => []);
        
        for (const unit of units.values()) {
            // Calculate heat score (0-10)
            let heat = 0;
            
            // More findings = more heat
            heat += Math.min(5, unit.findings.length);
            
            // Critical/high severity = more heat
            for (const finding of unit.findings) {
                if (finding.severity === 'critical') heat += 2;
                else if (finding.severity === 'high') heat += 1;
            }
            
            // Not verified recently = more heat
            if (!unit.lastVerified) {
                heat += 3;
            } else {
                const age = Date.now() - unit.lastVerified.getTime();
                if (age > 60000) heat += 1; // >1 min
                if (age > 300000) heat += 1; // >5 min
            }
            
            heat = Math.min(10, Math.max(0, heat));
            
            // Add line ranges
            for (let line = unit.range.start.line; line <= unit.range.end.line; line++) {
                heatRanges[heat].push(new vscode.Range(line, 0, line, 0));
            }
        }
        
        // Apply decorations
        for (let i = 0; i <= 10; i++) {
            editor.setDecorations(this.heatmapDecorations[i], heatRanges[i]);
        }
    }

    /**
     * Hide heat map
     */
    hideHeatMap(editor: vscode.TextEditor): void {
        for (const decoration of this.heatmapDecorations) {
            editor.setDecorations(decoration, []);
        }
    }

    /**
     * Get constraint visualization for a unit
     */
    getConstraintVisualization(document: vscode.TextDocument, position: vscode.Position): {
        constraints: string[];
        variables: string[];
        satisfiable: boolean;
    } | null {
        const uri = document.uri.toString();
        const units = this.documentUnits.get(uri);
        if (!units) return null;
        
        // Find unit at position
        for (const unit of units.values()) {
            if (unit.range.contains(position)) {
                return {
                    constraints: unit.constraints || [],
                    variables: this.extractVariables(unit.constraints || []),
                    satisfiable: unit.status === VerificationStatus.VERIFIED,
                };
            }
        }
        
        return null;
    }

    /**
     * Extract variable names from constraints
     */
    private extractVariables(constraints: string[]): string[] {
        const variables = new Set<string>();
        const varRegex = /\b([a-zA-Z_][a-zA-Z0-9_]*)\b/g;
        
        for (const constraint of constraints) {
            let match;
            while ((match = varRegex.exec(constraint)) !== null) {
                const name = match[1];
                // Filter out common keywords
                if (!['And', 'Or', 'Not', 'Implies', 'ForAll', 'Exists', 'True', 'False'].includes(name)) {
                    variables.add(name);
                }
            }
        }
        
        return Array.from(variables);
    }

    /**
     * Get verification statistics
     */
    getStatistics(): {
        totalUnits: number;
        verifiedCount: number;
        pendingCount: number;
        failedCount: number;
        cacheHitRate: number;
    } {
        let total = 0;
        let verified = 0;
        let pending = 0;
        let failed = 0;
        
        for (const units of this.documentUnits.values()) {
            for (const unit of units.values()) {
                total++;
                switch (unit.status) {
                    case VerificationStatus.VERIFIED:
                        verified++;
                        break;
                    case VerificationStatus.PENDING:
                    case VerificationStatus.VERIFYING:
                        pending++;
                        break;
                    case VerificationStatus.FAILED:
                        failed++;
                        break;
                }
            }
        }
        
        return {
            totalUnits: total,
            verifiedCount: verified,
            pendingCount: pending,
            failedCount: failed,
            cacheHitRate: 0, // Would track this in production
        };
    }

    /**
     * Get cache key for a unit
     */
    private getCacheKey(document: vscode.TextDocument, unit: VerifiableUnit): string {
        const code = document.getText(unit.range);
        return `${document.languageId}:${this.hashCode(code)}`;
    }

    /**
     * Simple hash function
     */
    private hashCode(str: string): string {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return hash.toString(16);
    }

    /**
     * Create decoration type with gutter icon
     */
    private createDecorationType(color: string, icon: string): vscode.TextEditorDecorationType {
        return vscode.window.createTextEditorDecorationType({
            gutterIconPath: this.createIconSvg(color, icon),
            gutterIconSize: 'contain',
        });
    }

    /**
     * Create SVG icon for gutter
     */
    private createIconSvg(color: string, text: string): vscode.Uri {
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="${color}">
            <text x="8" y="12" font-size="12" text-anchor="middle">${text}</text>
        </svg>`;
        
        return vscode.Uri.parse(`data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`);
    }

    /**
     * Update status bar
     */
    private updateStatusBar(): void {
        if (!this.enabled) {
            this.statusBarItem.text = '$(circle-slash) CV: Off';
            this.statusBarItem.tooltip = 'Continuous Verification: Disabled\nClick to enable';
            this.statusBarItem.backgroundColor = undefined;
        } else if (this.pendingVerifications.size > 0) {
            this.statusBarItem.text = `$(sync~spin) CV: ${this.pendingVerifications.size}`;
            this.statusBarItem.tooltip = `Continuous Verification: ${this.pendingVerifications.size} pending`;
            this.statusBarItem.backgroundColor = undefined;
        } else {
            const stats = this.getStatistics();
            const icon = stats.failedCount > 0 ? '$(warning)' : '$(pass)';
            this.statusBarItem.text = `${icon} CV: ${stats.verifiedCount}/${stats.totalUnits}`;
            this.statusBarItem.tooltip = `Continuous Verification: ${stats.verifiedCount} verified, ${stats.failedCount} failed`;
            this.statusBarItem.backgroundColor = stats.failedCount > 0 
                ? new vscode.ThemeColor('statusBarItem.warningBackground')
                : undefined;
        }
    }

    /**
     * Clear all debounce timers
     */
    private clearAllTimers(): void {
        for (const timer of this.debounceTimers.values()) {
            clearTimeout(timer);
        }
        this.debounceTimers.clear();
    }

    /**
     * Check if language is supported
     */
    private isSupportedLanguage(languageId: string): boolean {
        return ['python', 'typescript', 'javascript', 'typescriptreact', 'javascriptreact'].includes(languageId);
    }

    /**
     * Dispose resources
     */
    dispose(): void {
        this.clearAllTimers();
        this.verifiedDecoration.dispose();
        this.pendingDecoration.dispose();
        this.verifyingDecoration.dispose();
        this.failedDecoration.dispose();
        for (const decoration of this.heatmapDecorations) {
            decoration.dispose();
        }
        this.statusBarItem.dispose();
        this._onVerificationComplete.dispose();
        this._onStatusChange.dispose();
        for (const disposable of this.disposables) {
            disposable.dispose();
        }
    }
}
