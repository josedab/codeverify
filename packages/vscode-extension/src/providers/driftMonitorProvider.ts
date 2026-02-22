/**
 * Drift Monitor Provider
 *
 * Monitors for behavioral drift when files change, showing diagnostics
 * for signature changes, behavior changes, and invariant violations.
 * Includes a status bar indicator for overall drift status.
 */

import * as vscode from 'vscode';

interface DriftAlert {
    drift_type: 'behavior_change' | 'signature_change' | 'invariant_violation' | 'exception_change' | 'pattern_inconsistency';
    severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
    function_name: string;
    file_path: string;
    message: string;
    old_fingerprint: string;
    new_fingerprint: string;
    commit_sha: string;
}

interface DriftReport {
    repo: string;
    commit_sha: string;
    alerts: DriftAlert[];
    functions_checked: number;
    functions_drifted: number;
    invariants_checked: number;
    invariants_violated: number;
}

interface BehavioralFingerprint {
    function_name: string;
    file_path: string;
    signature_hash: string;
    content_hash: string;
    parameters: string[];
    complexity: number;
}

export class DriftMonitorProvider implements vscode.Disposable {
    private diagnosticCollection: vscode.DiagnosticCollection;
    private statusBarItem: vscode.StatusBarItem;
    private baselines: Map<string, BehavioralFingerprint[]> = new Map();
    private disposables: vscode.Disposable[] = [];
    private isMonitoring: boolean = false;
    private driftAlerts: DriftAlert[] = [];

    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeverify-drift');
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left, 80
        );
        this.statusBarItem.command = 'codeverify.showDriftReport';
        this.updateStatusBar(0);

        this.disposables.push(
            vscode.commands.registerCommand('codeverify.enableDriftMonitor', () => this.enable()),
            vscode.commands.registerCommand('codeverify.disableDriftMonitor', () => this.disable()),
            vscode.commands.registerCommand('codeverify.setDriftBaseline', () => this.setBaseline()),
            vscode.commands.registerCommand('codeverify.showDriftReport', () => this.showReport()),
            vscode.commands.registerCommand('codeverify.acknowledgeDrift', (alertIdx: number) => {
                if (alertIdx >= 0 && alertIdx < this.driftAlerts.length) {
                    this.driftAlerts.splice(alertIdx, 1);
                    this.updateStatusBar(this.driftAlerts.length);
                }
            }),
            vscode.workspace.onDidSaveTextDocument((doc) => {
                if (this.isMonitoring) {
                    this.checkForDrift(doc);
                }
            })
        );
    }

    show(): void {
        this.statusBarItem.show();
    }

    enable(): void {
        this.isMonitoring = true;
        this.statusBarItem.show();
        vscode.window.showInformationMessage('CodeVerify Drift Monitor enabled');
    }

    disable(): void {
        this.isMonitoring = false;
        this.diagnosticCollection.clear();
        this.driftAlerts = [];
        this.updateStatusBar(0);
        vscode.window.showInformationMessage('CodeVerify Drift Monitor disabled');
    }

    async setBaseline(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const doc = editor.document;
        const fingerprints = this.extractFingerprints(doc.getText(), doc.uri.fsPath, doc.languageId);
        this.baselines.set(doc.uri.fsPath, fingerprints);
        vscode.window.showInformationMessage(
            `Baseline set: ${fingerprints.length} functions fingerprinted in ${doc.fileName}`
        );
    }

    private checkForDrift(document: vscode.TextDocument): void {
        const supportedLanguages = ['python', 'typescript', 'javascript', 'go'];
        if (!supportedLanguages.includes(document.languageId)) {
            return;
        }

        const filePath = document.uri.fsPath;
        const oldFingerprints = this.baselines.get(filePath);
        if (!oldFingerprints) {
            return;
        }

        const newFingerprints = this.extractFingerprints(
            document.getText(), filePath, document.languageId
        );

        const alerts = this.detectDrift(oldFingerprints, newFingerprints, filePath);
        const diagnostics: vscode.Diagnostic[] = [];

        for (const alert of alerts) {
            const funcLine = this.findFunctionLine(document.getText(), alert.function_name);
            const range = new vscode.Range(funcLine, 0, funcLine, Number.MAX_VALUE);

            const severity = alert.severity === 'critical' || alert.severity === 'high'
                ? vscode.DiagnosticSeverity.Error
                : alert.severity === 'medium'
                    ? vscode.DiagnosticSeverity.Warning
                    : vscode.DiagnosticSeverity.Information;

            const diagnostic = new vscode.Diagnostic(range, alert.message, severity);
            diagnostic.source = 'CodeVerify Drift';
            diagnostic.code = alert.drift_type;
            diagnostics.push(diagnostic);
        }

        this.diagnosticCollection.set(document.uri, diagnostics);

        // Update global alerts
        this.driftAlerts = this.driftAlerts
            .filter(a => a.file_path !== filePath)
            .concat(alerts);
        this.updateStatusBar(this.driftAlerts.length);
    }

    private extractFingerprints(
        content: string, filePath: string, languageId: string
    ): BehavioralFingerprint[] {
        const fingerprints: BehavioralFingerprint[] = [];
        const lines = content.split('\n');
        const funcPattern = languageId === 'python' ? /^(\s*)def\s+(\w+)\s*\(([^)]*)\)/ :
            languageId === 'go' ? /^func\s+(\w+)\s*\(([^)]*)\)/ :
            /(?:function|async function|export function)\s+(\w+)\s*\(([^)]*)\)/;

        let currentFunc: string | null = null;
        let funcLines: string[] = [];
        let params: string[] = [];

        for (const line of lines) {
            const match = line.match(funcPattern);
            if (match) {
                if (currentFunc && funcLines.length > 0) {
                    fingerprints.push(this.createFingerprint(
                        currentFunc, filePath, params, funcLines
                    ));
                }
                currentFunc = match[2] || match[1];
                const paramStr = match[3] || match[2] || '';
                params = paramStr.split(',').map(p => p.trim().split(':')[0].trim()).filter(Boolean);
                funcLines = [line];
            } else if (currentFunc) {
                funcLines.push(line);
            }
        }

        if (currentFunc && funcLines.length > 0) {
            fingerprints.push(this.createFingerprint(currentFunc, filePath, params, funcLines));
        }

        return fingerprints;
    }

    private createFingerprint(
        name: string, filePath: string, params: string[], lines: string[]
    ): BehavioralFingerprint {
        const content = lines.join('\n');
        const sigStr = `${name}(${params.join(',')})`;
        return {
            function_name: name,
            file_path: filePath,
            signature_hash: this.simpleHash(sigStr),
            content_hash: this.simpleHash(content),
            parameters: params,
            complexity: lines.length,
        };
    }

    private detectDrift(
        oldFps: BehavioralFingerprint[], newFps: BehavioralFingerprint[], filePath: string
    ): DriftAlert[] {
        const alerts: DriftAlert[] = [];
        const oldMap = new Map(oldFps.map(fp => [fp.function_name, fp]));

        for (const newFp of newFps) {
            const oldFp = oldMap.get(newFp.function_name);
            if (!oldFp) continue;
            if (oldFp.content_hash === newFp.content_hash) continue;

            if (oldFp.signature_hash !== newFp.signature_hash) {
                alerts.push({
                    drift_type: 'signature_change',
                    severity: 'high',
                    function_name: newFp.function_name,
                    file_path: filePath,
                    message: `Signature changed: ${newFp.function_name}(${oldFp.parameters.join(', ')}) → (${newFp.parameters.join(', ')})`,
                    old_fingerprint: oldFp.signature_hash,
                    new_fingerprint: newFp.signature_hash,
                    commit_sha: '',
                });
            } else {
                const complexityDelta = Math.abs(newFp.complexity - oldFp.complexity);
                alerts.push({
                    drift_type: 'behavior_change',
                    severity: complexityDelta > 10 ? 'medium' : 'low',
                    function_name: newFp.function_name,
                    file_path: filePath,
                    message: `Behavior changed in ${newFp.function_name} (complexity Δ${complexityDelta})`,
                    old_fingerprint: oldFp.content_hash,
                    new_fingerprint: newFp.content_hash,
                    commit_sha: '',
                });
            }
        }

        return alerts;
    }

    private findFunctionLine(content: string, funcName: string): number {
        const lines = content.split('\n');
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].includes(`def ${funcName}`) ||
                lines[i].includes(`function ${funcName}`) ||
                lines[i].includes(`func ${funcName}`)) {
                return i;
            }
        }
        return 0;
    }

    private async showReport(): Promise<void> {
        if (this.driftAlerts.length === 0) {
            vscode.window.showInformationMessage('No drift detected');
            return;
        }

        const items = this.driftAlerts.map((a, i) => ({
            label: `$(${a.severity === 'high' || a.severity === 'critical' ? 'error' : 'warning'}) ${a.function_name}`,
            description: a.drift_type.replace('_', ' '),
            detail: a.message,
            index: i,
        }));

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a drift alert to acknowledge',
            title: `Drift Monitor — ${this.driftAlerts.length} alerts`,
        });

        if (selected) {
            this.driftAlerts.splice(selected.index, 1);
            this.updateStatusBar(this.driftAlerts.length);
        }
    }

    private updateStatusBar(alertCount: number): void {
        if (alertCount === 0) {
            this.statusBarItem.text = '$(shield-check) No Drift';
            this.statusBarItem.backgroundColor = undefined;
        } else {
            this.statusBarItem.text = `$(alert) ${alertCount} Drift Alert${alertCount > 1 ? 's' : ''}`;
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        }
    }

    private simpleHash(str: string): string {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash |= 0;
        }
        return Math.abs(hash).toString(36).substring(0, 12);
    }

    dispose(): void {
        this.diagnosticCollection.dispose();
        this.statusBarItem.dispose();
        this.disposables.forEach(d => d.dispose());
    }
}
