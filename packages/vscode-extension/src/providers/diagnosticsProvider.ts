/**
 * Diagnostics Provider
 * 
 * Converts CodeVerify findings to VS Code diagnostics.
 */

import * as vscode from 'vscode';
import { CodeVerifyClient, Finding } from '../client';

export class DiagnosticsProvider implements vscode.Disposable {
    private diagnosticCollection: vscode.DiagnosticCollection;
    private client: CodeVerifyClient;

    constructor(client: CodeVerifyClient) {
        this.client = client;
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeverify');
    }

    /**
     * Update diagnostics for a document
     */
    updateDiagnostics(uri: vscode.Uri, findings: Finding[]): void {
        const diagnostics: vscode.Diagnostic[] = findings.map(finding => {
            const range = new vscode.Range(
                (finding.line_start || 1) - 1,
                0,
                (finding.line_end || finding.line_start || 1) - 1,
                Number.MAX_VALUE
            );

            const diagnostic = new vscode.Diagnostic(
                range,
                `${finding.title}: ${finding.description}`,
                this.severityToDiagnosticSeverity(finding.severity)
            );

            diagnostic.source = 'CodeVerify';
            diagnostic.code = {
                value: finding.category,
                target: vscode.Uri.parse(`https://codeverify.dev/docs/findings/${finding.category}`),
            };

            // Add related information
            if (finding.confidence) {
                diagnostic.message += ` (${Math.round(finding.confidence * 100)}% confidence)`;
            }

            // Store finding data for code actions
            (diagnostic as any).finding = finding;

            return diagnostic;
        });

        this.diagnosticCollection.set(uri, diagnostics);
    }

    /**
     * Clear diagnostics for a document
     */
    clearDiagnostics(uri: vscode.Uri): void {
        this.diagnosticCollection.delete(uri);
    }

    /**
     * Clear all diagnostics
     */
    clearAll(): void {
        this.diagnosticCollection.clear();
    }

    /**
     * Convert severity string to VS Code DiagnosticSeverity
     */
    private severityToDiagnosticSeverity(severity: string): vscode.DiagnosticSeverity {
        switch (severity.toLowerCase()) {
            case 'critical':
            case 'high':
                return vscode.DiagnosticSeverity.Error;
            case 'medium':
                return vscode.DiagnosticSeverity.Warning;
            case 'low':
                return vscode.DiagnosticSeverity.Information;
            case 'info':
                return vscode.DiagnosticSeverity.Hint;
            default:
                return vscode.DiagnosticSeverity.Warning;
        }
    }

    dispose(): void {
        this.diagnosticCollection.dispose();
    }
}
