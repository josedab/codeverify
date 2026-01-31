/**
 * Code Action Provider
 * 
 * Provides quick fixes for CodeVerify findings.
 */

import * as vscode from 'vscode';
import { CodeVerifyClient, Finding } from '../client';

export class CodeActionProvider implements vscode.CodeActionProvider {
    private client: CodeVerifyClient;

    constructor(client: CodeVerifyClient) {
        this.client = client;
    }

    provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
        token: vscode.CancellationToken
    ): vscode.ProviderResult<(vscode.CodeAction | vscode.Command)[]> {
        const actions: vscode.CodeAction[] = [];

        // Find CodeVerify diagnostics in range
        const codeverifyDiagnostics = context.diagnostics.filter(d => d.source === 'CodeVerify');

        for (const diagnostic of codeverifyDiagnostics) {
            const finding = (diagnostic as any).finding as Finding | undefined;
            
            if (finding) {
                // Add "Apply Fix" action if fix is available
                if (finding.fix_suggestion) {
                    const fixAction = new vscode.CodeAction(
                        `Fix: ${finding.title}`,
                        vscode.CodeActionKind.QuickFix
                    );
                    fixAction.diagnostics = [diagnostic];
                    fixAction.edit = this.createFixEdit(document, finding);
                    fixAction.isPreferred = true;
                    actions.push(fixAction);
                }

                // Add "Dismiss" action
                const dismissAction = new vscode.CodeAction(
                    `Dismiss: ${finding.title}`,
                    vscode.CodeActionKind.QuickFix
                );
                dismissAction.diagnostics = [diagnostic];
                dismissAction.command = {
                    command: 'codeverify.dismissFinding',
                    title: 'Dismiss Finding',
                    arguments: [finding]
                };
                actions.push(dismissAction);

                // Add "Report False Positive" action
                const reportAction = new vscode.CodeAction(
                    `Report False Positive`,
                    vscode.CodeActionKind.QuickFix
                );
                reportAction.diagnostics = [diagnostic];
                reportAction.command = {
                    command: 'codeverify.reportFalsePositive',
                    title: 'Report False Positive',
                    arguments: [finding]
                };
                actions.push(reportAction);

                // Add "Learn More" action
                const learnMoreAction = new vscode.CodeAction(
                    `Learn more about ${finding.category}`,
                    vscode.CodeActionKind.QuickFix
                );
                learnMoreAction.diagnostics = [diagnostic];
                learnMoreAction.command = {
                    command: 'vscode.open',
                    title: 'Learn More',
                    arguments: [vscode.Uri.parse(`https://codeverify.dev/docs/findings/${finding.category}`)]
                };
                actions.push(learnMoreAction);
            }
        }

        return actions;
    }

    /**
     * Create a workspace edit for applying a fix
     */
    private createFixEdit(document: vscode.TextDocument, finding: Finding): vscode.WorkspaceEdit {
        const edit = new vscode.WorkspaceEdit();
        
        if (!finding.fix_suggestion) {
            return edit;
        }

        const startLine = (finding.line_start || 1) - 1;
        const endLine = (finding.line_end || finding.line_start || 1) - 1;

        // Get the indentation of the original line
        const originalLine = document.lineAt(startLine);
        const indentation = originalLine.text.match(/^\s*/)?.[0] || '';

        // Apply indentation to fix suggestion
        const fixLines = finding.fix_suggestion.split('\n');
        const indentedFix = fixLines.map((line, i) => {
            if (i === 0 || line.trim() === '') {
                return line;
            }
            return indentation + line;
        }).join('\n');

        const range = new vscode.Range(
            startLine, 0,
            endLine + 1, 0
        );

        edit.replace(document.uri, range, indentedFix + '\n');

        return edit;
    }
}
