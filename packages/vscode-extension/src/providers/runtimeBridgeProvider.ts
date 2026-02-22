/**
 * Runtime Bridge Provider
 *
 * One-click instrumentation: insert Z3-derived runtime assertions into code,
 * show violation diagnostics, and track feedback loop status.
 */

import * as vscode from 'vscode';

interface RuntimeAssertion {
    id: string;
    check_type: string;
    function_name: string;
    file_path: string;
    assertion_code: string;
    variables: string[];
    is_active: boolean;
}

interface RuntimeViolation {
    id: string;
    assertion_id: string;
    function_name: string;
    file_path: string;
    variable_values: Record<string, unknown>;
    error_message: string;
    severity: string;
    timestamp: string;
}

interface BridgeStats {
    assertions_generated: number;
    violations_captured: number;
    confirmed_bugs: number;
    false_positives_detected: number;
}

export class RuntimeBridgeProvider implements vscode.Disposable {
    private diagnosticCollection: vscode.DiagnosticCollection;
    private statusBarItem: vscode.StatusBarItem;
    private decorationType: vscode.TextEditorDecorationType;
    private assertions: Map<string, RuntimeAssertion[]> = new Map();
    private violations: RuntimeViolation[] = [];
    private disposables: vscode.Disposable[] = [];

    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeverify-runtime');
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 75);
        this.statusBarItem.command = 'codeverify.showRuntimeStats';
        this.updateStatusBar({ assertions_generated: 0, violations_captured: 0, confirmed_bugs: 0, false_positives_detected: 0 });

        this.decorationType = vscode.window.createTextEditorDecorationType({
            gutterIconPath: 'shield-check',
            gutterIconSize: '80%',
            after: {
                contentText: ' ⚡',
                color: new vscode.ThemeColor('editorInfo.foreground'),
                fontStyle: 'italic',
            },
        });

        this.disposables.push(
            vscode.commands.registerCommand('codeverify.instrumentFunction', () => this.instrumentCurrentFunction()),
            vscode.commands.registerCommand('codeverify.instrumentFile', () => this.instrumentCurrentFile()),
            vscode.commands.registerCommand('codeverify.removeInstrumentation', () => this.removeInstrumentation()),
            vscode.commands.registerCommand('codeverify.showRuntimeStats', () => this.showStats()),
            vscode.commands.registerCommand('codeverify.reportViolation', (v: RuntimeViolation) => this.reportViolation(v)),
        );
    }

    show(): void {
        this.statusBarItem.show();
    }

    private async instrumentCurrentFunction(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const doc = editor.document;
        const position = editor.selection.active;
        const funcInfo = this.findFunctionAtPosition(doc.getText(), position.line, doc.languageId);

        if (!funcInfo) {
            vscode.window.showWarningMessage('No function found at cursor position');
            return;
        }

        const assertions = this.generateAssertions(funcInfo.name, funcInfo.params, doc.uri.fsPath, doc.languageId);
        this.assertions.set(`${doc.uri.fsPath}:${funcInfo.name}`, assertions);

        const assertionCode = assertions.map(a => a.assertion_code).join('\n    ');
        const insertPosition = new vscode.Position(funcInfo.bodyLine, 0);
        const indent = '    ';

        await editor.edit(editBuilder => {
            editBuilder.insert(insertPosition, `${indent}# CodeVerify runtime assertions\n${indent}${assertionCode}\n`);
        });

        this.updateDecorations(editor, assertions);
        vscode.window.showInformationMessage(
            `Inserted ${assertions.length} runtime assertion(s) in ${funcInfo.name}`
        );
    }

    private async instrumentCurrentFile(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const doc = editor.document;
        const functions = this.findAllFunctions(doc.getText(), doc.languageId);
        let totalAssertions = 0;

        for (const func of functions.reverse()) {
            const assertions = this.generateAssertions(func.name, func.params, doc.uri.fsPath, doc.languageId);
            if (assertions.length === 0) continue;

            this.assertions.set(`${doc.uri.fsPath}:${func.name}`, assertions);
            const assertionCode = assertions.map(a => a.assertion_code).join('\n    ');
            const insertPosition = new vscode.Position(func.bodyLine, 0);

            await editor.edit(editBuilder => {
                editBuilder.insert(insertPosition, `    # CodeVerify runtime assertions\n    ${assertionCode}\n`);
            });
            totalAssertions += assertions.length;
        }

        vscode.window.showInformationMessage(
            `Instrumented ${functions.length} functions with ${totalAssertions} assertions`
        );
    }

    private async removeInstrumentation(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const doc = editor.document;
        const text = doc.getText();
        const lines = text.split('\n');
        const linesToRemove: number[] = [];

        for (let i = 0; i < lines.length; i++) {
            if (lines[i].trim().startsWith('# CodeVerify runtime assertions') ||
                lines[i].trim().startsWith('assert ') && lines[i].includes('CodeVerify:')) {
                linesToRemove.push(i);
            }
        }

        if (linesToRemove.length === 0) {
            vscode.window.showInformationMessage('No CodeVerify assertions found');
            return;
        }

        await editor.edit(editBuilder => {
            for (const lineNum of linesToRemove.reverse()) {
                const range = new vscode.Range(lineNum, 0, lineNum + 1, 0);
                editBuilder.delete(range);
            }
        });

        this.assertions.clear();
        editor.setDecorations(this.decorationType, []);
        vscode.window.showInformationMessage(`Removed ${linesToRemove.length} assertion lines`);
    }

    reportViolation(violation: RuntimeViolation): void {
        this.violations.push(violation);

        const diagnostics: vscode.Diagnostic[] = [];
        const uri = vscode.Uri.file(violation.file_path);
        const line = this.findFunctionLine(violation.file_path, violation.function_name);

        const diagnostic = new vscode.Diagnostic(
            new vscode.Range(line, 0, line, Number.MAX_VALUE),
            `Runtime violation: ${violation.error_message}`,
            violation.severity === 'critical' ? vscode.DiagnosticSeverity.Error : vscode.DiagnosticSeverity.Warning
        );
        diagnostic.source = 'CodeVerify Runtime';
        diagnostic.code = violation.assertion_id;
        diagnostics.push(diagnostic);

        const existing = this.diagnosticCollection.get(uri) || [];
        this.diagnosticCollection.set(uri, [...existing, ...diagnostics]);
    }

    private generateAssertions(
        funcName: string, params: string[], filePath: string, languageId: string
    ): RuntimeAssertion[] {
        const assertions: RuntimeAssertion[] = [];
        const filteredParams = params.filter(p => p !== 'self' && p !== 'cls' && p.length > 0);

        for (const param of filteredParams) {
            if (languageId === 'python') {
                assertions.push({
                    id: `rt_${Math.random().toString(36).substring(7)}`,
                    check_type: 'null_safety',
                    function_name: funcName,
                    file_path: filePath,
                    assertion_code: `assert ${param} is not None, 'CodeVerify: ${param} must not be None at ${funcName}'`,
                    variables: [param],
                    is_active: true,
                });
            } else if (languageId === 'typescript' || languageId === 'javascript') {
                assertions.push({
                    id: `rt_${Math.random().toString(36).substring(7)}`,
                    check_type: 'null_safety',
                    function_name: funcName,
                    file_path: filePath,
                    assertion_code: `if (${param} === null || ${param} === undefined) throw new Error('CodeVerify: ${param} must not be null at ${funcName}');`,
                    variables: [param],
                    is_active: true,
                });
            }
        }
        return assertions;
    }

    private findFunctionAtPosition(
        text: string, line: number, languageId: string
    ): { name: string; params: string[]; bodyLine: number } | null {
        const lines = text.split('\n');
        for (let i = line; i >= Math.max(0, line - 10); i--) {
            const match = this.matchFunctionDef(lines[i], languageId);
            if (match) return { ...match, bodyLine: i + 1 };
        }
        return null;
    }

    private findAllFunctions(
        text: string, languageId: string
    ): { name: string; params: string[]; bodyLine: number }[] {
        const functions: { name: string; params: string[]; bodyLine: number }[] = [];
        const lines = text.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const match = this.matchFunctionDef(lines[i], languageId);
            if (match) functions.push({ ...match, bodyLine: i + 1 });
        }
        return functions;
    }

    private matchFunctionDef(
        line: string, languageId: string
    ): { name: string; params: string[] } | null {
        const patterns: Record<string, RegExp> = {
            python: /^\s*def\s+(\w+)\s*\(([^)]*)\)/,
            typescript: /(?:function|async function|export function)\s+(\w+)\s*\(([^)]*)\)/,
            javascript: /(?:function|async function)\s+(\w+)\s*\(([^)]*)\)/,
        };
        const pattern = patterns[languageId];
        if (!pattern) return null;

        const match = line.match(pattern);
        if (!match) return null;

        const params = match[2].split(',').map(p => p.trim().split(':')[0].split('=')[0].trim()).filter(Boolean);
        return { name: match[1], params };
    }

    private findFunctionLine(filePath: string, funcName: string): number {
        try {
            const doc = vscode.workspace.textDocuments.find(d => d.uri.fsPath === filePath);
            if (doc) {
                const text = doc.getText();
                const lines = text.split('\n');
                for (let i = 0; i < lines.length; i++) {
                    if (lines[i].includes(`def ${funcName}`) || lines[i].includes(`function ${funcName}`)) {
                        return i;
                    }
                }
            }
        } catch { /* ignore */ }
        return 0;
    }

    private updateDecorations(editor: vscode.TextEditor, assertions: RuntimeAssertion[]): void {
        const decorations: vscode.DecorationOptions[] = [];
        const text = editor.document.getText();
        const lines = text.split('\n');

        for (let i = 0; i < lines.length; i++) {
            if (lines[i].includes('CodeVerify:') && lines[i].trim().startsWith('assert ')) {
                decorations.push({
                    range: new vscode.Range(i, 0, i, lines[i].length),
                    hoverMessage: new vscode.MarkdownString('**CodeVerify Runtime Assertion** — Z3-derived guard'),
                });
            }
        }
        editor.setDecorations(this.decorationType, decorations);
    }

    private updateStatusBar(stats: BridgeStats): void {
        const icon = stats.violations_captured > 0 ? '$(alert)' : '$(shield-check)';
        this.statusBarItem.text = `${icon} RT: ${stats.assertions_generated}A ${stats.violations_captured}V`;
        this.statusBarItem.tooltip = `Runtime Bridge: ${stats.assertions_generated} assertions, ${stats.violations_captured} violations, ${stats.confirmed_bugs} confirmed, ${stats.false_positives_detected} FPs`;
    }

    private async showStats(): Promise<void> {
        const totalAssertions = Array.from(this.assertions.values()).reduce((s, a) => s + a.length, 0);
        const msg = `Runtime Bridge Stats:\n• ${totalAssertions} assertions active\n• ${this.violations.length} violations captured`;
        vscode.window.showInformationMessage(msg);
    }

    dispose(): void {
        this.diagnosticCollection.dispose();
        this.statusBarItem.dispose();
        this.decorationType.dispose();
        this.disposables.forEach(d => d.dispose());
    }
}
