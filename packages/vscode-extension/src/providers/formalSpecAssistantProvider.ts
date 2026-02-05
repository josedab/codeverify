/**
 * Formal Specification Assistant Provider
 *
 * Provides VS Code integration for natural language to Z3 conversion.
 * Makes formal verification accessible through intuitive UI.
 */

import * as vscode from 'vscode';
import { CodeVerifyClient, NLToZ3Result, SpecTemplate } from '../client';
import { logger } from '../logger';

interface SpecHistoryEntry {
    naturalLanguage: string;
    result: NLToZ3Result;
    timestamp: number;
}

export class FormalSpecAssistantProvider implements vscode.Disposable {
    private readonly client: CodeVerifyClient;
    private readonly outputChannel: vscode.OutputChannel;
    private specHistory: SpecHistoryEntry[] = [];
    private readonly maxHistorySize = 50;
    private templateCache: SpecTemplate[] | null = null;
    private disposables: vscode.Disposable[] = [];

    constructor(client: CodeVerifyClient) {
        this.client = client;
        this.outputChannel = vscode.window.createOutputChannel('CodeVerify Formal Specs');

        // Register code lens provider for spec suggestions
        this.disposables.push(
            vscode.languages.registerCodeLensProvider(
                ['python', 'typescript', 'javascript'],
                new SpecSuggestionCodeLensProvider(this)
            )
        );
    }

    /**
     * Convert natural language to Z3 specification
     */
    async convertToZ3(naturalLanguage: string, context?: Record<string, any>): Promise<NLToZ3Result> {
        logger.info('Converting NL to Z3', { input: naturalLanguage.substring(0, 50) });

        const result = await this.client.convertNLToZ3(naturalLanguage, context);

        // Store in history
        this.specHistory.unshift({
            naturalLanguage,
            result,
            timestamp: Date.now(),
        });

        if (this.specHistory.length > this.maxHistorySize) {
            this.specHistory.pop();
        }

        return result;
    }

    /**
     * Show interactive input for spec conversion
     */
    async showConversionInput(): Promise<void> {
        const input = await vscode.window.showInputBox({
            prompt: 'Enter a natural language specification',
            placeHolder: 'e.g., "x must be positive" or "array must not be empty"',
            ignoreFocusOut: true,
        });

        if (!input) {
            return;
        }

        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Converting specification...',
                cancellable: false,
            },
            async () => {
                try {
                    const result = await this.convertToZ3(input);
                    await this.showConversionResult(input, result);
                } catch (error) {
                    vscode.window.showErrorMessage(`Conversion failed: ${error}`);
                }
            }
        );
    }

    /**
     * Show conversion result with options
     */
    private async showConversionResult(input: string, result: NLToZ3Result): Promise<void> {
        if (!result.success) {
            const action = await vscode.window.showWarningMessage(
                `Could not convert: ${result.explanation}`,
                'Try Again',
                'Show Templates'
            );

            if (action === 'Try Again') {
                await this.showConversionInput();
            } else if (action === 'Show Templates') {
                await this.showTemplateLibrary();
            }
            return;
        }

        // Create result panel
        const panel = vscode.window.createWebviewPanel(
            'formalSpecResult',
            'Formal Specification Result',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        panel.webview.html = this.getResultWebviewContent(input, result);

        // Handle webview messages
        panel.webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'copy':
                    await vscode.env.clipboard.writeText(message.text);
                    vscode.window.showInformationMessage('Copied to clipboard');
                    break;
                case 'insert':
                    await this.insertSpecAtCursor(message.text);
                    break;
                case 'validate':
                    await this.validateSpec(result);
                    break;
                case 'refine':
                    await this.refineSpecInteractive(input, result);
                    panel.dispose();
                    break;
            }
        });
    }

    /**
     * Get webview content for result display
     */
    private getResultWebviewContent(input: string, result: NLToZ3Result): string {
        const confidenceColor = result.confidence >= 0.8 ? '#4caf50' :
                               result.confidence >= 0.5 ? '#ff9800' : '#f44336';

        return `<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 20px;
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
        }
        .section {
            margin-bottom: 20px;
            padding: 15px;
            background: var(--vscode-input-background);
            border-radius: 6px;
        }
        .section-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: var(--vscode-textLink-foreground);
        }
        .code-block {
            background: var(--vscode-textCodeBlock-background);
            padding: 10px;
            border-radius: 4px;
            font-family: var(--vscode-editor-font-family);
            white-space: pre-wrap;
            margin: 5px 0;
        }
        .confidence-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            background: ${confidenceColor};
            color: white;
        }
        .button-row {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        .variables {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .variable-badge {
            background: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        .warning {
            color: var(--vscode-editorWarning-foreground);
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h2>Specification Conversion Result</h2>

    <div class="section">
        <div class="section-title">Input</div>
        <div class="code-block">${this.escapeHtml(input)}</div>
    </div>

    <div class="section">
        <div class="section-title">
            Z3 Expression
            <span class="confidence-badge">${Math.round(result.confidence * 100)}% confident</span>
        </div>
        <div class="code-block">${this.escapeHtml(result.z3_expr || 'N/A')}</div>
        <div class="button-row">
            <button onclick="copy('${this.escapeJs(result.z3_expr || '')}')">Copy Z3</button>
            <button onclick="insert('${this.escapeJs(result.z3_expr || '')}')">Insert at Cursor</button>
        </div>
    </div>

    ${result.python_assert ? `
    <div class="section">
        <div class="section-title">Python Assertion</div>
        <div class="code-block">${this.escapeHtml(result.python_assert)}</div>
        <div class="button-row">
            <button onclick="copy('${this.escapeJs(result.python_assert)}')">Copy</button>
            <button onclick="insert('${this.escapeJs(result.python_assert)}')">Insert</button>
        </div>
    </div>
    ` : ''}

    ${result.smtlib ? `
    <div class="section">
        <div class="section-title">SMT-LIB Format</div>
        <div class="code-block">${this.escapeHtml(result.smtlib)}</div>
        <div class="button-row">
            <button onclick="copy('${this.escapeJs(result.smtlib)}')">Copy</button>
        </div>
    </div>
    ` : ''}

    ${Object.keys(result.variables).length > 0 ? `
    <div class="section">
        <div class="section-title">Variables</div>
        <div class="variables">
            ${Object.entries(result.variables).map(([name, type]) =>
                `<span class="variable-badge">${this.escapeHtml(name)}: ${this.escapeHtml(type)}</span>`
            ).join('')}
        </div>
    </div>
    ` : ''}

    ${result.ambiguities.length > 0 ? `
    <div class="section">
        <div class="section-title">Ambiguities Detected</div>
        <ul class="warning">
            ${result.ambiguities.map(a => `<li>${this.escapeHtml(a)}</li>`).join('')}
        </ul>
    </div>
    ` : ''}

    <div class="section">
        <div class="section-title">Actions</div>
        <div class="button-row">
            <button onclick="validate()">Validate with Z3</button>
            <button onclick="refine()">Refine Specification</button>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Explanation</div>
        <p>${this.escapeHtml(result.explanation)}</p>
        <p style="color: var(--vscode-descriptionForeground); font-size: 12px;">
            Processing time: ${result.processing_time_ms.toFixed(1)}ms
        </p>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        function copy(text) {
            vscode.postMessage({ command: 'copy', text });
        }

        function insert(text) {
            vscode.postMessage({ command: 'insert', text });
        }

        function validate() {
            vscode.postMessage({ command: 'validate' });
        }

        function refine() {
            vscode.postMessage({ command: 'refine' });
        }
    </script>
</body>
</html>`;
    }

    /**
     * Validate a specification with Z3
     */
    async validateSpec(result: NLToZ3Result): Promise<void> {
        if (!result.z3_expr) {
            vscode.window.showWarningMessage('No Z3 expression to validate');
            return;
        }

        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Validating with Z3...',
            },
            async () => {
                try {
                    const validation = await this.client.validateZ3Spec(
                        result.z3_expr!,
                        result.variables
                    );

                    if (validation.is_satisfiable) {
                        const modelStr = validation.model
                            ? `\nExample model: ${JSON.stringify(validation.model)}`
                            : '';
                        vscode.window.showInformationMessage(
                            `Specification is satisfiable.${modelStr}`
                        );
                    } else {
                        vscode.window.showWarningMessage(
                            `Specification issue: ${validation.message}`
                        );
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Validation failed: ${error}`);
                }
            }
        );
    }

    /**
     * Interactive refinement of a specification
     */
    async refineSpecInteractive(originalSpec: string, currentResult: NLToZ3Result): Promise<void> {
        const feedback = await vscode.window.showInputBox({
            prompt: 'How should the specification be refined?',
            placeHolder: 'e.g., "should be strictly greater" or "include both bounds"',
        });

        if (!feedback) {
            return;
        }

        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Refining specification...',
            },
            async () => {
                try {
                    const refined = await this.client.refineSpec(
                        originalSpec,
                        currentResult.z3_expr || '',
                        feedback
                    );
                    await this.showConversionResult(originalSpec, refined);
                } catch (error) {
                    vscode.window.showErrorMessage(`Refinement failed: ${error}`);
                }
            }
        );
    }

    /**
     * Insert specification at cursor
     */
    private async insertSpecAtCursor(text: string): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        await editor.edit((editBuilder) => {
            editBuilder.insert(editor.selection.active, text);
        });
    }

    /**
     * Show template library
     */
    async showTemplateLibrary(): Promise<void> {
        if (!this.templateCache) {
            const result = await this.client.getSpecTemplates();
            this.templateCache = result.templates;
        }

        const items = this.templateCache.map((t) => ({
            label: t.name,
            description: t.domain,
            detail: t.nl_pattern,
            template: t,
        }));

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a specification template',
            matchOnDetail: true,
        });

        if (selected) {
            await this.useTemplate(selected.template);
        }
    }

    /**
     * Use a template with user input
     */
    private async useTemplate(template: SpecTemplate): Promise<void> {
        // Extract variable placeholders
        const varPattern = /\{(\w+)\}/g;
        const variables: string[] = [];
        let match;
        while ((match = varPattern.exec(template.nl_pattern)) !== null) {
            if (!variables.includes(match[1])) {
                variables.push(match[1]);
            }
        }

        // Prompt for each variable
        const values: Record<string, string> = {};
        for (const varName of variables) {
            const value = await vscode.window.showInputBox({
                prompt: `Enter value for {${varName}}`,
                placeHolder: varName,
            });

            if (value === undefined) {
                return; // Cancelled
            }
            values[varName] = value;
        }

        // Fill template
        let spec = template.nl_pattern;
        for (const [varName, value] of Object.entries(values)) {
            spec = spec.replace(new RegExp(`\\{${varName}\\}`, 'g'), value);
        }

        // Convert
        const result = await this.convertToZ3(spec);
        await this.showConversionResult(spec, result);
    }

    /**
     * Generate specs for current function
     */
    async suggestSpecsForCurrentFunction(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        // Find function at cursor
        const position = editor.selection.active;
        const document = editor.document;
        const signature = this.extractFunctionSignature(document, position);

        if (!signature) {
            vscode.window.showWarningMessage('No function found at cursor');
            return;
        }

        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Generating specification suggestions...',
            },
            async () => {
                try {
                    const result = await this.client.suggestSpecs(signature);

                    if (result.suggestions.length === 0) {
                        vscode.window.showInformationMessage('No suggestions available');
                        return;
                    }

                    const selected = await vscode.window.showQuickPick(
                        result.suggestions.map((s) => ({ label: s, suggestion: s })),
                        {
                            placeHolder: 'Select a suggested specification',
                            canPickMany: true,
                        }
                    );

                    if (selected && selected.length > 0) {
                        // Convert selected suggestions
                        for (const item of selected) {
                            const convResult = await this.convertToZ3(item.suggestion);
                            this.outputChannel.appendLine(`\n# ${item.suggestion}`);
                            this.outputChannel.appendLine(`# Z3: ${convResult.z3_expr}`);
                            this.outputChannel.appendLine(`# Python: ${convResult.python_assert}`);
                        }
                        this.outputChannel.show();
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Suggestion failed: ${error}`);
                }
            }
        );
    }

    /**
     * Extract function signature from document
     */
    private extractFunctionSignature(
        document: vscode.TextDocument,
        position: vscode.Position
    ): string | null {
        const language = document.languageId;

        // Search backwards for function definition
        for (let line = position.line; line >= Math.max(0, position.line - 10); line--) {
            const text = document.lineAt(line).text;

            if (language === 'python') {
                const match = text.match(/def\s+\w+\s*\([^)]*\)(?:\s*->\s*\w+)?/);
                if (match) {
                    return match[0];
                }
            } else if (language === 'typescript' || language === 'javascript') {
                const match = text.match(/(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)(?:\s*:\s*\w+)?)/);
                if (match) {
                    return match[0];
                }
            }
        }

        return null;
    }

    /**
     * Show specification history
     */
    async showHistory(): Promise<void> {
        if (this.specHistory.length === 0) {
            vscode.window.showInformationMessage('No conversion history');
            return;
        }

        const items = this.specHistory.map((entry, index) => ({
            label: entry.naturalLanguage.substring(0, 50) + (entry.naturalLanguage.length > 50 ? '...' : ''),
            description: `${Math.round(entry.result.confidence * 100)}% confidence`,
            detail: entry.result.z3_expr || 'Failed',
            entry,
        }));

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a previous conversion',
        });

        if (selected) {
            await this.showConversionResult(
                selected.entry.naturalLanguage,
                selected.entry.result
            );
        }
    }

    /**
     * Get suggestions for code lens
     */
    async getSuggestionsForSignature(signature: string): Promise<string[]> {
        try {
            const result = await this.client.suggestSpecs(signature);
            return result.suggestions;
        } catch {
            return [];
        }
    }

    private escapeHtml(text: string): string {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    private escapeJs(text: string): string {
        return text
            .replace(/\\/g, '\\\\')
            .replace(/'/g, "\\'")
            .replace(/"/g, '\\"')
            .replace(/\n/g, '\\n');
    }

    dispose(): void {
        this.outputChannel.dispose();
        this.disposables.forEach((d) => d.dispose());
    }
}

/**
 * Code lens provider for spec suggestions
 */
class SpecSuggestionCodeLensProvider implements vscode.CodeLensProvider {
    private readonly provider: FormalSpecAssistantProvider;

    constructor(provider: FormalSpecAssistantProvider) {
        this.provider = provider;
    }

    provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
        const lenses: vscode.CodeLens[] = [];
        const language = document.languageId;

        for (let i = 0; i < document.lineCount; i++) {
            const line = document.lineAt(i);
            let match: RegExpMatchArray | null = null;

            if (language === 'python') {
                match = line.text.match(/def\s+(\w+)\s*\([^)]*\)/);
            } else if (language === 'typescript' || language === 'javascript') {
                match = line.text.match(/function\s+(\w+)\s*\([^)]*\)/);
            }

            if (match) {
                const range = new vscode.Range(i, 0, i, line.text.length);
                lenses.push(
                    new vscode.CodeLens(range, {
                        title: 'Suggest Specifications',
                        command: 'codeverify.suggestSpecs',
                        arguments: [document, i],
                    })
                );
            }
        }

        return lenses;
    }
}
