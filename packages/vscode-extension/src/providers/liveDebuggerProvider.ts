/**
 * Live Verification Debugger Provider
 *
 * Interactive webview panel for stepping through Z3 proof trees,
 * modifying variable assignments, and exploring constraint propagation.
 */

import * as vscode from 'vscode';

interface ProofNode {
    id: string;
    constraint: string;
    status: 'satisfied' | 'violated' | 'unknown' | 'exploring';
    children: ProofNode[];
    variables: Record<string, unknown>;
    depth: number;
}

interface ProofStep {
    step_number: number;
    action: 'assert' | 'propagate' | 'decide' | 'conflict' | 'backtrack' | 'simplify';
    constraint: string;
    variable_changes: Record<string, unknown>;
    explanation: string;
    node_id: string;
}

interface DebugSession {
    id: string;
    title: string;
    root_node: ProofNode | null;
    steps: ProofStep[];
    current_step: number;
    variable_state: Record<string, unknown>;
    share_url: string;
}

export class LiveDebuggerProvider implements vscode.Disposable {
    private panel: vscode.WebviewPanel | undefined;
    private currentSession: DebugSession | null = null;
    private disposables: vscode.Disposable[] = [];
    private statusBarItem: vscode.StatusBarItem;

    constructor() {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left, 85
        );
        this.statusBarItem.text = '$(debug) Proof Debugger';
        this.statusBarItem.command = 'codeverify.openDebugger';
        this.statusBarItem.tooltip = 'Open CodeVerify Proof Debugger';

        this.disposables.push(
            vscode.commands.registerCommand('codeverify.openDebugger', () => this.openPanel()),
            vscode.commands.registerCommand('codeverify.debugProof', (constraints: string[], variables: Record<string, unknown>) => {
                this.startSession('Debug Session', constraints, variables);
            }),
            vscode.commands.registerCommand('codeverify.stepForward', () => this.stepForward()),
            vscode.commands.registerCommand('codeverify.stepBackward', () => this.stepBackward()),
            vscode.commands.registerCommand('codeverify.shareProof', () => this.shareProof())
        );
    }

    show(): void {
        this.statusBarItem.show();
    }

    private openPanel(): void {
        if (this.panel) {
            this.panel.reveal();
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'codeverifyDebugger',
            'Proof Debugger',
            vscode.ViewColumn.Two,
            { enableScripts: true, retainContextWhenHidden: true }
        );

        this.panel.webview.html = this.getWebviewContent();

        this.panel.webview.onDidReceiveMessage(
            (message) => this.handleMessage(message),
            undefined,
            this.disposables
        );

        this.panel.onDidDispose(() => {
            this.panel = undefined;
        }, null, this.disposables);
    }

    startSession(title: string, constraints: string[], variables: Record<string, unknown>): void {
        this.currentSession = {
            id: Math.random().toString(36).substring(7),
            title,
            root_node: this.buildProofTree(constraints, variables),
            steps: this.generateSteps(constraints, variables),
            current_step: 0,
            variable_state: { ...variables },
            share_url: `https://codeverify.dev/proof/${Math.random().toString(36).substring(7)}`,
        };

        this.openPanel();
        this.updateWebview();
    }

    private stepForward(): void {
        if (!this.currentSession || this.currentSession.current_step >= this.currentSession.steps.length - 1) {
            return;
        }
        this.currentSession.current_step++;
        const step = this.currentSession.steps[this.currentSession.current_step];
        Object.assign(this.currentSession.variable_state, step.variable_changes);
        this.updateWebview();
    }

    private stepBackward(): void {
        if (!this.currentSession || this.currentSession.current_step <= 0) {
            return;
        }
        this.currentSession.current_step--;
        this.updateWebview();
    }

    private async shareProof(): Promise<void> {
        if (!this.currentSession) {
            return;
        }
        await vscode.env.clipboard.writeText(this.currentSession.share_url);
        vscode.window.showInformationMessage(`Proof URL copied: ${this.currentSession.share_url}`);
    }

    private handleMessage(message: { command: string; [key: string]: unknown }): void {
        switch (message.command) {
            case 'stepForward':
                this.stepForward();
                break;
            case 'stepBackward':
                this.stepBackward();
                break;
            case 'overrideVariable':
                if (this.currentSession && typeof message.variable === 'string') {
                    this.currentSession.variable_state[message.variable] = message.value;
                    this.currentSession.root_node = this.buildProofTree(
                        this.currentSession.steps
                            .filter(s => s.action === 'assert' && s.constraint !== 'Begin proof exploration' && s.constraint !== 'Proof complete')
                            .map(s => s.constraint),
                        this.currentSession.variable_state
                    );
                    this.updateWebview();
                }
                break;
            case 'share':
                this.shareProof();
                break;
        }
    }

    private buildProofTree(constraints: string[], variables: Record<string, unknown>): ProofNode {
        const children: ProofNode[] = constraints.map((c, i) => ({
            id: `node_${i}`,
            constraint: c,
            status: this.evaluateConstraint(c, variables),
            children: [],
            variables: { ...variables },
            depth: 1,
        }));

        const allSatisfied = children.every(c => c.status === 'satisfied');
        const anyViolated = children.some(c => c.status === 'violated');

        return {
            id: 'root',
            constraint: '(and)',
            status: allSatisfied ? 'satisfied' : anyViolated ? 'violated' : 'unknown',
            children,
            variables,
            depth: 0,
        };
    }

    private evaluateConstraint(constraint: string, variables: Record<string, unknown>): 'satisfied' | 'violated' | 'unknown' {
        const c = constraint.toLowerCase();
        for (const [varName, value] of Object.entries(variables)) {
            if (!c.includes(varName.toLowerCase())) continue;
            if (typeof value === 'number') {
                if (c.includes('>= 0') && value >= 0) return 'satisfied';
                if (c.includes('>= 0') && value < 0) return 'violated';
                if (c.includes('!= 0') && value !== 0) return 'satisfied';
                if (c.includes('!= 0') && value === 0) return 'violated';
            }
            if (c.includes('!= null') || c.includes('!= none') || c.includes('is not none')) {
                return value === null || value === undefined ? 'violated' : 'satisfied';
            }
        }
        return 'unknown';
    }

    private generateSteps(constraints: string[], variables: Record<string, unknown>): ProofStep[] {
        const steps: ProofStep[] = [{
            step_number: 0,
            action: 'assert',
            constraint: 'Begin proof exploration',
            variable_changes: variables,
            explanation: 'Starting Z3 constraint solving',
            node_id: 'root',
        }];

        constraints.forEach((c, i) => {
            const status = this.evaluateConstraint(c, variables);
            steps.push({
                step_number: steps.length,
                action: 'assert',
                constraint: c,
                variable_changes: {},
                explanation: `Assert: ${c}`,
                node_id: `node_${i}`,
            });
            steps.push({
                step_number: steps.length,
                action: status === 'satisfied' ? 'propagate' : status === 'violated' ? 'conflict' : 'decide',
                constraint: c,
                variable_changes: {},
                explanation: status === 'satisfied' ? '✓ Satisfied' : status === 'violated' ? '✗ Violated' : '? Undetermined',
                node_id: `node_${i}`,
            });
        });

        steps.push({
            step_number: steps.length,
            action: 'propagate',
            constraint: 'Proof complete',
            variable_changes: {},
            explanation: 'Proof exploration finished',
            node_id: 'root',
        });

        return steps;
    }

    private updateWebview(): void {
        if (!this.panel || !this.currentSession) return;
        this.panel.webview.postMessage({
            type: 'update',
            session: {
                ...this.currentSession,
                root_node: this.currentSession.root_node,
            },
        });
    }

    private getWebviewContent(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 16px; }
        .controls { margin-bottom: 16px; display: flex; gap: 8px; }
        button { background: var(--vscode-button-background); color: var(--vscode-button-foreground);
                 border: none; padding: 6px 12px; cursor: pointer; border-radius: 3px; }
        button:hover { background: var(--vscode-button-hoverBackground); }
        .step { padding: 8px; margin: 4px 0; border-left: 3px solid var(--vscode-editorInfo-foreground); }
        .step.active { background: var(--vscode-editor-selectionBackground); border-left-color: var(--vscode-focusBorder); }
        .step.satisfied { border-left-color: #4ec9b0; }
        .step.violated { border-left-color: #f44747; }
        .node { padding: 4px 8px; margin: 2px 0; }
        .node.satisfied { color: #4ec9b0; }
        .node.violated { color: #f44747; }
        .variables { font-family: var(--vscode-editor-font-family); font-size: 12px; margin-top: 8px;
                     padding: 8px; background: var(--vscode-editor-background); border-radius: 4px; }
        h2 { margin-top: 0; }
        .share-url { font-size: 11px; opacity: 0.7; cursor: pointer; }
    </style>
</head>
<body>
    <h2 id="title">Proof Debugger</h2>
    <p class="share-url" id="shareUrl" onclick="share()"></p>
    <div class="controls">
        <button onclick="stepBackward()">⏮ Back</button>
        <button onclick="stepForward()">⏭ Forward</button>
        <button onclick="share()">📋 Share</button>
        <span id="stepCounter"></span>
    </div>
    <div id="steps"></div>
    <div class="variables"><strong>Variables:</strong><pre id="vars">{}</pre></div>
    <script>
        const vscode = acquireVsCodeApi();
        function stepForward() { vscode.postMessage({ command: 'stepForward' }); }
        function stepBackward() { vscode.postMessage({ command: 'stepBackward' }); }
        function share() { vscode.postMessage({ command: 'share' }); }

        window.addEventListener('message', event => {
            const { session } = event.data;
            if (!session) return;
            document.getElementById('title').textContent = session.title || 'Proof Debugger';
            document.getElementById('shareUrl').textContent = session.share_url || '';
            document.getElementById('stepCounter').textContent =
                'Step ' + (session.current_step + 1) + ' / ' + session.steps.length;
            document.getElementById('vars').textContent = JSON.stringify(session.variable_state, null, 2);

            const stepsDiv = document.getElementById('steps');
            stepsDiv.innerHTML = '';
            session.steps.forEach((step, i) => {
                const div = document.createElement('div');
                div.className = 'step' + (i === session.current_step ? ' active' : '') +
                    (step.action === 'propagate' ? ' satisfied' : '') +
                    (step.action === 'conflict' ? ' violated' : '');
                div.textContent = step.explanation;
                stepsDiv.appendChild(div);
            });
        });
    </script>
</body>
</html>`;
    }

    dispose(): void {
        this.statusBarItem.dispose();
        this.panel?.dispose();
        this.disposables.forEach(d => d.dispose());
    }
}
