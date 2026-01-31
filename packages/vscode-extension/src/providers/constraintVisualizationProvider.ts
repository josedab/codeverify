/**
 * Constraint Visualization Provider
 * 
 * Provides real-time visualization of Z3 constraints and verification status.
 */

import * as vscode from 'vscode';
import { ContinuousVerificationProvider } from './continuousVerificationProvider';
import { logger } from '../logger';

interface ConstraintNode {
    id: string;
    label: string;
    type: 'variable' | 'constraint' | 'assertion' | 'result';
    children: ConstraintNode[];
    status: 'satisfied' | 'unsatisfied' | 'unknown';
}

export class ConstraintVisualizationProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'codeverify.constraintVisualization';
    
    private _view?: vscode.WebviewView;
    private continuousProvider: ContinuousVerificationProvider;

    constructor(
        private readonly extensionUri: vscode.Uri,
        continuousProvider: ContinuousVerificationProvider
    ) {
        this.continuousProvider = continuousProvider;
        
        // Listen for verification completion
        continuousProvider.onVerificationComplete(({ uri }) => {
            this.refresh();
        });
    }

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        token: vscode.CancellationToken
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.extensionUri],
        };

        webviewView.webview.html = this.getHtmlContent();

        webviewView.webview.onDidReceiveMessage(
            message => {
                switch (message.type) {
                    case 'refresh':
                        this.refresh();
                        break;
                    case 'selectNode':
                        this.handleNodeSelection(message.nodeId);
                        break;
                }
            }
        );

        // Initial refresh
        this.refresh();
    }

    /**
     * Refresh the visualization
     */
    async refresh(): Promise<void> {
        if (!this._view) return;

        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            this._view.webview.postMessage({
                type: 'update',
                data: { constraints: [], variables: [], satisfiable: null },
            });
            return;
        }

        const position = editor.selection.active;
        const visualization = this.continuousProvider.getConstraintVisualization(
            editor.document,
            position
        );

        if (visualization) {
            const tree = this.buildConstraintTree(visualization);
            this._view.webview.postMessage({
                type: 'update',
                data: {
                    ...visualization,
                    tree,
                },
            });
        } else {
            this._view.webview.postMessage({
                type: 'update',
                data: { constraints: [], variables: [], satisfiable: null },
            });
        }
    }

    /**
     * Build constraint tree from flat constraints
     */
    private buildConstraintTree(visualization: {
        constraints: string[];
        variables: string[];
        satisfiable: boolean;
    }): ConstraintNode {
        const root: ConstraintNode = {
            id: 'root',
            label: visualization.satisfiable ? 'Satisfiable ✓' : 'Unsatisfiable ✗',
            type: 'result',
            status: visualization.satisfiable ? 'satisfied' : 'unsatisfied',
            children: [],
        };

        // Add variables node
        if (visualization.variables.length > 0) {
            const varsNode: ConstraintNode = {
                id: 'variables',
                label: 'Variables',
                type: 'variable',
                status: 'unknown',
                children: visualization.variables.map((v, i) => ({
                    id: `var-${i}`,
                    label: v,
                    type: 'variable' as const,
                    status: 'unknown' as const,
                    children: [],
                })),
            };
            root.children.push(varsNode);
        }

        // Add constraints node
        if (visualization.constraints.length > 0) {
            const constraintsNode: ConstraintNode = {
                id: 'constraints',
                label: 'Constraints',
                type: 'constraint',
                status: visualization.satisfiable ? 'satisfied' : 'unsatisfied',
                children: visualization.constraints.map((c, i) => ({
                    id: `constraint-${i}`,
                    label: this.formatConstraint(c),
                    type: 'constraint' as const,
                    status: 'unknown' as const,
                    children: [],
                })),
            };
            root.children.push(constraintsNode);
        }

        return root;
    }

    /**
     * Format constraint for display
     */
    private formatConstraint(constraint: string): string {
        // Shorten long constraints
        if (constraint.length > 50) {
            return constraint.substring(0, 47) + '...';
        }
        return constraint;
    }

    /**
     * Handle node selection in tree
     */
    private handleNodeSelection(nodeId: string): void {
        // Could navigate to related code or show details
        logger.debug('Constraint node selected', { nodeId });
    }

    /**
     * Get HTML content for webview
     */
    private getHtmlContent(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <style>
        :root {
            --bg-color: var(--vscode-editor-background);
            --text-color: var(--vscode-foreground);
            --border-color: var(--vscode-widget-border);
            --success-color: #2ecc71;
            --error-color: #e74c3c;
            --warning-color: #f39c12;
        }
        
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--text-color);
            padding: 10px;
            margin: 0;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .header h3 {
            margin: 0;
            font-size: 14px;
        }
        
        .refresh-btn {
            background: none;
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 4px 8px;
            cursor: pointer;
            border-radius: 4px;
        }
        
        .refresh-btn:hover {
            background: var(--vscode-button-secondaryHoverBackground);
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 15px;
            padding: 8px;
            border-radius: 4px;
            background: var(--vscode-editor-inactiveSelectionBackground);
        }
        
        .status.satisfied {
            border-left: 3px solid var(--success-color);
        }
        
        .status.unsatisfied {
            border-left: 3px solid var(--error-color);
        }
        
        .status.unknown {
            border-left: 3px solid var(--warning-color);
        }
        
        .status-icon {
            font-size: 16px;
        }
        
        .section {
            margin-bottom: 15px;
        }
        
        .section-title {
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            color: var(--vscode-descriptionForeground);
            margin-bottom: 8px;
        }
        
        .tree {
            list-style: none;
            padding-left: 0;
            margin: 0;
        }
        
        .tree-item {
            padding: 4px 0;
        }
        
        .tree-item-content {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .tree-item-content:hover {
            background: var(--vscode-list-hoverBackground);
        }
        
        .tree-item-icon {
            font-size: 12px;
            width: 16px;
            text-align: center;
        }
        
        .tree-item-label {
            font-family: var(--vscode-editor-font-family);
            font-size: 12px;
        }
        
        .tree-children {
            list-style: none;
            padding-left: 20px;
            margin: 0;
        }
        
        .variable-tag {
            display: inline-block;
            padding: 2px 6px;
            background: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            border-radius: 3px;
            font-size: 11px;
            margin: 2px;
        }
        
        .constraint-item {
            font-family: var(--vscode-editor-font-family);
            font-size: 12px;
            padding: 6px 8px;
            background: var(--vscode-textCodeBlock-background);
            border-radius: 4px;
            margin: 4px 0;
            word-break: break-all;
        }
        
        .empty-state {
            text-align: center;
            padding: 30px;
            color: var(--vscode-descriptionForeground);
        }
        
        .empty-state-icon {
            font-size: 40px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h3>Constraint Visualization</h3>
        <button class="refresh-btn" onclick="refresh()">↻ Refresh</button>
    </div>
    
    <div id="content">
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <p>No constraints to display</p>
            <p style="font-size: 11px;">Place cursor in a verifiable code block</p>
        </div>
    </div>
    
    <script>
        const vscode = acquireVsCodeApi();
        
        function refresh() {
            vscode.postMessage({ type: 'refresh' });
        }
        
        function selectNode(nodeId) {
            vscode.postMessage({ type: 'selectNode', nodeId });
        }
        
        function renderTree(node, depth = 0) {
            if (!node) return '';
            
            const icon = getIcon(node.type, node.status);
            const children = node.children?.length 
                ? '<ul class="tree-children">' + node.children.map(c => renderTree(c, depth + 1)).join('') + '</ul>'
                : '';
            
            return \`
                <li class="tree-item">
                    <div class="tree-item-content" onclick="selectNode('\${node.id}')">
                        <span class="tree-item-icon">\${icon}</span>
                        <span class="tree-item-label">\${node.label}</span>
                    </div>
                    \${children}
                </li>
            \`;
        }
        
        function getIcon(type, status) {
            switch (type) {
                case 'result':
                    return status === 'satisfied' ? '✅' : '❌';
                case 'variable':
                    return '📌';
                case 'constraint':
                    return status === 'satisfied' ? '✓' : status === 'unsatisfied' ? '✗' : '○';
                case 'assertion':
                    return '⚡';
                default:
                    return '•';
            }
        }
        
        window.addEventListener('message', event => {
            const message = event.data;
            
            if (message.type === 'update') {
                const data = message.data;
                const content = document.getElementById('content');
                
                if (!data.constraints.length && !data.variables.length) {
                    content.innerHTML = \`
                        <div class="empty-state">
                            <div class="empty-state-icon">📋</div>
                            <p>No constraints to display</p>
                            <p style="font-size: 11px;">Place cursor in a verifiable code block</p>
                        </div>
                    \`;
                    return;
                }
                
                const statusClass = data.satisfiable === null ? 'unknown' : 
                    data.satisfiable ? 'satisfied' : 'unsatisfied';
                const statusText = data.satisfiable === null ? 'Unknown' :
                    data.satisfiable ? 'Satisfiable' : 'Unsatisfiable';
                const statusIcon = data.satisfiable === null ? '❓' :
                    data.satisfiable ? '✅' : '❌';
                
                let html = \`
                    <div class="status \${statusClass}">
                        <span class="status-icon">\${statusIcon}</span>
                        <span>Verification Status: <strong>\${statusText}</strong></span>
                    </div>
                \`;
                
                if (data.variables.length) {
                    html += \`
                        <div class="section">
                            <div class="section-title">Variables (\${data.variables.length})</div>
                            <div>
                                \${data.variables.map(v => '<span class="variable-tag">' + v + '</span>').join('')}
                            </div>
                        </div>
                    \`;
                }
                
                if (data.constraints.length) {
                    html += \`
                        <div class="section">
                            <div class="section-title">Constraints (\${data.constraints.length})</div>
                            <div>
                                \${data.constraints.map(c => '<div class="constraint-item">' + c + '</div>').join('')}
                            </div>
                        </div>
                    \`;
                }
                
                if (data.tree) {
                    html += \`
                        <div class="section">
                            <div class="section-title">Constraint Tree</div>
                            <ul class="tree">
                                \${renderTree(data.tree)}
                            </ul>
                        </div>
                    \`;
                }
                
                content.innerHTML = html;
            }
        });
    </script>
</body>
</html>`;
    }
}

/**
 * Heat Map Webview Provider
 * 
 * Provides a visualization of verification heat map across the file.
 */
export class HeatMapProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'codeverify.heatMap';
    
    private _view?: vscode.WebviewView;
    private continuousProvider: ContinuousVerificationProvider;

    constructor(
        private readonly extensionUri: vscode.Uri,
        continuousProvider: ContinuousVerificationProvider
    ) {
        this.continuousProvider = continuousProvider;
        
        continuousProvider.onVerificationComplete(() => {
            this.refresh();
        });
    }

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        token: vscode.CancellationToken
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.extensionUri],
        };

        webviewView.webview.html = this.getHtmlContent();

        webviewView.webview.onDidReceiveMessage(message => {
            switch (message.type) {
                case 'toggleHeatMap':
                    const editor = vscode.window.activeTextEditor;
                    if (editor) {
                        if (message.enabled) {
                            this.continuousProvider.showHeatMap(editor);
                        } else {
                            this.continuousProvider.hideHeatMap(editor);
                        }
                    }
                    break;
                case 'goToLine':
                    this.goToLine(message.line);
                    break;
            }
        });

        this.refresh();
    }

    private async refresh(): Promise<void> {
        if (!this._view) return;

        const stats = this.continuousProvider.getStatistics();
        
        this._view.webview.postMessage({
            type: 'update',
            data: stats,
        });
    }

    private goToLine(line: number): void {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const position = new vscode.Position(line, 0);
            editor.selection = new vscode.Selection(position, position);
            editor.revealRange(new vscode.Range(position, position));
        }
    }

    private getHtmlContent(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            padding: 10px;
            margin: 0;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .header h3 {
            margin: 0;
            font-size: 14px;
        }
        
        .toggle {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .toggle input {
            cursor: pointer;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .stat {
            padding: 10px;
            border-radius: 6px;
            background: var(--vscode-editor-inactiveSelectionBackground);
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
        }
        
        .stat-label {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
        }
        
        .stat.verified .stat-value { color: #2ecc71; }
        .stat.pending .stat-value { color: #f39c12; }
        .stat.failed .stat-value { color: #e74c3c; }
        
        .legend {
            display: flex;
            gap: 5px;
            margin-top: 15px;
        }
        
        .legend-item {
            flex: 1;
            height: 20px;
            border-radius: 2px;
        }
        
        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: var(--vscode-descriptionForeground);
        }
    </style>
</head>
<body>
    <div class="header">
        <h3>Verification Heat Map</h3>
        <div class="toggle">
            <input type="checkbox" id="heatMapToggle" onchange="toggleHeatMap(this.checked)">
            <label for="heatMapToggle">Show</label>
        </div>
    </div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value" id="totalUnits">0</div>
            <div class="stat-label">Total Units</div>
        </div>
        <div class="stat verified">
            <div class="stat-value" id="verifiedCount">0</div>
            <div class="stat-label">Verified</div>
        </div>
        <div class="stat pending">
            <div class="stat-value" id="pendingCount">0</div>
            <div class="stat-label">Pending</div>
        </div>
        <div class="stat failed">
            <div class="stat-value" id="failedCount">0</div>
            <div class="stat-label">Failed</div>
        </div>
    </div>
    
    <div style="margin-top: 20px;">
        <div class="legend">
            <div class="legend-item" style="background: rgba(0, 255, 0, 0.3);"></div>
            <div class="legend-item" style="background: rgba(51, 204, 0, 0.3);"></div>
            <div class="legend-item" style="background: rgba(102, 153, 0, 0.3);"></div>
            <div class="legend-item" style="background: rgba(153, 153, 0, 0.3);"></div>
            <div class="legend-item" style="background: rgba(204, 102, 0, 0.3);"></div>
            <div class="legend-item" style="background: rgba(255, 0, 0, 0.3);"></div>
        </div>
        <div class="legend-labels">
            <span>Low Risk</span>
            <span>High Risk</span>
        </div>
    </div>
    
    <script>
        const vscode = acquireVsCodeApi();
        
        function toggleHeatMap(enabled) {
            vscode.postMessage({ type: 'toggleHeatMap', enabled });
        }
        
        window.addEventListener('message', event => {
            const message = event.data;
            
            if (message.type === 'update') {
                const data = message.data;
                document.getElementById('totalUnits').textContent = data.totalUnits;
                document.getElementById('verifiedCount').textContent = data.verifiedCount;
                document.getElementById('pendingCount').textContent = data.pendingCount;
                document.getElementById('failedCount').textContent = data.failedCount;
            }
        });
    </script>
</body>
</html>`;
    }
}
