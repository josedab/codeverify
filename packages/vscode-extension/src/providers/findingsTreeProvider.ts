/**
 * Findings Tree Provider
 * 
 * Provides tree view of CodeVerify findings in the sidebar.
 */

import * as vscode from 'vscode';
import { Finding } from '../client';
import * as path from 'path';

export class FindingsTreeProvider implements vscode.TreeDataProvider<FindingTreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<FindingTreeItem | undefined | null | void> = 
        new vscode.EventEmitter<FindingTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<FindingTreeItem | undefined | null | void> = 
        this._onDidChangeTreeData.event;

    private findings: Map<string, Finding[]> = new Map();

    /**
     * Update findings for a specific file
     */
    updateFindings(filePath: string, findings: Finding[]): void {
        this.findings.set(filePath, findings);
        this._onDidChangeTreeData.fire();
    }

    /**
     * Update all findings from a workspace scan
     */
    updateAllFindings(findings: Finding[]): void {
        this.findings.clear();
        
        for (const finding of findings) {
            const filePath = finding.file_path;
            if (!this.findings.has(filePath)) {
                this.findings.set(filePath, []);
            }
            this.findings.get(filePath)!.push(finding);
        }
        
        this._onDidChangeTreeData.fire();
    }

    /**
     * Remove a specific finding
     */
    removeFinding(finding: Finding): void {
        const fileFindings = this.findings.get(finding.file_path);
        if (fileFindings) {
            const index = fileFindings.findIndex(f => 
                f.line_start === finding.line_start && f.title === finding.title
            );
            if (index !== -1) {
                fileFindings.splice(index, 1);
                if (fileFindings.length === 0) {
                    this.findings.delete(finding.file_path);
                }
                this._onDidChangeTreeData.fire();
            }
        }
    }

    /**
     * Clear all findings
     */
    clear(): void {
        this.findings.clear();
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: FindingTreeItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: FindingTreeItem): Thenable<FindingTreeItem[]> {
        if (!element) {
            // Root level - show files with findings
            const items: FindingTreeItem[] = [];
            
            for (const [filePath, findings] of this.findings) {
                if (findings.length > 0) {
                    const counts = this.countBySeverity(findings);
                    items.push(new FindingTreeItem(
                        path.basename(filePath),
                        vscode.TreeItemCollapsibleState.Expanded,
                        {
                            type: 'file',
                            filePath,
                            counts,
                        }
                    ));
                }
            }
            
            // Sort by severity (files with critical/high first)
            items.sort((a, b) => {
                const aScore = (a.data.counts?.critical || 0) * 1000 + (a.data.counts?.high || 0) * 100;
                const bScore = (b.data.counts?.critical || 0) * 1000 + (b.data.counts?.high || 0) * 100;
                return bScore - aScore;
            });
            
            return Promise.resolve(items);
        } else if (element.data.type === 'file') {
            // File level - show findings in this file
            const findings = this.findings.get(element.data.filePath) || [];
            
            // Sort by severity and line number
            const sorted = [...findings].sort((a, b) => {
                const severityOrder: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };
                const aSev = severityOrder[a.severity] ?? 5;
                const bSev = severityOrder[b.severity] ?? 5;
                if (aSev !== bSev) return aSev - bSev;
                return (a.line_start || 0) - (b.line_start || 0);
            });
            
            return Promise.resolve(
                sorted.map(finding => new FindingTreeItem(
                    finding.title,
                    vscode.TreeItemCollapsibleState.None,
                    {
                        type: 'finding',
                        finding,
                        filePath: element.data.filePath,
                    }
                ))
            );
        }
        
        return Promise.resolve([]);
    }

    private countBySeverity(findings: Finding[]): Record<string, number> {
        const counts: Record<string, number> = {};
        for (const finding of findings) {
            counts[finding.severity] = (counts[finding.severity] || 0) + 1;
        }
        return counts;
    }
}

class FindingTreeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly data: {
            type: 'file' | 'finding';
            filePath?: string;
            finding?: Finding;
            counts?: Record<string, number>;
        }
    ) {
        super(label, collapsibleState);
        
        if (data.type === 'file') {
            this.contextValue = 'file';
            this.iconPath = new vscode.ThemeIcon('file-code');
            this.description = this.formatCounts(data.counts || {});
            this.resourceUri = vscode.Uri.file(data.filePath!);
        } else if (data.type === 'finding' && data.finding) {
            const finding = data.finding;
            this.contextValue = 'finding';
            this.iconPath = this.getSeverityIcon(finding.severity);
            this.description = `Line ${finding.line_start}`;
            this.tooltip = new vscode.MarkdownString(
                `**${finding.title}**\n\n${finding.description}\n\n` +
                `*Severity: ${finding.severity}* | *Confidence: ${Math.round(finding.confidence * 100)}%*`
            );
            
            // Click to go to the finding location
            this.command = {
                command: 'vscode.open',
                title: 'Go to Finding',
                arguments: [
                    vscode.Uri.file(data.filePath!),
                    {
                        selection: new vscode.Range(
                            (finding.line_start || 1) - 1, 0,
                            (finding.line_start || 1) - 1, 0
                        )
                    }
                ]
            };
        }
    }

    private formatCounts(counts: Record<string, number>): string {
        const parts: string[] = [];
        if (counts.critical) parts.push(`🔴${counts.critical}`);
        if (counts.high) parts.push(`🟠${counts.high}`);
        if (counts.medium) parts.push(`🟡${counts.medium}`);
        if (counts.low) parts.push(`🔵${counts.low}`);
        return parts.join(' ');
    }

    private getSeverityIcon(severity: string): vscode.ThemeIcon {
        switch (severity.toLowerCase()) {
            case 'critical':
                return new vscode.ThemeIcon('error', new vscode.ThemeColor('errorForeground'));
            case 'high':
                return new vscode.ThemeIcon('warning', new vscode.ThemeColor('errorForeground'));
            case 'medium':
                return new vscode.ThemeIcon('warning', new vscode.ThemeColor('editorWarning.foreground'));
            case 'low':
                return new vscode.ThemeIcon('info', new vscode.ThemeColor('editorInfo.foreground'));
            default:
                return new vscode.ThemeIcon('circle-outline');
        }
    }
}
