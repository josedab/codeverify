/**
 * CodeVerify VS Code Extension
 * 
 * Main extension entry point with real-time verification and trust score features.
 */

import * as vscode from 'vscode';
import { AnalysisProvider } from './providers/analysisProvider';
import { FindingsTreeProvider } from './providers/findingsTreeProvider';
import { DiagnosticsProvider } from './providers/diagnosticsProvider';
import { CodeActionProvider } from './providers/codeActionProvider';
import {
    ContinuousVerificationProvider,
    VerificationMode
} from './providers/continuousVerificationProvider';
import {
    ConstraintVisualizationProvider,
    HeatMapProvider
} from './providers/constraintVisualizationProvider';
import { PairReviewerProvider } from './providers/pairReviewerProvider';
import { PasteInterceptionProvider } from './providers/pasteInterceptionProvider';
import { FormalSpecAssistantProvider } from './providers/formalSpecAssistantProvider';
import { StreamingVerificationProvider } from './providers/streamingVerificationProvider';
import { HallucinationProvider } from './providers/hallucinationProvider';
import { CodeVerifyClient } from './client';
import { logger, initializeLogger } from './logger';

let client: CodeVerifyClient;
let diagnosticsProvider: DiagnosticsProvider;
let findingsTreeProvider: FindingsTreeProvider;
let continuousVerificationProvider: ContinuousVerificationProvider;
let pairReviewerProvider: PairReviewerProvider;
let pasteInterceptionProvider: PasteInterceptionProvider;
let formalSpecAssistantProvider: FormalSpecAssistantProvider;
let streamingVerificationProvider: StreamingVerificationProvider;
let hallucinationProvider: HallucinationProvider;
let statusBarItem: vscode.StatusBarItem;
let realTimeEnabled = false;
let realTimeTimeout: NodeJS.Timeout | undefined;

// Decoration types for verified/unverified code
let verifiedDecorationType: vscode.TextEditorDecorationType;
let unverifiedDecorationType: vscode.TextEditorDecorationType;
let errorDecorationType: vscode.TextEditorDecorationType;

export function activate(context: vscode.ExtensionContext) {
    // Initialize logger first
    const outputChannel = initializeLogger();
    context.subscriptions.push(outputChannel);

    logger.info('CodeVerify extension is now active');

    // Initialize client
    const config = vscode.workspace.getConfiguration('codeverify');
    client = new CodeVerifyClient({
        apiEndpoint: config.get('apiEndpoint', 'https://api.codeverify.io'),
        apiKey: config.get('apiKey', ''),
        cliPath: config.get('cliPath', 'codeverify'),
        localAnalysisEnabled: config.get('localAnalysisEnabled', true),
    });

    // Initialize providers
    diagnosticsProvider = new DiagnosticsProvider(client);
    findingsTreeProvider = new FindingsTreeProvider();
    continuousVerificationProvider = new ContinuousVerificationProvider(client);
    pairReviewerProvider = new PairReviewerProvider(client);
    pasteInterceptionProvider = new PasteInterceptionProvider(client);
    formalSpecAssistantProvider = new FormalSpecAssistantProvider(client);

    // Initialize next-gen providers (v0.4.0)
    const streamingEnabled = config.get('streamingVerification.enabled', true);
    if (streamingEnabled) {
        const debounceMs = config.get('streamingVerification.debounceMs', 500);
        streamingVerificationProvider = new StreamingVerificationProvider(debounceMs as number);
        context.subscriptions.push(streamingVerificationProvider);
        logger.info('Streaming verification provider enabled');
    }

    const hallucinationEnabled = config.get('hallucinationDetection.enabled', true);
    if (hallucinationEnabled) {
        hallucinationProvider = new HallucinationProvider();
        context.subscriptions.push(hallucinationProvider);
        logger.info('Hallucination detection provider enabled');
    }

    // Register tree view
    vscode.window.registerTreeDataProvider('codeverifyFindings', findingsTreeProvider);
    
    // Register webview providers for continuous verification
    const constraintVisualizationProvider = new ConstraintVisualizationProvider(
        context.extensionUri,
        continuousVerificationProvider
    );
    const heatMapProvider = new HeatMapProvider(
        context.extensionUri,
        continuousVerificationProvider
    );
    
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            ConstraintVisualizationProvider.viewType,
            constraintVisualizationProvider
        ),
        vscode.window.registerWebviewViewProvider(
            HeatMapProvider.viewType,
            heatMapProvider
        )
    );
    
    // Listen for continuous verification events
    continuousVerificationProvider.onVerificationComplete(({ uri, units }) => {
        // Update diagnostics from continuous verification
        const allFindings = units.flatMap(u => u.findings);
        diagnosticsProvider.updateDiagnostics(uri, allFindings);
        findingsTreeProvider.updateFindings(uri.fsPath, allFindings);
    });

    // Register code actions
    const codeActionProvider = new CodeActionProvider(client);
    context.subscriptions.push(
        vscode.languages.registerCodeActionsProvider(
            [
                { scheme: 'file', language: 'python' },
                { scheme: 'file', language: 'typescript' },
                { scheme: 'file', language: 'javascript' },
                { scheme: 'file', language: 'go' },
                { scheme: 'file', language: 'java' },
                { scheme: 'file', language: 'rust' },
            ],
            codeActionProvider,
            { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] }
        )
    );

    // Create status bar item for trust score
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'codeverify.showTrustScore';
    context.subscriptions.push(statusBarItem);
    
    // Initialize decoration types
    initializeDecorations();

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('codeverify.analyze', () => analyzeCurrentFile()),
        vscode.commands.registerCommand('codeverify.analyzeWorkspace', () => analyzeWorkspace()),
        vscode.commands.registerCommand('codeverify.analyzeSelection', () => analyzeSelection()),
        vscode.commands.registerCommand('codeverify.showFindings', () => showFindings()),
        vscode.commands.registerCommand('codeverify.applyFix', (finding) => applyFix(finding)),
        vscode.commands.registerCommand('codeverify.dismissFinding', (finding) => dismissFinding(finding)),
        vscode.commands.registerCommand('codeverify.openDashboard', () => openDashboard()),
        vscode.commands.registerCommand('codeverify.refreshFindings', () => refreshFindings()),
        vscode.commands.registerCommand('codeverify.showTrustScore', () => showTrustScore()),
        vscode.commands.registerCommand('codeverify.explainFinding', () => explainFinding()),
        vscode.commands.registerCommand('codeverify.debugVerification', () => debugVerification()),
        vscode.commands.registerCommand('codeverify.toggleRealTime', () => toggleRealTimeVerification()),
        // Continuous verification commands
        vscode.commands.registerCommand('codeverify.toggleContinuousMode', () => toggleContinuousMode()),
        vscode.commands.registerCommand('codeverify.setContinuousMode', (mode: string) => setContinuousMode(mode)),
        vscode.commands.registerCommand('codeverify.showHeatMap', () => showHeatMap()),
        vscode.commands.registerCommand('codeverify.hideHeatMap', () => hideHeatMap()),
        vscode.commands.registerCommand('codeverify.showConstraints', () => showConstraints()),
        // AI Pair Reviewer commands
        vscode.commands.registerCommand('codeverify.togglePairReviewer', () => togglePairReviewer()),
        vscode.commands.registerCommand('codeverify.reviewUnit', (unit) => reviewUnit(unit)),
        vscode.commands.registerCommand('codeverify.applyFixes', (unit, findings) => applyFixes(unit, findings)),
        vscode.commands.registerCommand('codeverify.dismissPairReviewFinding', (finding, reason) => dismissPairReviewFinding(finding, reason)),
        vscode.commands.registerCommand('codeverify.showPairReviewStats', () => showPairReviewStats()),
        // Paste Interception commands
        vscode.commands.registerCommand('codeverify.togglePasteInterception', () => togglePasteInterception()),
        vscode.commands.registerCommand('codeverify.interceptPaste', () => pasteInterceptionProvider.handlePaste()),
        vscode.commands.registerCommand('codeverify.showPasteInterceptionStats', () => showPasteInterceptionStats()),
        // Formal Spec Assistant commands
        vscode.commands.registerCommand('codeverify.convertNLToZ3', () => formalSpecAssistantProvider.showConversionInput()),
        vscode.commands.registerCommand('codeverify.showSpecTemplates', () => formalSpecAssistantProvider.showTemplateLibrary()),
        vscode.commands.registerCommand('codeverify.suggestSpecs', () => formalSpecAssistantProvider.suggestSpecsForCurrentFunction()),
        vscode.commands.registerCommand('codeverify.showSpecHistory', () => formalSpecAssistantProvider.showHistory()),
    );

    // Auto-analyze on save
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument((document) => {
            if (config.get('analyzeOnSave', true)) {
                analyzeDocument(document);
            }
        })
    );

    // Real-time verification on text change
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument((event) => {
            if (config.get('realTimeVerification', false) && realTimeEnabled) {
                handleRealTimeVerification(event.document);
            }
            // Also handle continuous verification
            if (config.get('continuousVerification', false)) {
                continuousVerificationProvider.handleChange(event);
            }
            // Handle pair reviewer
            if (config.get('pairReviewer', false)) {
                pairReviewerProvider.handleChange(event);
            }
        })
    );

    // Update status bar on editor change
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor && config.get('showTrustScore', true)) {
                updateTrustScoreStatusBar(editor.document);
            }
        })
    );

    // Initialize real-time verification state
    realTimeEnabled = config.get('realTimeVerification', false);

    // Analyze open documents on activation
    if (config.get('enabled', true)) {
        vscode.workspace.textDocuments.forEach(doc => analyzeDocument(doc));
    }

    // Initialize continuous verification if enabled
    if (config.get('continuousVerification', false)) {
        continuousVerificationProvider.enable();
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            continuousVerificationProvider.initializeDocument(editor.document);
        }
    }

    // Initialize pair reviewer if enabled
    if (config.get('pairReviewer', false)) {
        pairReviewerProvider.enable();
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            pairReviewerProvider.initializeDocument(editor.document);
        }
    }

    // Initialize paste interception if enabled
    if (config.get('pasteInterception.enabled', true)) {
        pasteInterceptionProvider.enable();

        // Listen for paste interception events
        pasteInterceptionProvider.onPasteIntercepted((result) => {
            logger.info('Paste intercepted', {
                isAiGenerated: result.isAiGenerated,
                trustScore: result.trustScore,
                analysisTimeMs: result.analysisTimeMs,
            });
        });

        pasteInterceptionProvider.onPasteDecision(({ result, decision }) => {
            logger.info('Paste decision made', { decision, trustScore: result.trustScore });

            // Update findings if code was accepted with issues
            if ((decision === 'accept' || decision === 'accept_with_review') && result.findings.length > 0) {
                const editor = vscode.window.activeTextEditor;
                if (editor) {
                    diagnosticsProvider.updateDiagnostics(editor.document.uri, result.findings);
                }
            }
        });
    }

    // Show initial trust score
    const editor = vscode.window.activeTextEditor;
    if (editor && config.get('showTrustScore', true)) {
        updateTrustScoreStatusBar(editor.document);
    }
}

export function deactivate() {
    logger.info('CodeVerify extension deactivating');
    diagnosticsProvider?.dispose();
    continuousVerificationProvider?.dispose();
    pairReviewerProvider?.dispose();
    pasteInterceptionProvider?.dispose();
    formalSpecAssistantProvider?.dispose();
    verifiedDecorationType?.dispose();
    unverifiedDecorationType?.dispose();
    errorDecorationType?.dispose();
    logger.dispose();
}

function initializeDecorations() {
    verifiedDecorationType = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('codeverify.verified'),
        isWholeLine: true,
        overviewRulerColor: '#2ecc71',
        overviewRulerLane: vscode.OverviewRulerLane.Right,
    });

    unverifiedDecorationType = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('codeverify.unverified'),
        isWholeLine: true,
        overviewRulerColor: '#f39c12',
        overviewRulerLane: vscode.OverviewRulerLane.Right,
    });

    errorDecorationType = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('codeverify.error'),
        isWholeLine: true,
        overviewRulerColor: '#e74c3c',
        overviewRulerLane: vscode.OverviewRulerLane.Right,
    });
}

function handleRealTimeVerification(document: vscode.TextDocument) {
    if (!isSupportedLanguage(document.languageId)) {
        return;
    }

    // Clear existing timeout
    if (realTimeTimeout) {
        clearTimeout(realTimeTimeout);
    }

    // Set new timeout
    const config = vscode.workspace.getConfiguration('codeverify');
    const delay = config.get('realTimeDelay', 1500);

    realTimeTimeout = setTimeout(async () => {
        await analyzeDocument(document, true);
    }, delay);
}

async function analyzeCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active file to analyze');
        return;
    }

    await analyzeDocument(editor.document);
}

async function analyzeSelection() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) {
        vscode.window.showWarningMessage('No selection to analyze');
        return;
    }

    const selection = editor.selection;
    const selectedText = editor.document.getText(selection);

    try {
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'CodeVerify: Analyzing selection...',
                cancellable: false,
            },
            async () => {
                const findings = await client.analyzeCode(selectedText, editor.document.languageId);
                
                // Adjust line numbers to selection offset
                const adjustedFindings = findings.map((f: any) => ({
                    ...f,
                    line_start: (f.line_start || 1) + selection.start.line,
                    line_end: (f.line_end || f.line_start || 1) + selection.start.line,
                }));

                diagnosticsProvider.updateDiagnostics(editor.document.uri, adjustedFindings);
                findingsTreeProvider.updateFindings(editor.document.uri.fsPath, adjustedFindings);

                if (adjustedFindings.length === 0) {
                    vscode.window.showInformationMessage('CodeVerify: No issues found in selection');
                } else {
                    vscode.window.showWarningMessage(`CodeVerify: Found ${adjustedFindings.length} issue(s) in selection`);
                }
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`CodeVerify analysis failed: ${error}`);
    }
}

async function analyzeDocument(document: vscode.TextDocument, silent = false) {
    if (!isSupportedLanguage(document.languageId)) {
        return;
    }

    const config = vscode.workspace.getConfiguration('codeverify');
    if (!config.get('enabled', true)) {
        return;
    }

    try {
        const progressOptions = silent ? undefined : {
            location: vscode.ProgressLocation.Notification,
            title: 'CodeVerify: Analyzing...',
            cancellable: false,
        };

        const doAnalysis = async () => {
            const findings = await client.analyzeFile(document.uri.fsPath);
            diagnosticsProvider.updateDiagnostics(document.uri, findings);
            findingsTreeProvider.updateFindings(document.uri.fsPath, findings);

            // Update decorations
            if (config.get('showVerificationDecorations', true)) {
                updateDecorations(document, findings);
            }

            // Update trust score
            if (config.get('showTrustScore', true)) {
                updateTrustScoreStatusBar(document);
            }
        };

        if (progressOptions) {
            await vscode.window.withProgress(progressOptions, doAnalysis);
        } else {
            await doAnalysis();
        }
    } catch (error) {
        if (!silent) {
            vscode.window.showErrorMessage(`CodeVerify analysis failed: ${error}`);
        }
    }
}

function updateDecorations(document: vscode.TextDocument, findings: any[]) {
    const editor = vscode.window.visibleTextEditors.find(e => e.document === document);
    if (!editor) {
        return;
    }

    // Find lines with errors
    const errorLines = new Set<number>();
    const warningLines = new Set<number>();

    for (const finding of findings) {
        const startLine = (finding.line_start || 1) - 1;
        const endLine = (finding.line_end || finding.line_start || 1) - 1;
        
        for (let line = startLine; line <= endLine; line++) {
            if (finding.severity === 'critical' || finding.severity === 'high') {
                errorLines.add(line);
            } else {
                warningLines.add(line);
            }
        }
    }

    // Create decoration ranges
    const errorRanges: vscode.Range[] = [];
    const warningRanges: vscode.Range[] = [];
    const verifiedRanges: vscode.Range[] = [];

    for (let i = 0; i < document.lineCount; i++) {
        const range = new vscode.Range(i, 0, i, document.lineAt(i).text.length);
        
        if (errorLines.has(i)) {
            errorRanges.push(range);
        } else if (warningLines.has(i)) {
            warningRanges.push(range);
        } else if (!document.lineAt(i).isEmptyOrWhitespace) {
            verifiedRanges.push(range);
        }
    }

    // Apply decorations
    editor.setDecorations(errorDecorationType, errorRanges);
    editor.setDecorations(unverifiedDecorationType, warningRanges);
    editor.setDecorations(verifiedDecorationType, verifiedRanges);
}

async function updateTrustScoreStatusBar(document: vscode.TextDocument) {
    if (!isSupportedLanguage(document.languageId)) {
        statusBarItem.hide();
        return;
    }

    try {
        const trustScore = await client.getTrustScore(document.uri.fsPath);
        
        const score = trustScore.score || 0;
        const emoji = score >= 80 ? '✅' : score >= 60 ? '⚠️' : '❌';
        
        statusBarItem.text = `${emoji} Trust: ${score}%`;
        statusBarItem.tooltip = `CodeVerify Trust Score: ${score}%\n` +
            `AI Detection: ${trustScore.ai_probability || 0}%\n` +
            `Risk Level: ${trustScore.risk_level || 'unknown'}\n` +
            `Click for details`;
        statusBarItem.show();
    } catch (error) {
        // Silently fail - don't spam errors for trust score
        statusBarItem.text = '$(shield) Trust: --';
        statusBarItem.tooltip = 'Trust score unavailable';
        statusBarItem.show();
    }
}

async function showTrustScore() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active file');
        return;
    }

    try {
        const trustScore = await client.getTrustScore(editor.document.uri.fsPath);
        
        const panel = vscode.window.createWebviewPanel(
            'codeverifyTrustScore',
            'CodeVerify Trust Score',
            vscode.ViewColumn.Beside,
            { enableScripts: true }
        );

        panel.webview.html = getTrustScoreHtml(trustScore);
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to get trust score: ${error}`);
    }
}

function getTrustScoreHtml(trustScore: any): string {
    const score = trustScore.score || 0;
    const color = score >= 80 ? '#2ecc71' : score >= 60 ? '#f1c40f' : '#e74c3c';
    
    return `<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; padding: 20px; }
        .score-circle { width: 150px; height: 150px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 20px auto; background: conic-gradient(${color} ${score * 3.6}deg, #333 0deg); }
        .score-inner { width: 120px; height: 120px; border-radius: 50%; background: var(--vscode-editor-background); display: flex; align-items: center; justify-content: center; }
        .score-value { font-size: 36px; font-weight: bold; color: ${color}; }
        .factors { margin-top: 30px; }
        .factor { display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #333; }
        .factor-name { color: var(--vscode-foreground); }
        .factor-value { font-weight: bold; }
        h2 { color: var(--vscode-foreground); }
    </style>
</head>
<body>
    <h2>Trust Score Analysis</h2>
    <div class="score-circle">
        <div class="score-inner">
            <span class="score-value">${score}%</span>
        </div>
    </div>
    <div class="factors">
        <div class="factor">
            <span class="factor-name">AI Detection Probability</span>
            <span class="factor-value">${trustScore.ai_probability || 0}%</span>
        </div>
        <div class="factor">
            <span class="factor-name">Code Complexity</span>
            <span class="factor-value">${trustScore.complexity_score || 0}%</span>
        </div>
        <div class="factor">
            <span class="factor-name">Pattern Analysis</span>
            <span class="factor-value">${trustScore.pattern_score || 0}%</span>
        </div>
        <div class="factor">
            <span class="factor-name">Quality Indicators</span>
            <span class="factor-value">${trustScore.quality_score || 0}%</span>
        </div>
        <div class="factor">
            <span class="factor-name">Verification Status</span>
            <span class="factor-value">${trustScore.verification_score || 0}%</span>
        </div>
        <div class="factor">
            <span class="factor-name">Risk Level</span>
            <span class="factor-value" style="color: ${trustScore.risk_level === 'high' ? '#e74c3c' : trustScore.risk_level === 'medium' ? '#f1c40f' : '#2ecc71'}">${trustScore.risk_level || 'unknown'}</span>
        </div>
    </div>
</body>
</html>`;
}

async function explainFinding() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active file');
        return;
    }

    // Get diagnostics at cursor position
    const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
    const cursorPos = editor.selection.active;
    
    const relevantDiagnostic = diagnostics.find(d => d.range.contains(cursorPos));
    
    if (!relevantDiagnostic) {
        vscode.window.showInformationMessage('No finding at cursor position');
        return;
    }

    try {
        const explanation = await client.explainFinding(relevantDiagnostic);
        
        const panel = vscode.window.createWebviewPanel(
            'codeverifyExplanation',
            'Finding Explanation',
            vscode.ViewColumn.Beside,
            {}
        );

        panel.webview.html = `<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; padding: 20px; line-height: 1.6; }
        h2 { color: var(--vscode-foreground); }
        .section { margin: 20px 0; padding: 15px; background: var(--vscode-editor-inactiveSelectionBackground); border-radius: 8px; }
        code { background: var(--vscode-textCodeBlock-background); padding: 2px 6px; border-radius: 4px; }
        pre { background: var(--vscode-textCodeBlock-background); padding: 15px; border-radius: 8px; overflow-x: auto; }
    </style>
</head>
<body>
    <h2>${relevantDiagnostic.message}</h2>
    <div class="section">
        <h3>Explanation</h3>
        <p>${explanation.explanation || 'No detailed explanation available.'}</p>
    </div>
    <div class="section">
        <h3>Why This Matters</h3>
        <p>${explanation.impact || 'This finding may affect code quality or security.'}</p>
    </div>
    ${explanation.fix_suggestion ? `
    <div class="section">
        <h3>Suggested Fix</h3>
        <pre><code>${explanation.fix_suggestion}</code></pre>
    </div>
    ` : ''}
    ${explanation.references ? `
    <div class="section">
        <h3>References</h3>
        <ul>
            ${explanation.references.map((ref: string) => `<li><a href="${ref}">${ref}</a></li>`).join('')}
        </ul>
    </div>
    ` : ''}
</body>
</html>`;
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to get explanation: ${error}`);
    }
}

async function debugVerification() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active file');
        return;
    }

    try {
        const debugInfo = await client.debugVerification(editor.document.uri.fsPath);
        
        const panel = vscode.window.createWebviewPanel(
            'codeverifyDebug',
            'Verification Debug',
            vscode.ViewColumn.Beside,
            { enableScripts: true }
        );

        // Create visualization of verification steps
        const stepsHtml = debugInfo.steps?.map((step: any, i: number) => `
            <div class="step ${step.status}">
                <div class="step-header">
                    <span class="step-number">${i + 1}</span>
                    <span class="step-title">${step.title}</span>
                    <span class="step-status">${step.status === 'passed' ? '✓' : step.status === 'failed' ? '✗' : '○'}</span>
                </div>
                <div class="step-body">
                    <p>${step.description}</p>
                    ${step.constraint ? `<pre><code>${step.constraint}</code></pre>` : ''}
                </div>
            </div>
        `).join('') || '<p>No verification steps available.</p>';

        panel.webview.html = `<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; padding: 20px; }
        h2 { color: var(--vscode-foreground); }
        .step { margin: 10px 0; border: 1px solid #444; border-radius: 8px; overflow: hidden; }
        .step.passed { border-color: #2ecc71; }
        .step.failed { border-color: #e74c3c; }
        .step-header { display: flex; align-items: center; padding: 10px 15px; background: var(--vscode-editor-inactiveSelectionBackground); }
        .step-number { width: 24px; height: 24px; border-radius: 50%; background: #444; color: white; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-size: 12px; }
        .step.passed .step-number { background: #2ecc71; }
        .step.failed .step-number { background: #e74c3c; }
        .step-title { flex: 1; font-weight: 500; }
        .step-status { font-size: 18px; }
        .step-body { padding: 15px; }
        pre { background: var(--vscode-textCodeBlock-background); padding: 10px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <h2>Verification Debug Trace</h2>
    <p>File: ${editor.document.uri.fsPath}</p>
    <p>Result: ${debugInfo.result || 'unknown'}</p>
    <hr>
    <h3>Verification Steps</h3>
    ${stepsHtml}
</body>
</html>`;
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to debug verification: ${error}`);
    }
}

function toggleRealTimeVerification() {
    realTimeEnabled = !realTimeEnabled;
    
    const config = vscode.workspace.getConfiguration('codeverify');
    config.update('realTimeVerification', realTimeEnabled, vscode.ConfigurationTarget.Global);
    
    vscode.window.showInformationMessage(
        `CodeVerify: Real-time verification ${realTimeEnabled ? 'enabled' : 'disabled'}`
    );
}

async function analyzeWorkspace() {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        vscode.window.showWarningMessage('No workspace folder open');
        return;
    }

    try {
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'CodeVerify: Analyzing workspace...',
                cancellable: true,
            },
            async (progress, token) => {
                for (const folder of workspaceFolders) {
                    if (token.isCancellationRequested) {
                        break;
                    }
                    
                    progress.report({ message: folder.name });
                    const findings = await client.analyzeDirectory(folder.uri.fsPath);
                    
                    // Group findings by file and update diagnostics
                    const byFile = new Map<string, any[]>();
                    for (const finding of findings) {
                        const file = finding.file_path;
                        if (!byFile.has(file)) {
                            byFile.set(file, []);
                        }
                        byFile.get(file)!.push(finding);
                    }
                    
                    for (const [file, fileFindings] of byFile) {
                        const uri = vscode.Uri.file(file);
                        diagnosticsProvider.updateDiagnostics(uri, fileFindings);
                    }
                    
                    findingsTreeProvider.updateAllFindings(findings);
                }
            }
        );
        
        vscode.window.showInformationMessage('CodeVerify: Workspace analysis complete');
    } catch (error) {
        vscode.window.showErrorMessage(`CodeVerify analysis failed: ${error}`);
    }
}

function showFindings() {
    vscode.commands.executeCommand('codeverifyFindings.focus');
}

async function applyFix(finding: any) {
    if (!finding.fix_suggestion) {
        vscode.window.showWarningMessage('No fix available for this finding');
        return;
    }

    const document = await vscode.workspace.openTextDocument(finding.file_path);
    const edit = new vscode.WorkspaceEdit();
    
    // Create range for the fix
    const startLine = (finding.line_start || 1) - 1;
    const endLine = (finding.line_end || finding.line_start || 1) - 1;
    const range = new vscode.Range(startLine, 0, endLine + 1, 0);
    
    edit.replace(document.uri, range, finding.fix_suggestion + '\n');
    
    const applied = await vscode.workspace.applyEdit(edit);
    if (applied) {
        vscode.window.showInformationMessage('Fix applied successfully');
        // Re-analyze the file
        analyzeDocument(document);
    } else {
        vscode.window.showErrorMessage('Failed to apply fix');
    }
}

async function dismissFinding(finding: any) {
    try {
        await client.dismissFinding(finding.id, 'Dismissed by user');
        findingsTreeProvider.removeFinding(finding);
        vscode.window.showInformationMessage('Finding dismissed');
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to dismiss finding: ${error}`);
    }
}

function openDashboard() {
    const config = vscode.workspace.getConfiguration('codeverify');
    const endpoint = config.get('apiEndpoint', 'https://api.codeverify.io');
    const dashboardUrl = endpoint.replace('/api', '').replace('api.', '') + '/dashboard';
    vscode.env.openExternal(vscode.Uri.parse(dashboardUrl));
}

function refreshFindings() {
    // Re-analyze all open documents
    vscode.workspace.textDocuments.forEach(doc => analyzeDocument(doc));
}

function isSupportedLanguage(languageId: string): boolean {
    const supported = ['python', 'typescript', 'javascript', 'typescriptreact', 'javascriptreact', 'go', 'java', 'rust'];
    return supported.includes(languageId);
}

// =============================================================================
// Continuous Verification Functions
// =============================================================================

function toggleContinuousMode() {
    continuousVerificationProvider.toggle();
    
    const config = vscode.workspace.getConfiguration('codeverify');
    const isEnabled = config.get('continuousVerification', false);
    config.update('continuousVerification', !isEnabled, vscode.ConfigurationTarget.Global);
    
    vscode.window.showInformationMessage(
        `CodeVerify: Continuous verification ${!isEnabled ? 'enabled' : 'disabled'}`
    );
    
    // Initialize if enabling
    if (!isEnabled) {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            continuousVerificationProvider.initializeDocument(editor.document);
        }
    }
}

async function setContinuousMode(mode?: string) {
    if (!mode) {
        // Show quick pick to select mode
        const modes = [
            { label: '⚡ Quick', description: '~100ms latency, fast pattern checks', value: 'quick' },
            { label: '⚖️ Standard', description: '~300ms latency, balanced analysis', value: 'standard' },
            { label: '🔍 Deep', description: '~500ms+ latency, full formal verification', value: 'deep' },
        ];
        
        const selected = await vscode.window.showQuickPick(modes, {
            placeHolder: 'Select verification mode',
        });
        
        if (selected) {
            mode = selected.value;
        } else {
            return;
        }
    }
    
    const modeMap: Record<string, VerificationMode> = {
        'quick': VerificationMode.QUICK,
        'standard': VerificationMode.STANDARD,
        'deep': VerificationMode.DEEP,
    };
    
    const verificationMode = modeMap[mode.toLowerCase()];
    if (verificationMode) {
        continuousVerificationProvider.setMode(verificationMode);
        vscode.window.showInformationMessage(`CodeVerify: Verification mode set to ${mode}`);
    }
}

function showHeatMap() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }
    
    continuousVerificationProvider.showHeatMap(editor);
    vscode.window.showInformationMessage('CodeVerify: Heat map enabled');
}

function hideHeatMap() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }
    
    continuousVerificationProvider.hideHeatMap(editor);
    vscode.window.showInformationMessage('CodeVerify: Heat map disabled');
}

function showConstraints() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }
    
    const position = editor.selection.active;
    const visualization = continuousVerificationProvider.getConstraintVisualization(
        editor.document,
        position
    );
    
    if (!visualization) {
        vscode.window.showInformationMessage('No constraints at cursor position');
        return;
    }
    
    // Show constraints in a panel
    const panel = vscode.window.createWebviewPanel(
        'codeverifyConstraints',
        'Verification Constraints',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );
    
    const variablesList = visualization.variables.length 
        ? visualization.variables.map(v => `<span class="var">${v}</span>`).join(' ')
        : 'None';
    
    const constraintsList = visualization.constraints.length
        ? visualization.constraints.map(c => `<div class="constraint">${escapeHtml(c)}</div>`).join('')
        : '<p>No constraints</p>';
    
    const statusClass = visualization.satisfiable ? 'satisfied' : 'unsatisfied';
    const statusText = visualization.satisfiable ? 'Satisfiable ✓' : 'Unsatisfiable ✗';
    
    panel.webview.html = `<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding: 20px;
            line-height: 1.6;
        }
        h2 { margin-top: 0; }
        .status {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: bold;
        }
        .status.satisfied { background: #d4edda; color: #155724; }
        .status.unsatisfied { background: #f8d7da; color: #721c24; }
        .section { margin: 20px 0; }
        .section-title {
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
        }
        .var {
            display: inline-block;
            background: #e9ecef;
            padding: 2px 8px;
            border-radius: 4px;
            margin: 2px;
            font-family: 'Fira Code', monospace;
        }
        .constraint {
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 4px 0;
            font-family: 'Fira Code', monospace;
            font-size: 13px;
            border-left: 3px solid #007acc;
        }
    </style>
</head>
<body>
    <h2>Constraint Analysis</h2>
    <p><span class="status ${statusClass}">${statusText}</span></p>
    
    <div class="section">
        <div class="section-title">Variables (${visualization.variables.length})</div>
        <div>${variablesList}</div>
    </div>
    
    <div class="section">
        <div class="section-title">Constraints (${visualization.constraints.length})</div>
        ${constraintsList}
    </div>
</body>
</html>`;
}

function escapeHtml(text: string): string {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// =============================================================================
// AI Pair Reviewer Functions
// =============================================================================

function togglePairReviewer() {
    pairReviewerProvider.toggle();
    
    const config = vscode.workspace.getConfiguration('codeverify');
    const isEnabled = config.get('pairReviewer', false);
    config.update('pairReviewer', !isEnabled, vscode.ConfigurationTarget.Global);
    
    vscode.window.showInformationMessage(
        `CodeVerify: AI Pair Reviewer ${!isEnabled ? 'enabled' : 'disabled'}`
    );
    
    // Initialize if enabling
    if (!isEnabled) {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            pairReviewerProvider.initializeDocument(editor.document);
        }
    }
}

async function reviewUnit(unit: any) {
    if (!unit) {
        vscode.window.showWarningMessage('No unit to review');
        return;
    }
    
    // Trigger immediate review for this unit
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        await pairReviewerProvider.initializeDocument(editor.document);
        vscode.window.showInformationMessage(`Reviewing ${unit.name || 'unit'}...`);
    }
}

async function applyFixes(unit: any, findings: any[]) {
    if (!unit || !findings || findings.length === 0) {
        vscode.window.showWarningMessage('No fixes available');
        return;
    }
    
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }
    
    const edit = new vscode.WorkspaceEdit();
    let fixCount = 0;
    
    for (const finding of findings) {
        if (finding.fixCode) {
            const range = new vscode.Range(
                finding.lineStart,
                finding.colStart || 0,
                finding.lineEnd,
                finding.colEnd || editor.document.lineAt(finding.lineEnd).text.length
            );
            edit.replace(editor.document.uri, range, finding.fixCode);
            fixCount++;
        }
    }
    
    if (fixCount > 0) {
        const applied = await vscode.workspace.applyEdit(edit);
        if (applied) {
            vscode.window.showInformationMessage(`Applied ${fixCount} fix(es)`);
            // Record feedback
            for (const finding of findings) {
                if (finding.fixCode) {
                    pairReviewerProvider.recordFeedback(finding, 'accepted');
                }
            }
        } else {
            vscode.window.showErrorMessage('Failed to apply fixes');
        }
    }
}

async function dismissPairReviewFinding(finding: any, reason?: string) {
    if (!finding) {
        vscode.window.showWarningMessage('No finding to dismiss');
        return;
    }
    
    // Ask for reason if not provided
    if (!reason) {
        reason = await vscode.window.showInputBox({
            prompt: 'Why are you dismissing this finding? (optional)',
            placeHolder: 'False positive, not applicable, etc.',
        });
    }
    
    pairReviewerProvider.recordFeedback(finding, 'dismissed', reason);
    vscode.window.showInformationMessage('Finding dismissed - feedback recorded');
}

function showPairReviewStats() {
    const stats = pairReviewerProvider.getStatistics();
    
    const panel = vscode.window.createWebviewPanel(
        'codeverifyPairReviewStats',
        'AI Pair Review Statistics',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );
    
    const statsObj = stats as any;
    
    panel.webview.html = `<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding: 20px;
            line-height: 1.6;
        }
        h2 { margin-top: 0; color: var(--vscode-foreground); }
        .stat-card {
            background: var(--vscode-editor-inactiveSelectionBackground);
            padding: 15px 20px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            font-size: 12px;
            text-transform: uppercase;
            color: #666;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .section { margin: 20px 0; }
        .section-title {
            font-weight: 600;
            margin-bottom: 10px;
            border-bottom: 1px solid #333;
            padding-bottom: 5px;
        }
    </style>
</head>
<body>
    <h2>AI Pair Review Statistics</h2>
    
    <div class="grid">
        <div class="stat-card">
            <div class="stat-value">${statsObj.totalUnits || 0}</div>
            <div class="stat-label">Total Units</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${statsObj.verified || 0}</div>
            <div class="stat-label">Verified</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${statsObj.pending || 0}</div>
            <div class="stat-label">Pending</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${statsObj.totalFindings || 0}</div>
            <div class="stat-label">Findings</div>
        </div>
    </div>
    
    <div class="section">
        <div class="section-title">Cache & Performance</div>
        <div class="stat-card">
            <p>Cache Size: ${statsObj.cacheSize || 0} entries</p>
            <p>Active Reviews: ${statsObj.activeReviews || 0}</p>
        </div>
    </div>
    
    <div class="section">
        <div class="section-title">Feedback Learning</div>
        <div class="stat-card">
            <p>Total Feedback: ${(statsObj.feedbackStats as any)?.totalFeedback || 0}</p>
            <p>The system learns from your feedback to reduce false positives over time.</p>
        </div>
    </div>
</body>
</html>`;
}

// =============================================================================
// Paste Interception Functions
// =============================================================================

function togglePasteInterception() {
    pasteInterceptionProvider.toggle();

    const config = vscode.workspace.getConfiguration('codeverify');
    const isEnabled = pasteInterceptionProvider.isEnabled();
    config.update('pasteInterception.enabled', isEnabled, vscode.ConfigurationTarget.Global);

    vscode.window.showInformationMessage(
        `CodeVerify: Paste Interception ${isEnabled ? 'enabled' : 'disabled'}`
    );
}

function showPasteInterceptionStats() {
    const stats = pasteInterceptionProvider.getStatistics();

    const panel = vscode.window.createWebviewPanel(
        'codeverifyPasteStats',
        'Paste Interception Statistics',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    const aiDetectionRate = stats.totalInterceptions > 0
        ? ((stats.aiDetectedCount / stats.totalInterceptions) * 100).toFixed(1)
        : '0';

    const acceptRate = stats.totalInterceptions > 0
        ? ((stats.acceptedCount / stats.totalInterceptions) * 100).toFixed(1)
        : '0';

    panel.webview.html = `<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding: 20px;
            line-height: 1.6;
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
        }
        h2 { margin-top: 0; }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: var(--vscode-editor-inactiveSelectionBackground);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #3498db;
        }
        .stat-value.warning { color: #f39c12; }
        .stat-value.success { color: #27ae60; }
        .stat-value.danger { color: #e74c3c; }
        .stat-label {
            font-size: 12px;
            text-transform: uppercase;
            color: var(--vscode-descriptionForeground);
            margin-top: 5px;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            background: var(--vscode-editor-inactiveSelectionBackground);
            border-radius: 12px;
        }
        .section-title {
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
            color: var(--vscode-descriptionForeground);
            margin-bottom: 15px;
        }
        .bar {
            height: 20px;
            background: #333;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .bar-fill {
            height: 100%;
            transition: width 0.3s;
        }
        .bar-fill.success { background: #27ae60; }
        .bar-fill.warning { background: #f39c12; }
        .bar-fill.danger { background: #e74c3c; }
        .legend {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 12px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h2>Paste Interception Statistics</h2>

    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">${stats.totalInterceptions}</div>
            <div class="stat-label">Total Interceptions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value warning">${stats.aiDetectedCount}</div>
            <div class="stat-label">AI Code Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-value success">${stats.acceptedCount}</div>
            <div class="stat-label">Accepted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value danger">${stats.rejectedCount}</div>
            <div class="stat-label">Rejected</div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Decision Breakdown</div>
        <div class="bar">
            <div class="bar-fill success" style="width: ${acceptRate}%; display: inline-block;"></div>
            <div class="bar-fill danger" style="width: ${100 - parseFloat(acceptRate)}%; display: inline-block;"></div>
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #27ae60;"></div>
                <span>Accepted (${acceptRate}%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e74c3c;"></div>
                <span>Rejected (${(100 - parseFloat(acceptRate)).toFixed(1)}%)</span>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Performance</div>
        <p>Average Trust Score: <strong>${stats.averageTrustScore.toFixed(1)}%</strong></p>
        <p>Average Analysis Time: <strong>${stats.averageAnalysisTimeMs.toFixed(0)}ms</strong></p>
        <p>AI Detection Rate: <strong>${aiDetectionRate}%</strong></p>
    </div>

    <div class="section">
        <div class="section-title">Actions</div>
        <p>Modified Before Accept: <strong>${stats.modifiedCount}</strong></p>
        <p><em>The system helps catch potentially problematic AI-generated code before it enters your codebase.</em></p>
    </div>
</body>
</html>`;
}
