/**
 * Hallucination Detection Provider
 *
 * Detects when AI coding assistants hallucinate non-existent APIs,
 * modules, or function signatures in the active editor.
 */

import * as vscode from 'vscode';

interface HallucinationFinding {
    importPath: string;
    symbol: string;
    line: number;
    suggestion: string;
    confidence: number;
}

// Known hallucinated APIs that AI models commonly generate
const KNOWN_HALLUCINATIONS: Record<string, Record<string, string>> = {
    python: {
        'requests.get_json': 'Use response.json() method instead',
        'os.path.joinpath': 'Use os.path.join() or pathlib.Path.joinpath()',
        'json.loads_file': 'Use json.load(open(f)) or json.loads(f.read())',
        'collections.DefaultDict': 'Use collections.defaultdict (lowercase)',
        'os.env': 'Use os.environ',
        'os.get_env': 'Use os.environ.get() or os.getenv()',
        'subprocess.execute': 'Use subprocess.run() or subprocess.call()',
        'itertools.flatten': 'Use itertools.chain.from_iterable()',
        'functools.memorize': 'Use functools.lru_cache',
        'asyncio.async': 'Use asyncio.create_task()',
        're.match_all': 'Use re.findall() or re.finditer()',
        'sys.args': 'Use sys.argv',
    },
    typescript: {
        'Array.flat_map': 'Use Array.flatMap() (camelCase)',
        'Promise.allResolved': 'Use Promise.allSettled()',
        'Object.deep_copy': 'Use structuredClone() or JSON parse/stringify',
        'String.contains': 'Use String.includes()',
        'fs.readFileAsync': 'Use fs.promises.readFile()',
    },
    go: {
        'fmt.FormatString': 'Use fmt.Sprintf()',
        'strings.Format': 'Use fmt.Sprintf()',
        'errors.Newf': 'Use fmt.Errorf()',
    },
    java: {
        'System.println': 'Use System.out.println()',
        'Collections.sort_by': 'Use list.sort(Comparator.comparing(...))',
    },
};

export class HallucinationProvider implements vscode.Disposable {
    private diagnosticCollection: vscode.DiagnosticCollection;
    private disposables: vscode.Disposable[] = [];
    private decorationType: vscode.TextEditorDecorationType;

    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeverify-hallucination');

        this.decorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: new vscode.ThemeColor('editorWarning.background'),
            border: '1px dashed',
            borderColor: new vscode.ThemeColor('editorWarning.foreground'),
            after: {
                contentText: ' (hallucinated API?)',
                color: new vscode.ThemeColor('editorWarning.foreground'),
                fontStyle: 'italic',
            },
        });

        this.disposables.push(
            vscode.workspace.onDidChangeTextDocument((e) => {
                this.checkDocument(e.document);
            }),
            vscode.workspace.onDidOpenTextDocument((doc) => {
                this.checkDocument(doc);
            })
        );

        // Check all open documents on activation
        for (const editor of vscode.window.visibleTextEditors) {
            this.checkDocument(editor.document);
        }
    }

    checkDocument(document: vscode.TextDocument): void {
        const lang = document.languageId;
        const hallucinations = KNOWN_HALLUCINATIONS[lang];
        if (!hallucinations) {
            return;
        }

        const text = document.getText();
        const diagnostics: vscode.Diagnostic[] = [];
        const lines = text.split('\n');

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            for (const [api, suggestion] of Object.entries(hallucinations)) {
                if (line.includes(api)) {
                    const col = line.indexOf(api);
                    const range = new vscode.Range(
                        new vscode.Position(i, col),
                        new vscode.Position(i, col + api.length)
                    );

                    const diagnostic = new vscode.Diagnostic(
                        range,
                        `Potential hallucinated API: '${api}'. ${suggestion}`,
                        vscode.DiagnosticSeverity.Warning
                    );
                    diagnostic.source = 'codeverify-hallucination';
                    diagnostic.code = 'HALLUCINATION';
                    diagnostics.push(diagnostic);
                }
            }
        }

        this.diagnosticCollection.set(document.uri, diagnostics);
    }

    dispose(): void {
        this.diagnosticCollection.dispose();
        this.decorationType.dispose();
        for (const d of this.disposables) {
            d.dispose();
        }
    }
}
