/**
 * Tests for FormalSpecAssistantProvider
 *
 * `../formalSpecAssistantProvider` imports `vscode` and creates a
 * `vscode.OutputChannel`/code lens provider in its constructor, so the class
 * itself can only be exercised inside the VS Code extension host. All of its
 * actual conversion/suggestion/template algorithms, however, live in the
 * pure, `vscode`-free `../../localAnalysis` and `../../collections` modules
 * (moved there verbatim so they can be unit tested directly), and the
 * provider delegates to them (see `convertToZ3`, `suggestSpecsForCurrentFunction`,
 * `showTemplateLibrary`/`useTemplate` in formalSpecAssistantProvider.ts). This
 * file imports and exercises those real functions instead of hand-copied
 * simulations of them.
 */

import * as assert from 'assert';
import { describe, it } from 'node:test';
import type { NLToZ3Result } from '../../client';
import {
    localNLToZ3,
    localSuggestSpecs,
    getLocalTemplates,
    fillTemplate,
} from '../../localAnalysis';
import { pushBounded } from '../../collections';

describe('FormalSpecAssistantProvider', () => {
    describe('NL Pattern Matching (localNLToZ3)', () => {
        it('should convert positive constraint', () => {
            const result = localNLToZ3('x must be positive');
            assert.strictEqual(result.success, true);
            assert.strictEqual(result.z3_expr, 'x > 0');
            assert.strictEqual(result.python_assert, 'assert x > 0');
            assert.strictEqual(result.variables['x'], 'Int');
        });

        it('should convert non-negative constraint', () => {
            const result = localNLToZ3('index must be non-negative');
            assert.strictEqual(result.success, true);
            assert.strictEqual(result.z3_expr, 'index >= 0');
            assert.strictEqual(result.python_assert, 'assert index >= 0');
        });

        it('should convert range constraint', () => {
            const result = localNLToZ3('age must be between 0 and 150');
            assert.strictEqual(result.success, true);
            assert.strictEqual(result.z3_expr, 'And(age >= 0, age <= 150)');
            assert.strictEqual(result.python_assert, 'assert 0 <= age <= 150');
        });

        it('should convert less than constraint', () => {
            const result = localNLToZ3('start must be less than end');
            assert.strictEqual(result.success, true);
            assert.strictEqual(result.z3_expr, 'start < end');
            assert.strictEqual(result.variables['start'], 'Int');
            assert.strictEqual(result.variables['end'], 'Int');
        });

        it('should convert not null constraint', () => {
            const result = localNLToZ3('user must not be null');
            assert.strictEqual(result.success, true);
            assert.strictEqual(result.z3_expr, 'user != None');
            assert.strictEqual(result.python_assert, 'assert user is not None');
        });

        it('should convert not empty constraint', () => {
            const result = localNLToZ3('items must not be empty');
            assert.strictEqual(result.success, true);
            assert.strictEqual(result.z3_expr, 'Length(items) > 0');
            assert.strictEqual(result.python_assert, 'assert len(items) > 0');
            assert.strictEqual(result.variables['items'], 'Seq');
        });

        it('should handle different phrasings', () => {
            const phrasings = [
                'x must be positive',
                'x is positive',
                'x should be positive',
            ];

            for (const phrase of phrasings) {
                const result = localNLToZ3(phrase);
                assert.strictEqual(result.success, true, `Failed for: ${phrase}`);
                assert.ok(result.z3_expr?.includes('> 0'), `Missing > 0 for: ${phrase}`);
            }
        });

        it('should return failure for unrecognized patterns', () => {
            const result = localNLToZ3('something completely different');
            assert.strictEqual(result.success, false);
            assert.ok(result.ambiguities.length > 0);
        });

        it('should include confidence score', () => {
            const result = localNLToZ3('x must be positive');
            assert.ok(result.confidence > 0);
            assert.ok(result.confidence <= 1);
        });
    });

    describe('Spec Suggestions (localSuggestSpecs)', () => {
        it('should suggest specs for int parameters', () => {
            const result = localSuggestSpecs('def process(count: int) -> int:');
            assert.ok(result.count > 0);
            assert.ok(result.suggestions.some(s => s.includes('count')));
            assert.ok(result.suggestions.some(s => s.includes('positive')));
        });

        it('should suggest specs for string parameters', () => {
            const result = localSuggestSpecs('def greet(name: str) -> str:');
            assert.ok(result.count > 0);
            assert.ok(result.suggestions.some(s => s.includes('name')));
            assert.ok(result.suggestions.some(s => s.includes('empty')));
        });

        it('should suggest return value specs', () => {
            const result = localSuggestSpecs('def calculate(x: int) -> int:');
            assert.ok(result.suggestions.some(s => s.includes('returns')));
        });

        it('should handle multiple parameters', () => {
            const result = localSuggestSpecs('def add(a: int, b: int) -> int:');
            assert.ok(result.suggestions.some(s => s.includes('a')));
            assert.ok(result.suggestions.some(s => s.includes('b')));
        });
    });

    describe('Template Library (getLocalTemplates)', () => {
        it('should have numeric templates', () => {
            const templates = getLocalTemplates();
            const numeric = templates.filter(t => t.domain === 'numeric');
            assert.ok(numeric.length > 0);
        });

        it('should have general templates', () => {
            const templates = getLocalTemplates();
            const general = templates.filter(t => t.domain === 'general');
            assert.ok(general.length > 0);
        });

        it('should have collection templates', () => {
            const templates = getLocalTemplates();
            const collection = templates.filter(t => t.domain === 'collection');
            assert.ok(collection.length > 0);
        });

        it('templates should have required fields', () => {
            const templates = getLocalTemplates();
            for (const t of templates) {
                assert.ok(t.id, 'Template missing id');
                assert.ok(t.name, 'Template missing name');
                assert.ok(t.domain, 'Template missing domain');
                assert.ok(t.nl_pattern, 'Template missing nl_pattern');
                assert.ok(t.z3_template, 'Template missing z3_template');
                assert.ok(t.smtlib_template, 'Template missing smtlib_template');
                assert.ok(t.python_template, 'Template missing python_template');
                assert.ok(t.examples.length > 0, 'Template missing examples');
            }
        });
    });

    describe('Template Variable Filling (fillTemplate)', () => {
        it('should fill single variable template', () => {
            const result = fillTemplate('{var} > 0', { var: 'x' });
            assert.strictEqual(result, 'x > 0');
        });

        it('should fill multiple variable template', () => {
            const result = fillTemplate('And({var} >= {min}, {var} <= {max})', {
                var: 'age',
                min: '0',
                max: '150',
            });
            assert.strictEqual(result, 'And(age >= 0, age <= 150)');
        });

        it('should handle repeated variables', () => {
            const result = fillTemplate('{var} >= 0 and {var} <= 100', { var: 'score' });
            assert.strictEqual(result, 'score >= 0 and score <= 100');
        });
    });

    describe('History Management (pushBounded)', () => {
        // Mirrors the SpecHistoryEntry shape and maxHistorySize=50 default
        // used by FormalSpecAssistantProvider.convertToZ3.
        interface SpecHistoryEntry {
            naturalLanguage: string;
            result: NLToZ3Result;
            timestamp: number;
        }

        function makeResult(z3Expr: string): NLToZ3Result {
            return {
                success: true,
                z3_expr: z3Expr,
                explanation: '',
                confidence: 0.9,
                variables: {},
                ambiguities: [],
                clarification_questions: [],
                processing_time_ms: 0,
            };
        }

        it('should add entries to history', () => {
            const history: SpecHistoryEntry[] = [];
            pushBounded(
                history,
                { naturalLanguage: 'x must be positive', result: makeResult('x > 0'), timestamp: Date.now() },
                10
            );

            assert.strictEqual(history.length, 1);
            assert.strictEqual(history[0].naturalLanguage, 'x must be positive');
        });

        it('should limit history size', () => {
            const history: SpecHistoryEntry[] = [];

            for (let i = 0; i < 5; i++) {
                pushBounded(
                    history,
                    { naturalLanguage: `spec ${i}`, result: makeResult(`z3_${i}`), timestamp: Date.now() },
                    3
                );
            }

            assert.strictEqual(history.length, 3);
            // Most recent should be first
            assert.strictEqual(history[0].naturalLanguage, 'spec 4');
        });

        it('should clear history', () => {
            const history: SpecHistoryEntry[] = [];
            pushBounded(
                history,
                { naturalLanguage: 'x must be positive', result: makeResult('x > 0'), timestamp: Date.now() },
                10
            );

            history.length = 0;
            assert.strictEqual(history.length, 0);
        });
    });

    describe('Spec Suggestion Parsing Edge Cases (localSuggestSpecs)', () => {
        it('should recognize typed parameters in a Python-style signature', () => {
            const result = localSuggestSpecs('def add(x: int, y: int) -> int:');
            assert.ok(result.suggestions.some(s => s.startsWith('x ')));
            assert.ok(result.suggestions.some(s => s.startsWith('y ')));
        });

        it('should not recognize TypeScript primitive type names like "number"', () => {
            // Documents actual current behavior: the heuristic only matches
            // int/integer/str/string/list/array (see localSuggestSpecs), so
            // TypeScript's `number` annotation produces no suggestions.
            const result = localSuggestSpecs('function add(x: number, y: number): number');
            assert.strictEqual(result.count, 0);
        });

        it('should return no suggestions when there are no parameters or recognized return type', () => {
            const result = localSuggestSpecs('def foo() -> None:');
            assert.strictEqual(result.count, 0);
        });
    });
});
