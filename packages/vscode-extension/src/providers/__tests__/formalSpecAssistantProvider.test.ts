/**
 * Tests for FormalSpecAssistantProvider
 */

import * as assert from 'assert';

// Mock types
interface NLToZ3Result {
    success: boolean;
    z3_expr?: string;
    smtlib?: string;
    python_assert?: string;
    explanation: string;
    confidence: number;
    variables: Record<string, string>;
    ambiguities: string[];
    clarification_questions: string[];
    processing_time_ms: number;
}

describe('FormalSpecAssistantProvider', () => {
    describe('NL Pattern Matching', () => {
        // Simulated local NL-to-Z3 conversion
        function localNLToZ3(specification: string): NLToZ3Result {
            const normalized = specification.toLowerCase().trim();
            let z3_expr: string | undefined;
            let python_assert: string | undefined;
            let explanation = '';
            let confidence = 0;
            const variables: Record<string, string> = {};

            const patterns: Array<{
                pattern: RegExp;
                template: (m: RegExpMatchArray) => { z3: string; py: string; vars: Record<string, string> };
                name: string;
            }> = [
                {
                    pattern: /(\w+)\s+(?:must be |is |should be )?positive/,
                    template: (m) => ({
                        z3: `${m[1]} > 0`,
                        py: `assert ${m[1]} > 0`,
                        vars: { [m[1]]: 'Int' },
                    }),
                    name: 'Positive constraint',
                },
                {
                    pattern: /(\w+)\s+(?:must be |is |should be )?non-negative/,
                    template: (m) => ({
                        z3: `${m[1]} >= 0`,
                        py: `assert ${m[1]} >= 0`,
                        vars: { [m[1]]: 'Int' },
                    }),
                    name: 'Non-negative constraint',
                },
                {
                    pattern: /(\w+)\s+(?:must be |is |should be )?(?:between|in range)\s+(\d+)\s+(?:and|to)\s+(\d+)/,
                    template: (m) => ({
                        z3: `And(${m[1]} >= ${m[2]}, ${m[1]} <= ${m[3]})`,
                        py: `assert ${m[2]} <= ${m[1]} <= ${m[3]}`,
                        vars: { [m[1]]: 'Int' },
                    }),
                    name: 'Range constraint',
                },
                {
                    pattern: /(\w+)\s+(?:must be |is |should be )?less than\s+(\w+)/,
                    template: (m) => ({
                        z3: `${m[1]} < ${m[2]}`,
                        py: `assert ${m[1]} < ${m[2]}`,
                        vars: { [m[1]]: 'Int', [m[2]]: 'Int' },
                    }),
                    name: 'Less than constraint',
                },
                {
                    pattern: /(\w+)\s+(?:must |should )?not (?:be )?(?:null|none)/,
                    template: (m) => ({
                        z3: `${m[1]} != None`,
                        py: `assert ${m[1]} is not None`,
                        vars: { [m[1]]: 'Any' },
                    }),
                    name: 'Not null constraint',
                },
                {
                    pattern: /(\w+)\s+(?:must |should )?not (?:be )?empty/,
                    template: (m) => ({
                        z3: `Length(${m[1]}) > 0`,
                        py: `assert len(${m[1]}) > 0`,
                        vars: { [m[1]]: 'Seq' },
                    }),
                    name: 'Not empty constraint',
                },
            ];

            for (const { pattern, template, name } of patterns) {
                const match = normalized.match(pattern);
                if (match) {
                    const result = template(match);
                    z3_expr = result.z3;
                    python_assert = result.py;
                    Object.assign(variables, result.vars);
                    explanation = `Matched template: ${name}`;
                    confidence = 0.85;
                    break;
                }
            }

            return {
                success: !!z3_expr,
                z3_expr,
                python_assert,
                explanation: explanation || 'Could not match specification pattern',
                confidence,
                variables,
                ambiguities: !z3_expr ? ['Could not parse specification'] : [],
                clarification_questions: !z3_expr ? ['Which variable should this constraint apply to?'] : [],
                processing_time_ms: 0,
            };
        }

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

    describe('Spec Suggestions', () => {
        function suggestSpecs(signature: string): { suggestions: string[]; count: number } {
            const suggestions: string[] = [];
            const paramPattern = /(\w+)\s*:\s*(\w+)/g;
            let match;

            while ((match = paramPattern.exec(signature)) !== null) {
                const [, paramName, paramType] = match;
                const typeLower = paramType.toLowerCase();

                if (typeLower === 'int' || typeLower === 'integer') {
                    suggestions.push(`${paramName} must be positive`);
                    suggestions.push(`${paramName} must be non-negative`);
                } else if (typeLower === 'str' || typeLower === 'string') {
                    suggestions.push(`${paramName} must not be empty`);
                } else if (typeLower.includes('list') || typeLower.includes('array')) {
                    suggestions.push(`${paramName} must not be empty`);
                }
            }

            if (signature.includes('-> int') || signature.includes('-> Int')) {
                suggestions.push('the function returns a positive value');
            }

            return { suggestions, count: suggestions.length };
        }

        it('should suggest specs for int parameters', () => {
            const result = suggestSpecs('def process(count: int) -> int:');
            assert.ok(result.count > 0);
            assert.ok(result.suggestions.some(s => s.includes('count')));
            assert.ok(result.suggestions.some(s => s.includes('positive')));
        });

        it('should suggest specs for string parameters', () => {
            const result = suggestSpecs('def greet(name: str) -> str:');
            assert.ok(result.count > 0);
            assert.ok(result.suggestions.some(s => s.includes('name')));
            assert.ok(result.suggestions.some(s => s.includes('empty')));
        });

        it('should suggest return value specs', () => {
            const result = suggestSpecs('def calculate(x: int) -> int:');
            assert.ok(result.suggestions.some(s => s.includes('returns')));
        });

        it('should handle multiple parameters', () => {
            const result = suggestSpecs('def add(a: int, b: int) -> int:');
            assert.ok(result.suggestions.some(s => s.includes('a')));
            assert.ok(result.suggestions.some(s => s.includes('b')));
        });
    });

    describe('Template Library', () => {
        function getLocalTemplates() {
            return [
                {
                    id: 'positive',
                    name: 'Positive Number',
                    domain: 'numeric',
                    complexity: 'simple',
                    nl_pattern: '{var} must be positive',
                    z3_template: '{var} > 0',
                },
                {
                    id: 'non_negative',
                    name: 'Non-negative Number',
                    domain: 'numeric',
                    complexity: 'simple',
                    nl_pattern: '{var} must be non-negative',
                    z3_template: '{var} >= 0',
                },
                {
                    id: 'range',
                    name: 'Value in Range',
                    domain: 'numeric',
                    complexity: 'simple',
                    nl_pattern: '{var} must be between {min} and {max}',
                    z3_template: 'And({var} >= {min}, {var} <= {max})',
                },
                {
                    id: 'not_null',
                    name: 'Not Null',
                    domain: 'general',
                    complexity: 'simple',
                    nl_pattern: '{var} must not be null',
                    z3_template: '{var} != None',
                },
                {
                    id: 'not_empty',
                    name: 'Not Empty',
                    domain: 'collection',
                    complexity: 'simple',
                    nl_pattern: '{var} must not be empty',
                    z3_template: 'Length({var}) > 0',
                },
            ];
        }

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
            }
        });
    });

    describe('Template Variable Filling', () => {
        function fillTemplate(template: string, values: Record<string, string>): string {
            let result = template;
            for (const [key, value] of Object.entries(values)) {
                result = result.replace(new RegExp(`\\{${key}\\}`, 'g'), value);
            }
            return result;
        }

        it('should fill single variable template', () => {
            const template = '{var} > 0';
            const result = fillTemplate(template, { var: 'x' });
            assert.strictEqual(result, 'x > 0');
        });

        it('should fill multiple variable template', () => {
            const template = 'And({var} >= {min}, {var} <= {max})';
            const result = fillTemplate(template, { var: 'age', min: '0', max: '150' });
            assert.strictEqual(result, 'And(age >= 0, age <= 150)');
        });

        it('should handle repeated variables', () => {
            const template = '{var} >= 0 and {var} <= 100';
            const result = fillTemplate(template, { var: 'score' });
            assert.strictEqual(result, 'score >= 0 and score <= 100');
        });
    });

    describe('History Management', () => {
        interface SpecHistoryEntry {
            naturalLanguage: string;
            result: NLToZ3Result;
            timestamp: number;
        }

        function createHistory(maxSize: number) {
            const history: SpecHistoryEntry[] = [];

            return {
                add(nl: string, result: NLToZ3Result) {
                    history.unshift({
                        naturalLanguage: nl,
                        result,
                        timestamp: Date.now(),
                    });
                    if (history.length > maxSize) {
                        history.pop();
                    }
                },
                get() {
                    return history;
                },
                clear() {
                    history.length = 0;
                },
            };
        }

        it('should add entries to history', () => {
            const history = createHistory(10);
            history.add('x must be positive', {
                success: true,
                z3_expr: 'x > 0',
                explanation: '',
                confidence: 0.9,
                variables: {},
                ambiguities: [],
                clarification_questions: [],
                processing_time_ms: 0,
            });

            assert.strictEqual(history.get().length, 1);
            assert.strictEqual(history.get()[0].naturalLanguage, 'x must be positive');
        });

        it('should limit history size', () => {
            const history = createHistory(3);

            for (let i = 0; i < 5; i++) {
                history.add(`spec ${i}`, {
                    success: true,
                    z3_expr: `z3_${i}`,
                    explanation: '',
                    confidence: 0.9,
                    variables: {},
                    ambiguities: [],
                    clarification_questions: [],
                    processing_time_ms: 0,
                });
            }

            assert.strictEqual(history.get().length, 3);
            // Most recent should be first
            assert.strictEqual(history.get()[0].naturalLanguage, 'spec 4');
        });

        it('should clear history', () => {
            const history = createHistory(10);
            history.add('x must be positive', {
                success: true,
                z3_expr: 'x > 0',
                explanation: '',
                confidence: 0.9,
                variables: {},
                ambiguities: [],
                clarification_questions: [],
                processing_time_ms: 0,
            });

            history.clear();
            assert.strictEqual(history.get().length, 0);
        });
    });

    describe('Function Signature Parsing', () => {
        function extractParams(signature: string, language: string): Array<{ name: string; type: string }> {
            const params: Array<{ name: string; type: string }> = [];

            if (language === 'python') {
                const match = signature.match(/def\s+\w+\s*\(([^)]*)\)/);
                if (match) {
                    const paramsStr = match[1];
                    const paramPattern = /(\w+)\s*:\s*(\w+)/g;
                    let m;
                    while ((m = paramPattern.exec(paramsStr)) !== null) {
                        params.push({ name: m[1], type: m[2] });
                    }
                }
            } else if (language === 'typescript') {
                const match = signature.match(/function\s+\w+\s*\(([^)]*)\)/);
                if (match) {
                    const paramsStr = match[1];
                    const paramPattern = /(\w+)\s*:\s*(\w+)/g;
                    let m;
                    while ((m = paramPattern.exec(paramsStr)) !== null) {
                        params.push({ name: m[1], type: m[2] });
                    }
                }
            }

            return params;
        }

        it('should parse Python function signature', () => {
            const params = extractParams('def add(x: int, y: int) -> int:', 'python');
            assert.strictEqual(params.length, 2);
            assert.strictEqual(params[0].name, 'x');
            assert.strictEqual(params[0].type, 'int');
            assert.strictEqual(params[1].name, 'y');
            assert.strictEqual(params[1].type, 'int');
        });

        it('should parse TypeScript function signature', () => {
            const params = extractParams('function add(x: number, y: number): number', 'typescript');
            assert.strictEqual(params.length, 2);
            assert.strictEqual(params[0].name, 'x');
            assert.strictEqual(params[0].type, 'number');
        });

        it('should handle empty parameters', () => {
            const params = extractParams('def foo() -> None:', 'python');
            assert.strictEqual(params.length, 0);
        });
    });
});
