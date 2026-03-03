'use client';

import { useState } from 'react';
import {
  Shield,
  ChevronRight,
  ChevronDown,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Play,
  Edit3,
  MessageCircle,
  Clock,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ProofTreeNode {
  id: string;
  type: string;
  label: string;
  expression: string;
  status: 'proved' | 'disproved' | 'unknown';
  children: ProofTreeNode[];
}

interface ConstraintVariable {
  id: string;
  name: string;
  type: string;
}

interface ConstraintEdge {
  source: string;
  target: string;
  constraint: string;
  satisfied: boolean;
}

interface CounterexampleVariable {
  name: string;
  value: unknown;
  type: string;
}

interface Counterexample {
  id: string;
  variables: CounterexampleVariable[];
  violated_property: string;
  execution_path: Array<{ step: number; line: number; action: string; error?: string }>;
}

interface Exploration {
  id: string;
  function_name: string;
  status: string;
  proof_tree: ProofTreeNode | null;
  constraint_graph: {
    variables: ConstraintVariable[];
    edges: ConstraintEdge[];
  } | null;
  counterexamples: Counterexample[];
  execution_trace: Array<{ step: number; type: string; description: string; time_ms: number }>;
  solver_time_ms: number;
}

// ---------------------------------------------------------------------------
// Mock data for initial render
// ---------------------------------------------------------------------------

const mockExploration: Exploration = {
  id: 'demo-1',
  function_name: 'process_user_input',
  status: 'disproved',
  proof_tree: {
    id: 'root',
    type: 'root',
    label: 'Verify: process_user_input',
    expression: 'verify(process_user_input)',
    status: 'disproved',
    children: [
      {
        id: 'null-safety',
        type: 'assertion',
        label: 'Null Safety',
        expression: 'ForAll([x], Implies(is_param(x), x != None))',
        status: 'disproved',
        children: [
          { id: 'ns-1', type: 'constraint', label: 'Parameter non-null', expression: 'param != None', status: 'disproved', children: [] },
          { id: 'ns-2', type: 'constraint', label: 'Return non-null', expression: 'result != None', status: 'proved', children: [] },
        ],
      },
      {
        id: 'bounds',
        type: 'assertion',
        label: 'Bounds Check',
        expression: 'ForAll([i, a], Implies(access(a, i), And(i >= 0, i < len(a))))',
        status: 'proved',
        children: [
          { id: 'bc-1', type: 'constraint', label: 'Lower bound', expression: 'index >= 0', status: 'proved', children: [] },
          { id: 'bc-2', type: 'constraint', label: 'Upper bound', expression: 'index < len(array)', status: 'proved', children: [] },
        ],
      },
    ],
  },
  constraint_graph: {
    variables: [
      { id: 'param', name: 'param', type: 'Optional[str]' },
      { id: 'index', name: 'index', type: 'int' },
      { id: 'result', name: 'result', type: 'str' },
    ],
    edges: [
      { source: 'param', target: 'result', constraint: 'result depends on param', satisfied: true },
      { source: 'index', target: 'result', constraint: 'index used in access', satisfied: true },
    ],
  },
  counterexamples: [
    {
      id: 'ce-1',
      variables: [
        { name: 'param', value: null, type: 'NoneType' },
        { name: 'index', value: 0, type: 'int' },
      ],
      violated_property: 'Null Safety: param != None',
      execution_path: [
        { step: 1, line: 1, action: 'enter' },
        { step: 2, line: 3, action: 'access', error: 'NoneType has no attribute' },
      ],
    },
  ],
  execution_trace: [
    { step: 1, type: 'parse', description: 'Parse process_user_input', time_ms: 2.1 },
    { step: 2, type: 'extract', description: 'Extract verification constraints', time_ms: 5.3 },
    { step: 3, type: 'solve', description: 'Run Z3 SMT solver', time_ms: 45.7 },
    { step: 4, type: 'check', description: 'Check satisfiability', time_ms: 12.4 },
  ],
  solver_time_ms: 65.5,
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'proved':
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case 'disproved':
      return <XCircle className="h-4 w-4 text-red-500" />;
    default:
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
  }
}

function ProofTreeView({ node, depth = 0 }: { node: ProofTreeNode; depth?: number }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = node.children.length > 0;

  return (
    <div className={`${depth > 0 ? 'ml-6 border-l border-gray-200 dark:border-gray-700 pl-4' : ''}`}>
      <div
        className="flex items-center gap-2 py-1.5 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded px-2"
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        {hasChildren ? (
          expanded ? <ChevronDown className="h-4 w-4 text-gray-400" /> : <ChevronRight className="h-4 w-4 text-gray-400" />
        ) : (
          <span className="w-4" />
        )}
        <StatusIcon status={node.status} />
        <span className="font-medium text-sm text-gray-900 dark:text-white">{node.label}</span>
        <code className="text-xs text-gray-500 bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded ml-2">
          {node.expression}
        </code>
      </div>
      {expanded && node.children.map((child) => (
        <ProofTreeView key={child.id} node={child} depth={depth + 1} />
      ))}
    </div>
  );
}

function ConstraintGraphView({ graph }: { graph: Exploration['constraint_graph'] }) {
  if (!graph) return null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        {graph.variables.map((v) => (
          <div key={v.id} className="px-3 py-2 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg">
            <span className="font-mono text-sm font-medium text-blue-800 dark:text-blue-200">{v.name}</span>
            <span className="text-xs text-blue-500 ml-2">{v.type}</span>
          </div>
        ))}
      </div>
      <div className="space-y-2">
        {graph.edges.map((e, i) => (
          <div key={i} className="flex items-center gap-2 text-sm">
            <code className="text-blue-600 dark:text-blue-400">{e.source}</code>
            <span className="text-gray-400">→</span>
            <code className="text-blue-600 dark:text-blue-400">{e.target}</code>
            <span className={`text-xs px-2 py-0.5 rounded ${e.satisfied ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'}`}>
              {e.constraint}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CounterexampleView({ ce }: { ce: Counterexample }) {
  const [editing, setEditing] = useState(false);

  return (
    <div className="border border-red-200 dark:border-red-800 rounded-lg p-4 bg-red-50/50 dark:bg-red-900/20">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-red-800 dark:text-red-200">
          Violates: {ce.violated_property}
        </h4>
        <button
          onClick={() => setEditing(!editing)}
          className="flex items-center gap-1 text-xs text-red-600 hover:text-red-800 dark:text-red-400"
        >
          <Edit3 className="h-3 w-3" />
          {editing ? 'Close' : 'Edit Values'}
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-3">
        {ce.variables.map((v, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="font-mono text-sm text-gray-700 dark:text-gray-300">{v.name} =</span>
            {editing ? (
              <input
                type="text"
                defaultValue={String(v.value)}
                className="text-sm border rounded px-2 py-1 w-24 bg-white dark:bg-gray-800 dark:text-white"
              />
            ) : (
              <code className="text-sm text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/50 px-2 py-0.5 rounded">
                {String(v.value)}
              </code>
            )}
            <span className="text-xs text-gray-400">({v.type})</span>
          </div>
        ))}
      </div>

      <div className="space-y-1">
        <h5 className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase">Execution Path</h5>
        {ce.execution_path.map((step) => (
          <div key={step.step} className={`text-xs flex items-center gap-2 ${step.error ? 'text-red-600' : 'text-gray-600 dark:text-gray-400'}`}>
            <span className="font-mono w-8">L{step.line}</span>
            <span>{step.action}</span>
            {step.error && <span className="text-red-500 font-medium">⚠ {step.error}</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function ProofExplorerPage() {
  const [activeTab, setActiveTab] = useState<'tree' | 'graph' | 'counterexamples' | 'trace'>('tree');
  const exploration = mockExploration;

  const tabs = [
    { id: 'tree' as const, label: 'Proof Tree', icon: Shield },
    { id: 'graph' as const, label: 'Constraint Graph', icon: Shield },
    { id: 'counterexamples' as const, label: `Counterexamples (${exploration.counterexamples.length})`, icon: XCircle },
    { id: 'trace' as const, label: 'Execution Trace', icon: Play },
  ];

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Proof Explorer
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Interactive visualization of Z3 verification results
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1 text-sm font-medium rounded-full ${exploration.status === 'proved' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            {exploration.status === 'proved' ? '✅ Verified' : '❌ Issues Found'}
          </span>
          <span className="flex items-center gap-1 text-xs text-gray-500">
            <Clock className="h-3 w-3" />
            {exploration.solver_time_ms}ms
          </span>
        </div>
      </div>

      {/* Function info */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 mb-6">
        <div className="flex items-center gap-4">
          <code className="text-lg font-mono font-semibold text-gray-900 dark:text-white">
            {exploration.function_name}()
          </code>
          <button className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800">
            <MessageCircle className="h-4 w-4" />
            Ask &quot;Why is this unsafe?&quot;
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
        <nav className="flex gap-6">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`pb-3 text-sm font-medium border-b-2 transition ${activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab content */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        {activeTab === 'tree' && exploration.proof_tree && (
          <ProofTreeView node={exploration.proof_tree} />
        )}

        {activeTab === 'graph' && (
          <ConstraintGraphView graph={exploration.constraint_graph} />
        )}

        {activeTab === 'counterexamples' && (
          <div className="space-y-4">
            {exploration.counterexamples.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No counterexamples — all properties hold ✅</p>
            ) : (
              exploration.counterexamples.map((ce) => (
                <CounterexampleView key={ce.id} ce={ce} />
              ))
            )}
          </div>
        )}

        {activeTab === 'trace' && (
          <div className="space-y-2">
            {exploration.execution_trace.map((step) => (
              <div key={step.step} className="flex items-center gap-4 py-2 border-b border-gray-100 dark:border-gray-700 last:border-0">
                <span className="text-xs font-mono text-gray-400 w-6">#{step.step}</span>
                <span className="text-sm font-medium text-gray-900 dark:text-white flex-1">
                  {step.description}
                </span>
                <span className="text-xs text-gray-500">{step.time_ms}ms</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
