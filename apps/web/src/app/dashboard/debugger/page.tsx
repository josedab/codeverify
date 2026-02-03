"use client";

import { useState, useCallback } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  Plus,
  Trash2,
  ChevronRight,
  CheckCircle,
  XCircle,
  AlertCircle,
  Code,
  Layers,
  Variable,
  ListChecks,
} from "lucide-react";
import api, { VerificationTrace, VerificationStep } from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  Input,
  Select,
  Badge,
  SlideOver,
} from "@/components/ui";

export default function DebuggerPage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [formula, setFormula] = useState(DEFAULT_FORMULA);
  const [currentStep, setCurrentStep] = useState(0);
  const [trace, setTrace] = useState<VerificationTrace | null>(null);

  // Create session mutation
  const createSession = useMutation({
    mutationFn: () => api.createDebuggerSession(60000),
    onSuccess: (data) => {
      setSessionId(data.session_id);
    },
  });

  // Execute action mutation
  const executeAction = useMutation({
    mutationFn: ({
      action,
      data,
    }: {
      action: string;
      data?: Record<string, unknown>;
    }) => api.executeDebuggerAction(sessionId!, action, data),
  });

  // Verify with trace mutation
  const verifyMutation = useMutation({
    mutationFn: () =>
      api.verifyWithTrace({
        formula,
        name: "Interactive Verification",
        timeout_ms: 60000,
      }),
    onSuccess: (data) => {
      setTrace(data);
      setCurrentStep(0);
    },
  });

  // Fetch session state
  const { data: sessionState, refetch: refetchState } = useQuery({
    queryKey: ["debuggerSession", sessionId],
    queryFn: () => api.getDebuggerSessionState(sessionId!),
    enabled: !!sessionId,
  });

  const handleAddVariable = useCallback(
    (name: string, type: string) => {
      executeAction.mutate(
        { action: "add_variable", data: { name, type } },
        { onSuccess: () => refetchState() }
      );
    },
    [executeAction, refetchState]
  );

  const handleAddConstraint = useCallback(
    (constraint: string, description: string) => {
      executeAction.mutate(
        { action: "add_constraint", data: { constraint, description } },
        { onSuccess: () => refetchState() }
      );
    },
    [executeAction, refetchState]
  );

  const handleCheck = useCallback(() => {
    executeAction.mutate(
      { action: "check" },
      { onSuccess: () => refetchState() }
    );
  }, [executeAction, refetchState]);

  const handleReset = useCallback(() => {
    executeAction.mutate({ action: "reset" }, { onSuccess: () => refetchState() });
  }, [executeAction, refetchState]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Verification Debugger
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Step through Z3 verification and visualize constraints
          </p>
        </div>
        <div className="flex items-center gap-2">
          {!sessionId ? (
            <Button
              onClick={() => createSession.mutate()}
              loading={createSession.isPending}
            >
              <Plus className="h-4 w-4 mr-2" />
              New Session
            </Button>
          ) : (
            <Badge variant="success">Session Active</Badge>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel: Code/Formula */}
        <div className="lg:col-span-2 space-y-4">
          {/* Formula Input */}
          <Card padding="none">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Code className="h-5 w-5" />
                SMT-LIB Formula
              </CardTitle>
              <Button
                onClick={() => verifyMutation.mutate()}
                loading={verifyMutation.isPending}
                disabled={!formula.trim()}
              >
                <Play className="h-4 w-4 mr-2" />
                Verify
              </Button>
            </CardHeader>
            <CardContent>
              <textarea
                value={formula}
                onChange={(e) => setFormula(e.target.value)}
                className="w-full h-48 font-mono text-sm bg-gray-900 text-green-400 p-4 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="Enter SMT-LIB formula..."
              />
            </CardContent>
          </Card>

          {/* Verification Trace */}
          {trace && (
            <Card padding="none">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <ListChecks className="h-5 w-5" />
                  Verification Steps
                </CardTitle>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                    disabled={currentStep === 0}
                  >
                    <SkipForward className="h-4 w-4 rotate-180" />
                  </Button>
                  <span className="text-sm text-gray-500">
                    Step {currentStep + 1} of {trace.steps.length}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      setCurrentStep(
                        Math.min(trace.steps.length - 1, currentStep + 1)
                      )
                    }
                    disabled={currentStep === trace.steps.length - 1}
                  >
                    <SkipForward className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {trace.steps.map((step, index) => (
                  <StepRow
                    key={step.id}
                    step={step}
                    isActive={index === currentStep}
                    isPast={index < currentStep}
                    onClick={() => setCurrentStep(index)}
                  />
                ))}
              </div>
            </Card>
          )}

          {/* Result */}
          {trace && (
            <Card>
              <CardTitle className="mb-4">Result</CardTitle>
              <ResultDisplay trace={trace} />
            </Card>
          )}
        </div>

        {/* Right Panel: Variables & Constraints */}
        <div className="space-y-4">
          {/* Variables */}
          <Card padding="none">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Variable className="h-5 w-5" />
                Variables
              </CardTitle>
            </CardHeader>
            <CardContent>
              {sessionId ? (
                <InteractiveVariables
                  variables={sessionState?.state?.variables || {}}
                  onAdd={handleAddVariable}
                />
              ) : trace ? (
                <StaticVariables variables={trace.variables} />
              ) : (
                <p className="text-sm text-gray-500 text-center py-4">
                  Start a session or run verification
                </p>
              )}
            </CardContent>
          </Card>

          {/* Constraints */}
          <Card padding="none">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Constraints
              </CardTitle>
            </CardHeader>
            <CardContent>
              {sessionId ? (
                <InteractiveConstraints
                  constraints={sessionState?.state?.constraints || []}
                  onAdd={handleAddConstraint}
                  onCheck={handleCheck}
                  onReset={handleReset}
                />
              ) : trace ? (
                <StaticConstraints constraints={trace.constraints} />
              ) : (
                <p className="text-sm text-gray-500 text-center py-4">
                  Start a session or run verification
                </p>
              )}
            </CardContent>
          </Card>

          {/* Counterexample */}
          {trace?.counterexample && (
            <Card>
              <CardTitle className="flex items-center gap-2 mb-4">
                <AlertCircle className="h-5 w-5 text-red-500" />
                Counterexample
              </CardTitle>
              <div className="space-y-2">
                {Object.entries(trace.counterexample).map(([key, value]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between p-2 bg-red-50 dark:bg-red-900/20 rounded"
                  >
                    <span className="font-mono text-sm">{key}</span>
                    <span className="font-mono text-sm text-red-600 dark:text-red-400">
                      {String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function StepRow({
  step,
  isActive,
  isPast,
  onClick,
}: {
  step: VerificationStep;
  isActive: boolean;
  isPast: boolean;
  onClick: () => void;
}) {
  const statusIcon =
    step.status === "completed" ? (
      <CheckCircle className="h-4 w-4 text-green-500" />
    ) : step.status === "failed" ? (
      <XCircle className="h-4 w-4 text-red-500" />
    ) : (
      <AlertCircle className="h-4 w-4 text-yellow-500" />
    );

  return (
    <div
      onClick={onClick}
      className={`p-4 cursor-pointer transition ${
        isActive
          ? "bg-primary-50 dark:bg-primary-900/20 border-l-4 border-primary-500"
          : isPast
          ? "bg-gray-50 dark:bg-gray-800/50"
          : "hover:bg-gray-50 dark:hover:bg-gray-800"
      }`}
    >
      <div className="flex items-center gap-3">
        {statusIcon}
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <Badge>{step.step_type}</Badge>
            <span className="text-sm text-gray-500">
              {step.duration_ms.toFixed(1)}ms
            </span>
          </div>
          <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
            {step.description}
          </p>
        </div>
        <ChevronRight
          className={`h-5 w-5 text-gray-400 transition ${
            isActive ? "rotate-90" : ""
          }`}
        />
      </div>
      {isActive && step.formula && (
        <pre className="mt-3 p-3 bg-gray-900 text-green-400 rounded text-xs overflow-auto">
          {step.formula}
        </pre>
      )}
    </div>
  );
}

function ResultDisplay({ trace }: { trace: VerificationTrace }) {
  const result = trace.result;
  const explanation = trace.explanation as any;

  return (
    <div className="space-y-4">
      <div
        className={`p-4 rounded-lg ${
          result === "unsat"
            ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
            : result === "sat"
            ? "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
            : "bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800"
        }`}
      >
        <div className="flex items-center gap-3">
          {result === "unsat" ? (
            <CheckCircle className="h-6 w-6 text-green-500" />
          ) : result === "sat" ? (
            <XCircle className="h-6 w-6 text-red-500" />
          ) : (
            <AlertCircle className="h-6 w-6 text-yellow-500" />
          )}
          <div>
            <p className="font-medium text-gray-900 dark:text-white">
              {explanation?.summary || `Result: ${result}`}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {explanation?.meaning || ""}
            </p>
          </div>
        </div>
      </div>

      {explanation?.recommendations?.length > 0 && (
        <div>
          <p className="font-medium text-gray-900 dark:text-white mb-2">
            Recommendations
          </p>
          <ul className="list-disc list-inside space-y-1 text-sm text-gray-600 dark:text-gray-400">
            {explanation.recommendations.map((rec: string, i: number) => (
              <li key={i}>{rec}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="text-sm text-gray-500">
        Total duration: {trace.total_duration_ms.toFixed(2)}ms
      </div>
    </div>
  );
}

function InteractiveVariables({
  variables,
  onAdd,
}: {
  variables: Record<string, string>;
  onAdd: (name: string, type: string) => void;
}) {
  const [name, setName] = useState("");
  const [type, setType] = useState("int");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) {
      onAdd(name, type);
      setName("");
    }
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Variable name"
          className="flex-1"
        />
        <Select value={type} onChange={(e) => setType(e.target.value)}>
          <option value="int">Int</option>
          <option value="bool">Bool</option>
          <option value="real">Real</option>
        </Select>
        <Button type="submit" size="sm">
          <Plus className="h-4 w-4" />
        </Button>
      </form>
      <div className="space-y-2">
        {Object.entries(variables).map(([k, v]) => (
          <div
            key={k}
            className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded"
          >
            <span className="font-mono text-sm">{k}</span>
            <Badge>{v}</Badge>
          </div>
        ))}
        {Object.keys(variables).length === 0 && (
          <p className="text-sm text-gray-500 text-center py-2">
            No variables defined
          </p>
        )}
      </div>
    </div>
  );
}

function StaticVariables({ variables }: { variables: Record<string, string> }) {
  return (
    <div className="space-y-2">
      {Object.entries(variables).map(([k, v]) => (
        <div
          key={k}
          className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded"
        >
          <span className="font-mono text-sm">{k}</span>
          <Badge>{v}</Badge>
        </div>
      ))}
      {Object.keys(variables).length === 0 && (
        <p className="text-sm text-gray-500 text-center py-2">
          No variables in trace
        </p>
      )}
    </div>
  );
}

function InteractiveConstraints({
  constraints,
  onAdd,
  onCheck,
  onReset,
}: {
  constraints: Array<{ formula: string; description: string }>;
  onAdd: (constraint: string, description: string) => void;
  onCheck: () => void;
  onReset: () => void;
}) {
  const [constraint, setConstraint] = useState("");
  const [description, setDescription] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (constraint.trim()) {
      onAdd(constraint, description);
      setConstraint("");
      setDescription("");
    }
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="space-y-2">
        <Input
          value={constraint}
          onChange={(e) => setConstraint(e.target.value)}
          placeholder="(> x 0)"
          className="font-mono"
        />
        <Input
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Description (optional)"
        />
        <div className="flex gap-2">
          <Button type="submit" size="sm" className="flex-1">
            <Plus className="h-4 w-4 mr-1" />
            Add
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={onCheck}
          >
            Check
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onReset}
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </form>
      <div className="space-y-2 max-h-64 overflow-auto">
        {constraints.map((c, i) => (
          <div key={i} className="p-2 bg-gray-50 dark:bg-gray-800 rounded">
            <code className="text-xs text-gray-700 dark:text-gray-300 block">
              {c.formula}
            </code>
            {c.description && (
              <p className="text-xs text-gray-500 mt-1">{c.description}</p>
            )}
          </div>
        ))}
        {constraints.length === 0 && (
          <p className="text-sm text-gray-500 text-center py-2">
            No constraints added
          </p>
        )}
      </div>
    </div>
  );
}

function StaticConstraints({ constraints }: { constraints: string[] }) {
  return (
    <div className="space-y-2 max-h-64 overflow-auto">
      {constraints.map((c, i) => (
        <div key={i} className="p-2 bg-gray-50 dark:bg-gray-800 rounded">
          <code className="text-xs text-gray-700 dark:text-gray-300 break-all">
            {c}
          </code>
        </div>
      ))}
      {constraints.length === 0 && (
        <p className="text-sm text-gray-500 text-center py-2">
          No constraints in trace
        </p>
      )}
    </div>
  );
}

const DEFAULT_FORMULA = `; Example: Check if division is safe
(declare-const x Int)
(declare-const y Int)

; Preconditions
(assert (>= x 0))
(assert (>= y 0))

; Check: Can y ever be zero?
(assert (= y 0))

; If SAT, we found values where y = 0 (unsafe!)
; If UNSAT, y can never be 0 (safe!)
(check-sat)
(get-model)`;
