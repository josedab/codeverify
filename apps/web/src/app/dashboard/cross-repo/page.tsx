"use client";

import { useState, useCallback, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  MarkerType,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  GitBranch,
  Package,
  AlertTriangle,
  ArrowRight,
  Layers,
  Search,
  RefreshCw,
} from "lucide-react";
import api, { CrossRepoGraph } from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  Input,
  Badge,
  SlideOver,
} from "@/components/ui";

export default function CrossRepoPage() {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  // Fetch graph data
  const {
    data: graphData,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["crossRepoGraph"],
    queryFn: () => api.getCrossRepoGraph(),
  });

  // Transform data for React Flow
  const { nodes, edges, stats } = useMemo((): {
    nodes: Node[];
    edges: Edge[];
    stats: CrossRepoGraph["stats"] | null;
  } => {
    if (!graphData) {
      return { nodes: [] as Node[], edges: [] as Edge[], stats: null };
    }

    const repos = Object.entries(graphData.repositories);
    const nodePositions: Record<string, { x: number; y: number }> = {};

    // Simple grid layout
    repos.forEach(([name], index) => {
      const row = Math.floor(index / 4);
      const col = index % 4;
      nodePositions[name] = {
        x: col * 280 + 50,
        y: row * 180 + 50,
      };
    });

    const flowNodes: Node[] = repos.map(([name, data]: [string, any]) => ({
      id: name,
      type: "custom",
      position: nodePositions[name],
      data: {
        label: name,
        language: data.language || "unknown",
        repoType: data.repo_type || "library",
        exportedContracts: data.exported_contracts?.length || 0,
        importedContracts: data.imported_contracts?.length || 0,
      },
    }));

    const flowEdges: Edge[] = graphData.dependencies.map((dep, index) => ({
      id: `edge-${index}`,
      source: dep.source,
      target: dep.target,
      label: dep.dependency_type,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: "#6366F1",
      },
      style: { stroke: "#6366F1", strokeWidth: 2 },
      animated: dep.dependency_type === "direct",
    }));

    return {
      nodes: flowNodes,
      edges: flowEdges,
      stats: graphData.stats,
    };
  }, [graphData]);

  // Filter nodes by search
  const filteredNodes = useMemo(() => {
    if (!searchTerm) return nodes;
    return nodes.filter((node) =>
      node.id.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [nodes, searchTerm]);

  const selectedNodeData = useMemo((): Record<string, any> | null => {
    if (!selectedNode || !graphData) return null;
    return graphData.repositories[selectedNode] as Record<string, any>;
  }, [selectedNode, graphData]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Cross-Repository Analysis
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Visualize dependencies and contracts between repositories
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Input
            placeholder="Search repositories..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-64"
          />
          <Button variant="outline" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card padding="md">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                <GitBranch className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Repositories
                </p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {stats.repository_count}
                </p>
              </div>
            </div>
          </Card>
          <Card padding="md">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 dark:bg-purple-900 rounded-lg">
                <ArrowRight className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Dependencies
                </p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {stats.dependency_count}
                </p>
              </div>
            </div>
          </Card>
          <Card padding="md">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 dark:bg-green-900 rounded-lg">
                <Package className="h-5 w-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Contracts
                </p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {stats.contract_count}
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Dependency Graph */}
      <Card padding="none" className="h-[600px]">
        {isLoading ? (
          <div className="h-full flex items-center justify-center">
            <div className="animate-spin h-8 w-8 border-2 border-primary-500 border-t-transparent rounded-full" />
          </div>
        ) : nodes.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <Layers className="h-12 w-12 mb-4" />
            <p className="text-lg font-medium">No repositories registered</p>
            <p className="text-sm mt-1">
              Register repositories to see the dependency graph
            </p>
          </div>
        ) : (
          <ReactFlow
            nodes={filteredNodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodeClick={(_, node) => setSelectedNode(node.id)}
            fitView
            minZoom={0.2}
            maxZoom={2}
          >
            <Controls />
            <MiniMap
              nodeStrokeColor="#6366F1"
              nodeColor="#EEF2FF"
              nodeBorderRadius={8}
            />
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
          </ReactFlow>
        )}
      </Card>

      {/* Contracts List */}
      {graphData && Object.keys(graphData.contracts).length > 0 && (
        <Card padding="none">
          <CardHeader>
            <CardTitle>Registered Contracts</CardTitle>
          </CardHeader>
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {Object.entries(graphData.contracts).map(
              ([id, contract]: [string, any]) => (
                <div
                  key={id}
                  className="p-4 hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">
                        {contract.name}
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {contract.owner_repo}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge>{contract.contract_type}</Badge>
                      <Badge
                        variant={
                          contract.stability === "stable"
                            ? "success"
                            : contract.stability === "deprecated"
                            ? "danger"
                            : "warning"
                        }
                      >
                        {contract.stability}
                      </Badge>
                      <span className="text-sm text-gray-500">
                        v{contract.version}
                      </span>
                    </div>
                  </div>
                  {contract.consumers?.length > 0 && (
                    <div className="mt-2 text-sm text-gray-500">
                      <span className="font-medium">Consumers:</span>{" "}
                      {contract.consumers.join(", ")}
                    </div>
                  )}
                </div>
              )
            )}
          </div>
        </Card>
      )}

      {/* Repository Detail Slide-over */}
      <SlideOver
        open={!!selectedNode}
        onClose={() => setSelectedNode(null)}
        title={selectedNode || ""}
        width="lg"
      >
        {selectedNodeData && (
          <RepositoryDetail
            name={selectedNode!}
            data={selectedNodeData}
            graphData={graphData!}
          />
        )}
      </SlideOver>
    </div>
  );
}

// Custom node component
function RepositoryNode({ data }: { data: any }) {
  const languageColors: Record<string, string> = {
    python: "bg-yellow-100 text-yellow-800",
    typescript: "bg-blue-100 text-blue-800",
    javascript: "bg-yellow-100 text-yellow-800",
    go: "bg-cyan-100 text-cyan-800",
    rust: "bg-orange-100 text-orange-800",
    unknown: "bg-gray-100 text-gray-800",
  };

  return (
    <div className="bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-700 rounded-lg p-3 shadow-md min-w-[200px]">
      <div className="flex items-center gap-2 mb-2">
        <GitBranch className="h-4 w-4 text-gray-500" />
        <span className="font-medium text-sm text-gray-900 dark:text-white truncate">
          {data.label}
        </span>
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <Badge
          className={languageColors[data.language] || languageColors.unknown}
        >
          {data.language}
        </Badge>
        <span className="text-xs text-gray-500">{data.repoType}</span>
      </div>
      <div className="mt-2 flex items-center gap-3 text-xs text-gray-500">
        <span>↑ {data.exportedContracts}</span>
        <span>↓ {data.importedContracts}</span>
      </div>
    </div>
  );
}

const nodeTypes = {
  custom: RepositoryNode,
};

function RepositoryDetail({
  name,
  data,
  graphData,
}: {
  name: string;
  data: any;
  graphData: CrossRepoGraph;
}) {
  // Find dependencies
  const dependencies = graphData.dependencies.filter(
    (d) => d.source === name
  );
  const dependents = graphData.dependencies.filter((d) => d.target === name);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-sm font-medium text-gray-500">Language</label>
          <p className="mt-1 text-gray-900 dark:text-white">
            {data.language || "Unknown"}
          </p>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-500">Type</label>
          <p className="mt-1 text-gray-900 dark:text-white">
            {data.repo_type || "Unknown"}
          </p>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-500">
            Default Branch
          </label>
          <p className="mt-1 text-gray-900 dark:text-white">
            {data.default_branch || "main"}
          </p>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-500">URL</label>
          <p className="mt-1 text-gray-900 dark:text-white truncate">
            {data.url || "-"}
          </p>
        </div>
      </div>

      {/* Dependencies */}
      <div>
        <h4 className="font-medium text-gray-900 dark:text-white mb-2">
          Dependencies ({dependencies.length})
        </h4>
        {dependencies.length > 0 ? (
          <div className="space-y-2">
            {dependencies.map((dep, i) => (
              <div
                key={i}
                className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded"
              >
                <span className="text-sm">{dep.target}</span>
                <Badge>{dep.dependency_type}</Badge>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">No dependencies</p>
        )}
      </div>

      {/* Dependents */}
      <div>
        <h4 className="font-medium text-gray-900 dark:text-white mb-2">
          Dependents ({dependents.length})
        </h4>
        {dependents.length > 0 ? (
          <div className="space-y-2">
            {dependents.map((dep, i) => (
              <div
                key={i}
                className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded"
              >
                <span className="text-sm">{dep.source}</span>
                <Badge>{dep.dependency_type}</Badge>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">No dependents</p>
        )}
      </div>

      {/* Exported Contracts */}
      <div>
        <h4 className="font-medium text-gray-900 dark:text-white mb-2">
          Exported Contracts ({data.exported_contracts?.length || 0})
        </h4>
        {data.exported_contracts?.length > 0 ? (
          <div className="space-y-2">
            {data.exported_contracts.map((contractId: string) => {
              const contract = graphData.contracts[contractId] as Record<string, any>;
              return contract ? (
                <div
                  key={contractId}
                  className="p-2 bg-gray-50 dark:bg-gray-800 rounded"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{contract.name}</span>
                    <Badge>{contract.contract_type}</Badge>
                  </div>
                </div>
              ) : null;
            })}
          </div>
        ) : (
          <p className="text-sm text-gray-500">No exported contracts</p>
        )}
      </div>
    </div>
  );
}
