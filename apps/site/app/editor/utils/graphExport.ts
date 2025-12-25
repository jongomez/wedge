import type { EditorNode, EditorEdge } from "../hooks/useEditorState";
import { OPERATIONS } from "../constants";

// Simplified Graph and GraphNode types matching Wedge's structure
interface ExportedGraphNode {
  name: string;
  op: {
    name: string;
    category?: string;
  };
  inputs: string[];
  inputNames: string[];
  params: Record<string, { source: "static"; type: string; value: unknown }>;
  children: string[];
}

interface ExportedGraph {
  nodes: Record<string, ExportedGraphNode>;
  placeholders: string[];
  inputs: string[];
  outputs: string[];
  weights: string[];
}

export function exportToWedgeGraph(
  nodes: EditorNode[],
  edges: EditorEdge[]
): ExportedGraph {
  const graph: ExportedGraph = {
    nodes: {},
    placeholders: [],
    inputs: [],
    outputs: [],
    weights: [],
  };

  // Build adjacency maps
  const incomingEdges = new Map<string, { source: string; sourceHandle: string; targetHandle: string }[]>();
  const outgoingEdges = new Map<string, string[]>();

  for (const edge of edges) {
    // Incoming
    if (!incomingEdges.has(edge.target)) {
      incomingEdges.set(edge.target, []);
    }
    incomingEdges.get(edge.target)!.push({
      source: edge.source,
      sourceHandle: edge.sourceHandle || "output",
      targetHandle: edge.targetHandle || "input",
    });

    // Outgoing (children)
    if (!outgoingEdges.has(edge.source)) {
      outgoingEdges.set(edge.source, []);
    }
    outgoingEdges.get(edge.source)!.push(edge.target);
  }

  // Convert each node
  for (const node of nodes) {
    const config = OPERATIONS[node.data.opName];
    if (!config) continue;

    const incoming = incomingEdges.get(node.id) || [];
    const children = outgoingEdges.get(node.id) || [];

    // Map incoming edges to input names
    const inputNames: string[] = [];
    for (const inc of incoming) {
      inputNames.push(inc.source);
    }

    // Convert params
    const params: Record<string, { source: "static"; type: string; value: unknown }> = {};
    for (const [key, value] of Object.entries(node.data.params)) {
      const type = Array.isArray(value)
        ? "number[]"
        : typeof value === "boolean"
          ? "bool"
          : typeof value === "number"
            ? "number"
            : "string";
      params[key] = { source: "static", type, value };
    }

    const exportedNode: ExportedGraphNode = {
      name: node.id,
      op: {
        name: node.data.opName,
        category: config.category,
      },
      inputs: inputNames,
      inputNames,
      params,
      children,
    };

    graph.nodes[node.id] = exportedNode;

    // Categorize nodes
    if (node.data.opName === "Placeholder") {
      graph.placeholders.push(node.id);
      graph.inputs.push(node.id);
    } else if (node.data.opName === "Output") {
      graph.outputs.push(node.id);
    }
  }

  return graph;
}

export function downloadWedgeGraph(nodes: EditorNode[], edges: EditorEdge[]) {
  const graph = exportToWedgeGraph(nodes, edges);
  const blob = new Blob([JSON.stringify(graph, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `wedge-graph-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
