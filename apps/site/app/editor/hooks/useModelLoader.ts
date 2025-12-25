"use client";

import { useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import { OPERATIONS } from "../constants";
import type { EditorNode, EditorEdge } from "./useEditorState";

// Available models to load
export const AVAILABLE_MODELS = [
  { name: "BlazeFace Lite", path: "/models/blaze_lite/model.json" },
];

// Map TF op names to our operation names
const OP_NAME_MAP: Record<string, string> = {
  "Placeholder": "Placeholder",
  "Const": "Placeholder", // Treat const as input for visualization
  "Conv2D": "Conv2D",
  "_FusedConv2D": "Conv2D",
  "FusedConv2D": "Conv2D",
  "DepthwiseConv2D": "DepthwiseConv2D",
  "DepthwiseConv2dNative": "DepthwiseConv2D",
  "FusedDepthwiseConv2dNative": "DepthwiseConv2D",
  "AddV2": "AddV2",
  "Add": "AddV2",
  "Mul": "Mul",
  "Relu": "Relu",
  "ReLU": "Relu",
  "Relu6": "Relu6",
  "Pad": "Pad",
  "PadV2": "PadV2",
  "MirrorPad": "MirrorPad",
  "ResizeBilinear": "ResizeBilinear",
  "Reshape": "Reshape",
  "Identity": "Relu", // Passthrough, treat as activation
};

interface ModelNode {
  name: string;
  op: string;
  inputs: string[];
  attrs?: Record<string, unknown>;
}

// Auto-layout nodes in layers
function layoutNodes(
  nodes: ModelNode[],
  edges: { source: string; target: string }[]
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();

  // Build dependency graph
  const inDegree = new Map<string, number>();
  const outEdges = new Map<string, string[]>();

  for (const node of nodes) {
    inDegree.set(node.name, 0);
    outEdges.set(node.name, []);
  }

  for (const edge of edges) {
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
    outEdges.get(edge.source)?.push(edge.target);
  }

  // Topological sort with layers
  const layers: string[][] = [];
  const remaining = new Set(nodes.map(n => n.name));

  while (remaining.size > 0) {
    const layer: string[] = [];

    for (const name of remaining) {
      if ((inDegree.get(name) || 0) === 0) {
        layer.push(name);
      }
    }

    if (layer.length === 0) {
      // Cycle detected, just add remaining
      layer.push(...remaining);
      remaining.clear();
    } else {
      for (const name of layer) {
        remaining.delete(name);
        for (const target of outEdges.get(name) || []) {
          inDegree.set(target, (inDegree.get(target) || 0) - 1);
        }
      }
    }

    layers.push(layer);
  }

  // Position nodes
  const nodeWidth = 180;
  const nodeHeight = 80;
  const layerGap = 100;
  const nodeGap = 40;

  for (let layerIdx = 0; layerIdx < layers.length; layerIdx++) {
    const layer = layers[layerIdx];
    const layerHeight = layer.length * nodeHeight + (layer.length - 1) * nodeGap;
    const startY = -layerHeight / 2;

    for (let nodeIdx = 0; nodeIdx < layer.length; nodeIdx++) {
      const name = layer[nodeIdx];
      positions.set(name, {
        x: layerIdx * (nodeWidth + layerGap) + 100,
        y: startY + nodeIdx * (nodeHeight + nodeGap) + 300,
      });
    }
  }

  return positions;
}

export function useModelLoader(
  setGraph: (graph: { nodes: EditorNode[]; edges: EditorEdge[] }) => void
) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadedModelName, setLoadedModelName] = useState<string | null>(null);

  const loadModel = useCallback(async (url: string, modelName: string) => {
    setLoading(true);
    setError(null);

    try {
      const graphModel = await tf.loadGraphModel(url);

      // Extract nodes from the model
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const modelArtifacts = (graphModel as any).artifacts;
      const modelTopology = modelArtifacts?.modelTopology;

      if (!modelTopology?.node) {
        throw new Error("Could not read model topology");
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const modelNodes: ModelNode[] = modelTopology.node.map((node: any) => ({
        name: node.name || "unnamed",
        op: node.op || "Unknown",
        inputs: (node.input || []).map((i: string) => i.split(":")[0]),
        attrs: node.attr,
      }));

      // Filter out Const nodes (weights) and unsupported ops for cleaner visualization
      const filteredNodes = modelNodes.filter(n => {
        // Skip const/weight nodes
        if (n.op === "Const") return false;
        // Only include nodes we have a mapping for
        return OP_NAME_MAP[n.op] !== undefined;
      });

      // Build edges from inputs
      const rawEdges: { source: string; target: string }[] = [];
      const nodeNames = new Set(filteredNodes.map(n => n.name));

      for (const node of filteredNodes) {
        for (const input of node.inputs) {
          // Only add edge if source exists in our filtered nodes
          if (nodeNames.has(input)) {
            rawEdges.push({ source: input, target: node.name });
          }
        }
      }

      // Layout nodes
      const positions = layoutNodes(filteredNodes, rawEdges);

      // Convert to editor nodes
      const editorNodes: EditorNode[] = filteredNodes.map((node) => {
        const mappedOp = OP_NAME_MAP[node.op] || "Relu";
        const config = OPERATIONS[mappedOp];
        const pos = positions.get(node.name) || { x: 100, y: 100 };

        return {
          id: node.name,
          type: "operation",
          position: pos,
          data: {
            opName: mappedOp,
            label: config?.displayName || node.op,
            params: config?.defaultParams ? { ...config.defaultParams } : {},
            originalOp: node.op, // Keep original op name for reference
          },
        };
      });

      // Convert to editor edges
      const editorEdges: EditorEdge[] = rawEdges.map((edge, idx) => {
        const sourceNode = filteredNodes.find(n => n.name === edge.source);
        const targetNode = filteredNodes.find(n => n.name === edge.target);

        const sourceOp = sourceNode ? OP_NAME_MAP[sourceNode.op] : "Relu";
        const targetOp = targetNode ? OP_NAME_MAP[targetNode.op] : "Relu";

        const sourceConfig = OPERATIONS[sourceOp];
        const targetConfig = OPERATIONS[targetOp];

        return {
          id: `e-${idx}-${edge.source}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          sourceHandle: sourceConfig?.outputs[0]?.id || "output",
          targetHandle: targetConfig?.inputs[0]?.id || "input",
        };
      });

      setGraph({ nodes: editorNodes, edges: editorEdges });
      setLoadedModelName(modelName);
    } catch (err) {
      console.error("Model load error:", err);
      setError(err instanceof Error ? err.message : "Failed to load model");
    } finally {
      setLoading(false);
    }
  }, [setGraph]);

  return { loadModel, loading, error, loadedModelName, AVAILABLE_MODELS };
}
