"use client";

import { useCallback } from "react";
import type { Connection, Edge, Node } from "@xyflow/react";
import { OPERATIONS } from "../constants";
import type { EditorNodeData } from "../components/nodes";

type EditorNode = Node<EditorNodeData>;

export function useValidation(nodes: EditorNode[], edges: Edge[]) {
  const isValidConnection = useCallback(
    (connection: Edge | Connection): boolean => {
      const { source, target, sourceHandle, targetHandle } = connection;

      // Rule 1: Cannot connect to self
      if (source === target) return false;

      // Rule 2: Must have valid handles
      if (!sourceHandle || !targetHandle) return false;

      const sourceNode = nodes.find((n) => n.id === source);
      const targetNode = nodes.find((n) => n.id === target);

      if (!sourceNode || !targetNode) return false;

      const sourceConfig = OPERATIONS[sourceNode.data.opName];
      const targetConfig = OPERATIONS[targetNode.data.opName];

      if (!sourceConfig || !targetConfig) return false;

      // Rule 3: Source handle must be an output
      const sourceOutput = sourceConfig.outputs.find((o) => o.id === sourceHandle);
      if (!sourceOutput) return false;

      // Rule 4: Target handle must be an input
      const targetInput = targetConfig.inputs.find((i) => i.id === targetHandle);
      if (!targetInput) return false;

      // Rule 5: Check if target input already has a connection
      const existingConnection = edges.find(
        (e) => e.target === target && e.targetHandle === targetHandle
      );
      if (existingConnection) return false;

      // Rule 6: Type compatibility - weight outputs can connect to weight inputs
      // but tensor outputs shouldn't connect to weight-only inputs
      if (sourceOutput.type === "tensor" && targetInput.type === "weight") {
        return false;
      }

      return true;
    },
    [nodes, edges]
  );

  return { isValidConnection };
}
