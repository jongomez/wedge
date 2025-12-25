"use client";

import { useState, useCallback } from "react";
import {
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Node,
  type Edge,
} from "@xyflow/react";
import { OPERATIONS } from "../constants";
import type { EditorNodeData } from "../components/nodes";

export type EditorNode = Node<EditorNodeData>;
export type EditorEdge = Edge;

export function useEditorState() {
  const [nodes, setNodes, onNodesChange] = useNodesState<EditorNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<EditorEdge>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) =>
        addEdge(
          {
            ...connection,
            type: "default",
          },
          eds
        )
      );
    },
    [setEdges]
  );

  const addNode = useCallback(
    (opName: string, position: { x: number; y: number }) => {
      const config = OPERATIONS[opName];
      if (!config) return null;

      const id = `${opName}-${Date.now()}`;
      const newNode: EditorNode = {
        id,
        type: "operation",
        position,
        data: {
          opName,
          label: config.displayName,
          params: { ...config.defaultParams },
        },
      };
      setNodes((nds) => [...nds, newNode]);
      return id;
    },
    [setNodes]
  );

  const deleteNode = useCallback(
    (nodeId: string) => {
      setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
      if (selectedNodeId === nodeId) setSelectedNodeId(null);
    },
    [setNodes, setEdges, selectedNodeId]
  );

  const updateNodeParams = useCallback(
    (nodeId: string, params: Record<string, unknown>) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, params: { ...n.data.params, ...params } } }
            : n
        )
      );
    },
    [setNodes]
  );

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) || null;

  // For file I/O
  const getGraph = useCallback(() => ({ nodes, edges }), [nodes, edges]);

  const setGraph = useCallback(
    (graph: { nodes: EditorNode[]; edges: EditorEdge[] }) => {
      setNodes(graph.nodes);
      setEdges(graph.edges);
      setSelectedNodeId(null);
    },
    [setNodes, setEdges]
  );

  const clearGraph = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setSelectedNodeId(null);
  }, [setNodes, setEdges]);

  return {
    nodes,
    edges,
    selectedNodeId,
    selectedNode,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    deleteNode,
    updateNodeParams,
    setSelectedNodeId,
    getGraph,
    setGraph,
    clearGraph,
  };
}
