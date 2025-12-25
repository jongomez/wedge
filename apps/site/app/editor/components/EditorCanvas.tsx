"use client";

import { useCallback, useRef, type DragEvent } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  type ReactFlowInstance,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { nodeTypes, type EditorNodeData } from "./nodes";
import { COLORS, DIMS } from "../constants";
import type { useEditorState } from "../hooks/useEditorState";
import type { useValidation } from "../hooks/useValidation";

type EditorNode = Node<EditorNodeData>;
type EditorEdge = Edge;

interface EditorCanvasProps {
  nodes: EditorNode[];
  edges: EditorEdge[];
  onNodesChange: ReturnType<typeof useEditorState>["onNodesChange"];
  onEdgesChange: ReturnType<typeof useEditorState>["onEdgesChange"];
  onConnect: ReturnType<typeof useEditorState>["onConnect"];
  addNode: ReturnType<typeof useEditorState>["addNode"];
  setSelectedNodeId: (id: string | null) => void;
  isValidConnection: ReturnType<typeof useValidation>["isValidConnection"];
}

export function EditorCanvas({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
  addNode,
  setSelectedNodeId,
  isValidConnection,
}: EditorCanvasProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const reactFlowInstance = useRef<ReactFlowInstance<EditorNode, EditorEdge> | null>(null);

  const onInit = useCallback((instance: ReactFlowInstance<EditorNode, EditorEdge>) => {
    reactFlowInstance.current = instance;
  }, []);

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();

      const opName = event.dataTransfer.getData("application/reactflow");
      if (!opName || !reactFlowInstance.current || !reactFlowWrapper.current) return;

      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = reactFlowInstance.current.screenToFlowPosition({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });

      // Snap to grid
      position.x = Math.round(position.x / DIMS.gridSize) * DIMS.gridSize;
      position.y = Math.round(position.y / DIMS.gridSize) * DIMS.gridSize;

      addNode(opName, position);
    },
    [addNode]
  );

  const handleNodeClick = useCallback(
    (_: unknown, node: EditorNode) => {
      setSelectedNodeId(node.id);
    },
    [setSelectedNodeId]
  );

  const handlePaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, [setSelectedNodeId]);

  return (
    <div ref={reactFlowWrapper} style={{ width: "100%", height: "100%" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={onInit}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeClick={handleNodeClick}
        onPaneClick={handlePaneClick}
        isValidConnection={isValidConnection}
        nodeTypes={nodeTypes}
        snapToGrid
        snapGrid={[DIMS.gridSize, DIMS.gridSize]}
        defaultViewport={{ x: 100, y: 100, zoom: 1 }}
        minZoom={0.25}
        maxZoom={2}
        fitView={false}
        panOnScroll
        panOnScrollSpeed={2}
        panOnDrag={[1]}
        zoomOnScroll
        zoomOnPinch
        preventScrolling={false}
        zoomActivationKeyCode="Control"
        zoomOnDoubleClick={false}
        selectionOnDrag={false}
        proOptions={{ hideAttribution: true }}
        style={{ background: COLORS.bg }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={DIMS.gridSize}
          size={1}
          color="rgba(99,102,241,0.15)"
        />
        <Controls
          showZoom
          showFitView
          showInteractive={false}
          style={{
            background: COLORS.bgSecondary,
            border: `1px solid ${COLORS.border}`,
            borderRadius: DIMS.borderRadius,
          }}
        />
        <MiniMap
          nodeColor={(node) => {
            const data = node.data as EditorNodeData;
            return data.opName === "Placeholder"
              ? COLORS.catInput
              : data.opName === "Output"
                ? COLORS.catOutput
                : COLORS.accent;
          }}
          maskColor="rgba(0,0,0,0.8)"
          style={{
            background: COLORS.bgSecondary,
            border: `1px solid ${COLORS.border}`,
            borderRadius: DIMS.borderRadius,
          }}
        />
      </ReactFlow>
    </div>
  );
}
