"use client";

import type { CSSProperties } from "react";
import { ReactFlowProvider } from "@xyflow/react";

import { EditorHeader } from "./components/EditorHeader";
import { EditorCanvas } from "./components/EditorCanvas";
import { Toolbar } from "./components/Toolbar";
import { PropertiesPanel } from "./components/PropertiesPanel";

import { useEditorState } from "./hooks/useEditorState";
import { useValidation } from "./hooks/useValidation";
import { useFileIO } from "./hooks/useFileIO";
import { useModelLoader, AVAILABLE_MODELS } from "./hooks/useModelLoader";
import { downloadWedgeGraph } from "./utils/graphExport";

import { COLORS } from "./constants";

export default function EditorPage() {
  const {
    nodes,
    edges,
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
  } = useEditorState();

  const { isValidConnection } = useValidation(nodes, edges);
  const { saveToFile, loadFromFile } = useFileIO(getGraph, setGraph);
  const { loadModel, loading: loadingModel, loadedModelName } = useModelLoader(setGraph);

  const handleExport = () => {
    downloadWedgeGraph(nodes, edges);
  };

  return (
    <ReactFlowProvider>
      <div style={styles.container}>
        <EditorHeader
          onSave={saveToFile}
          onLoad={loadFromFile}
          onExport={handleExport}
          onLoadModel={loadModel}
          availableModels={AVAILABLE_MODELS}
          loadingModel={loadingModel}
          loadedModelName={loadedModelName}
        />
        <div style={styles.body}>
          <Toolbar onClear={clearGraph} />
          <div style={styles.canvas}>
            <EditorCanvas
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              addNode={addNode}
              setSelectedNodeId={setSelectedNodeId}
              isValidConnection={isValidConnection}
            />
          </div>
          <PropertiesPanel
            node={selectedNode}
            onUpdateParams={updateNodeParams}
            onDelete={deleteNode}
          />
        </div>
      </div>
    </ReactFlowProvider>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    width: "100vw",
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    background: COLORS.bg,
    overflow: "hidden",
  },
  body: {
    flex: 1,
    display: "flex",
    overflow: "hidden",
  },
  canvas: {
    flex: 1,
    position: "relative",
  },
};
