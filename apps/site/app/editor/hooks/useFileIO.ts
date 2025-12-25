"use client";

import { useCallback, useRef } from "react";
import type { EditorNode, EditorEdge } from "./useEditorState";

interface GraphData {
  nodes: EditorNode[];
  edges: EditorEdge[];
  version: string;
}

export function useFileIO(
  getGraph: () => { nodes: EditorNode[]; edges: EditorEdge[] },
  setGraph: (graph: { nodes: EditorNode[]; edges: EditorEdge[] }) => void
) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const saveToFile = useCallback(() => {
    const graph = getGraph();
    const data: GraphData = {
      ...graph,
      version: "1.0",
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `graph-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [getGraph]);

  const loadFromFile = useCallback(() => {
    if (!fileInputRef.current) {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".json";
      input.style.display = "none";
      document.body.appendChild(input);
      fileInputRef.current = input;

      input.addEventListener("change", (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
          try {
            const data = JSON.parse(event.target?.result as string) as GraphData;
            if (data.nodes && data.edges) {
              setGraph({ nodes: data.nodes, edges: data.edges });
            }
          } catch (err) {
            console.error("Failed to parse graph file:", err);
          }
        };
        reader.readAsText(file);

        // Reset input so same file can be loaded again
        input.value = "";
      });
    }

    fileInputRef.current.click();
  }, [setGraph]);

  return { saveToFile, loadFromFile };
}
