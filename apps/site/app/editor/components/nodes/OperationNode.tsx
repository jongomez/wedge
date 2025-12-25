"use client";

import type { NodeProps } from "@xyflow/react";
import { BaseNode, type EditorNodeData } from "./BaseNode";

export function OperationNode(props: NodeProps) {
  return <BaseNode {...props} data={props.data as EditorNodeData} />;
}
