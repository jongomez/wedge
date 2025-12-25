import { OperationNode } from "./OperationNode";

export { OperationNode };
export type { EditorNodeData } from "./BaseNode";

// Map all operations to the same node component
export const nodeTypes = {
  operation: OperationNode,
};
