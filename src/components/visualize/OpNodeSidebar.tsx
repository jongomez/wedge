import { OpNodeWithWebGLData } from "@/lib/nnshaders/types";
import { FC } from "react";

type OpNodeSidebarProps = {
  opNode: OpNodeWithWebGLData | null;
  onClose: () => void;
};

export const OpNodeSidebar: FC<OpNodeSidebarProps> = ({ opNode, onClose }) => {
  if (!opNode) return null;

  return (
    <div className="op-node-sidebar sidebar-dimensions-and-border">
      <button onClick={onClose}>Close</button>
      <h3>{opNode.node.op}</h3>
      <p>{opNode.node.name}</p>

      TODO: Include more node details here

    </div>
  );
};