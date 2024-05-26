import { NNShaders } from "@/lib/nnshaders/NNShaders";
import { opNodeHasMissingData } from "@/lib/nnshaders/helpers";
import { NNShadersOptions, OpNodeWithWebGLData } from "@/lib/nnshaders/types";
import { Model } from "@/lib/types";
import { FC, useState } from "react";
import { OpNodeSidebar } from "./OpNodeSidebar";


type ViewOpNodeProps = {
  opNode: OpNodeWithWebGLData;
  onClick: () => void;
  isActive: boolean;
}

const getStateCSSClass = (opNode: OpNodeWithWebGLData) => {
  if (opNode.type === "NotSupported") {
    return "not-supported";
  } else if (opNodeHasMissingData(opNode)) {
    return "missing-data";
  } else {
    return "";
  }
}

const ViewOpNode: FC<ViewOpNodeProps> = ({ opNode, isActive, onClick }) => {
  const stateCSSClass = getStateCSSClass(opNode);
  const activeCSSClass = isActive ? "active-node" : "";
  const outputOriginalShape = opNode.output?.originalShape.join(", ");
  const outputTextureShape = opNode.output?.shape.join(", ");

  return (
    <div className={"card model-visualizer-node " + stateCSSClass + " " + activeCSSClass} onClick={onClick}>
      <h2>
        {opNode.node.op}
      </h2>

      <p>
        {opNode.node.name}
      </p>

      {!!outputOriginalShape &&
        <p>
          Output original shape: ({outputOriginalShape})
        </p>
      }

      {!!outputTextureShape &&
        <p>
          Output texture shape: ({outputTextureShape})
        </p>
      }
    </div>);
}

type ModelVisualizeProps = {
  model: Model;
  nnShadersOptions?: NNShadersOptions;
}

export const ModelVisualize: FC<ModelVisualizeProps> = ({
  model,
  nnShadersOptions }) => {
  const [selectedOpNode, setSelectedOpNode] = useState<OpNodeWithWebGLData | null>(null);

  const handleNodeClick = (opNode: OpNodeWithWebGLData) => {
    if (opNode === selectedOpNode) {
      setSelectedOpNode(null);
    } else {
      setSelectedOpNode(opNode);
    }
  };

  const handleClose = () => {
    setSelectedOpNode(null);
  };

  const opNodesIterableIterator = (model as NNShaders).opNodeMap.values();
  const opNodes = Array.from(opNodesIterableIterator);
  const hasSidebarClassName = selectedOpNode ? "model-visualizer-container-with-sidebar" : "";

  return (
    <>
      <div className={"model-visualizer-container " + hasSidebarClassName}>
        {opNodes.map((opNode) => {
          return <ViewOpNode
            opNode={opNode}
            key={opNode.node.name}
            onClick={() => handleNodeClick(opNode)}
            isActive={selectedOpNode === opNode}
          />
        })}

      </div >

      <OpNodeSidebar opNode={selectedOpNode} onClose={handleClose} />
    </>
  );

}