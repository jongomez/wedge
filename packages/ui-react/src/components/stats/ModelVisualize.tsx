import { Model } from "@/lib/types";
import { Wedge } from "@/lib/wedge/Wedge";
import { WebGLOpNode, WedgeOptions } from "@/lib/wedge/types";
import { FC } from "react";


type ViewOpNodeProps = {
  opNode: WebGLOpNode;
  onClick: () => void;
  isActive: boolean;
}

type ModelStatsProps = {
  model: Model;
  nnShadersOptions?: WedgeOptions;
}

export const ModelStats: FC<ModelStatsProps> = ({
  model,
  nnShadersOptions }) => {
  const opNodesIterableIterator = (model as Wedge).opNodeMap.values();
  const opNodes = Array.from(opNodesIterableIterator);
  const numberOfOpNodes = opNodes.length

  return (
    <>
      <div className={"model-stats"}>
        Number of Op Nodes: {numberOfOpNodes}
      </div >
    </>
  );

}