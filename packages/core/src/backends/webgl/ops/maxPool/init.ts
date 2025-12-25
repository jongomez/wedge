import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { opNodeHasMissingData } from "../../helpers";
import { ModelType, NodeWebGLDataMap, WebGLOpNode, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { getMaxPoolOutput } from "./output";
import { maxPoolWebGLShader } from "./webGLShader";

export function initMaxPoolWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  modelType: ModelType,
  options: WedgeOptions
): WebGLOpNode {
  if (node.inputs.length !== 1) {
    throw new Error(`MaxPool node ${node.name} has ${node.inputs.length} inputs, expected 1`);
  }

  const input = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);
  const { output, params } = getMaxPoolOutput(gl, node, nodeWebGLDataMap, opNodeMap, modelType, options);

  const opNode: WebGLOpNode = {
    node,
    inputs: [input],
    output,
    type: "MaxPool",
    weights: [],
    opParams: params,
    fsSource: "",
  };

  if (!opNodeHasMissingData(opNode)) {
    opNode.fsSource = maxPoolWebGLShader(opNode);
  }

  return opNode;
}
