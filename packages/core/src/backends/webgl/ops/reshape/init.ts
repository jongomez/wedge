import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { opNodeHasMissingData } from "../../helpers";
import { NodeWebGLDataMap, ReshapeParams, WebGLOpNode, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { getNewShape, getReshapeOutput } from "./output";
import { reshapeWebGLShader } from "./webGLShader";

export function initReshapeWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  weightMap: NamedTensorsMap,
  options: WedgeOptions
): WebGLOpNode {
  // Reshape has either:
  // - GraphModel: 2 inputs (tensor + shape const)
  // - LayersModel: 1 input (tensor) + targetShape in attrParams
  if (node.inputs.length < 1) {
    throw new Error(`Reshape operation expects at least 1 input, got ${node.inputs.length}`);
  }

  const input = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);
  const newShape = getNewShape(node, weightMap);
  const output = getReshapeOutput(gl, node, nodeWebGLDataMap, opNodeMap, newShape, options);

  const opParams: ReshapeParams = {
    newShape,
  };

  const opNode: WebGLOpNode = {
    node,
    inputs: [input],
    output,
    opParams,
    weights: [],
    type: "Reshape",
    fsSource: "",
  };

  if (!opNodeHasMissingData(opNode)) {
    opNode.fsSource = reshapeWebGLShader(opNode);
  }

  return opNode;
}
