import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { opNodeHasMissingData } from "../../helpers";
import { ModelType, NodeWebGLDataMap, PadParams, WebGLOpNode, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { getConstantValue, getPaddings, getPadOutput } from "./output";
import { padWebGLShader } from "./webGLShader";

// Note: The WebGLOpNode's node field is typed as Node from tfjs-converter

export function initPadWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  weightMap: NamedTensorsMap,
  modelType: ModelType,
  options: WedgeOptions
): WebGLOpNode {
  // GraphModel Pad: 2 inputs (tensor, paddings) or 3 inputs (tensor, paddings, constant)
  // LayersModel ZeroPadding2D: 1 input (tensor), padding in attrParams
  if (node.inputs.length < 1) {
    throw new Error(`Pad operation expects at least 1 input, got ${node.inputs.length}`);
  }

  const input = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);
  const output = getPadOutput(gl, node, nodeWebGLDataMap, opNodeMap, weightMap, options);

  // Get padding parameters
  const paddings = getPaddings(node, weightMap);
  const constantValue = getConstantValue(node, weightMap);

  const opParams: PadParams = {
    paddings,
    constantValue,
  };

  const opNode: WebGLOpNode = {
    node,
    inputs: [input],
    output,
    opParams,
    weights: [],
    type: "Pad",
    fsSource: "",
  };

  if (!opNodeHasMissingData(opNode)) {
    opNode.fsSource = padWebGLShader(opNode);
  }

  return opNode;
}
