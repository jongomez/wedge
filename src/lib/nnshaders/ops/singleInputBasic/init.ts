import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { opNodeHasMissingData } from "../../helpers";
import { NNShadersOptions, NodeWebGLDataMap, SingleInputBasicOpName, WebGLOpNode, WebGLOpNodeMap } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { getSingleInputBasicOutput } from "./output";
import { singleInputBasicWebGLShader } from "./webGLShader";

export function initSingleInputBasicWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  opName: SingleInputBasicOpName,
  options: NNShadersOptions
): WebGLOpNode {
  if (node.inputs.length !== 1) {
    throw new Error(`Node ${node.name} has ${node.inputs.length} inputs, expected 1`);
  }

  const input = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);
  const output = getSingleInputBasicOutput(gl, node, nodeWebGLDataMap, opNodeMap, opName, options);

  const opNode: WebGLOpNode = {
    node,
    inputs: [input],
    output,
    type: opName,
    weights: [],
    opParams: null,
    fsSource: "",
  }

  if (!opNodeHasMissingData(opNode)) {
    opNode.fsSource = singleInputBasicWebGLShader(opNode, opName)
  }

  return opNode;
}