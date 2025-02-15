import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { opNodeHasMissingData } from "../../helpers";
import { ArithmeticOpName, NodeWebGLDataMap, WebGLOpNode, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { getArithmeticOutput } from "./output";
import { arithmeticWebGLShader } from "./webGLShader";

export function initArithmeticWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  opName: ArithmeticOpName,
  options: WedgeOptions
): WebGLOpNode {

  const input1 = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);
  const input2 = getWebGLDataElseNull(node.inputs[1], nodeWebGLDataMap, opNodeMap);
  const inputs = [input1, input2];

  const output = getArithmeticOutput(gl, inputs, node, nodeWebGLDataMap, opNodeMap, options);

  const opNode: WebGLOpNode = {
    node,
    inputs,
    output,
    type: opName,
    weights: [],
    opParams: null,
    fsSource: "",
  }

  if (!opNodeHasMissingData(opNode)) {
    opNode.fsSource = arithmeticWebGLShader(opNode, opName)
  }

  return opNode;
}