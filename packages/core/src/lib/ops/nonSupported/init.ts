import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { NodeWebGLDataMap, WebGLData, WebGLOpNode, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";

export function initNotSupportedOpWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  options: WedgeOptions
): WebGLOpNode {

  let inputs: (WebGLData | null)[] = [];

  if (node.inputs.length > 0) {
    const currentInputWebGLData = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);
    inputs.push(currentInputWebGLData);
  }

  const webGLOpNode: WebGLOpNode = {
    node,
    inputs: inputs,
    output: null,
    type: "NotSupported",
    weights: [],
    opParams: null,
    fsSource: "",
  }

  return webGLOpNode;
}