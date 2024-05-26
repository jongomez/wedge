import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { NNShadersOptions, NodeWebGLDataMap, OpNodeWithWebGLData, OpNodeWithWebGLDataMap, WebGLData } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";

export function initNotSupportedOpWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: OpNodeWithWebGLDataMap,
  options: NNShadersOptions
): OpNodeWithWebGLData {

  let inputs: (WebGLData | null)[] = [];

  if (node.inputs.length > 0) {
    const currentInputWebGLData = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);
    inputs.push(currentInputWebGLData);
  }

  const opNodeWithWebGLData: OpNodeWithWebGLData = {
    node,
    inputs: inputs,
    output: null,
    type: "NotSupported",
    weights: [],
    opParams: null,
    fsSource: "",
  }

  return opNodeWithWebGLData;
}