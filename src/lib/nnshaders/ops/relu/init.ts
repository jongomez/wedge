import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { NNShadersOptions, NodeWebGLDataMap, WebGLOpNode, WebGLOpNodeMap } from "../../types";
import { initSingleInputBasicWebGLData } from "../singleInputBasic/init";

export function initReluWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  options: NNShadersOptions
): WebGLOpNode {
  const opNodeWithWebGLData = initSingleInputBasicWebGLData(
    gl,
    node,
    nodeWebGLDataMap,
    opNodeMap,
    "Relu",
    options);

  return opNodeWithWebGLData;
}