import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { NodeWebGLDataMap, WebGLOpNode, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { initSingleInputBasicWebGLData } from "../singleInputBasic/init";

export function initReluWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  options: WedgeOptions
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