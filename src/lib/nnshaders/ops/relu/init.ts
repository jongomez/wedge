import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { NNShadersOptions, NodeWebGLDataMap, OpNodeWithWebGLData, OpNodeWithWebGLDataMap } from "../../types";
import { initSingleInputBasicWebGLData } from "../singleInputBasic/init";

export function initReluWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: OpNodeWithWebGLDataMap,
  options: NNShadersOptions
): OpNodeWithWebGLData {
  const opNodeWithWebGLData = initSingleInputBasicWebGLData(
    gl,
    node,
    nodeWebGLDataMap,
    opNodeMap,
    "Relu",
    options);

  return opNodeWithWebGLData;
}