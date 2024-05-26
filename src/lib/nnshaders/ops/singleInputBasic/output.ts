import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray } from "../../buffersAndTextures";
import { NNShadersOptions, NodeWebGLDataMap, OpName, OpNodeWithWebGLDataMap, WebGLDataTextureArray } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";

export function getSingleInputBasicOutput(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: OpNodeWithWebGLDataMap,
  opName: OpName,
  options: NNShadersOptions): WebGLDataTextureArray | null {
  const inputData = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);

  if (inputData === null) {
    console.error("getSingleInputBasicOutput - some data may be missing");
    return null;
  }

  const output = createOutputTextureArray(gl, inputData?.originalShape, options, node.name);

  return output;
}