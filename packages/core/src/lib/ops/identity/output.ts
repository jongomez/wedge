import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray } from "../../buffersAndTextures";
import { NodeWebGLDataMap, OpName, WebGLDataTextureArray, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull, getWebGLOpOutputOriginalShape } from "../../webGLData";

export function getIdentityOutput(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  opName: OpName,
  options: WedgeOptions): WebGLDataTextureArray | null {
  const inputData = getWebGLDataElseNull(node, nodeWebGLDataMap, opNodeMap);

  if (!inputData) {
    console.error("getIdentityOutput - some data may be missing");
    return null;
  }

  let outputOriginalShape = getWebGLOpOutputOriginalShape(node, nodeWebGLDataMap, opNodeMap);
  const output = createOutputTextureArray(gl, outputOriginalShape, options, node.name);

  return output;
}