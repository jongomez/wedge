import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray } from "../../buffersAndTextures";
import { NodeWebGLDataMap, OpName, WebGLDataTextureArray, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull, getWebGLOpOutputOriginalShape } from "../../webGLData";

export function getSingleInputBasicOutput(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  opName: OpName,
  options: WedgeOptions): WebGLDataTextureArray | null {
  const inputData = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);

  if (inputData === null) {
    console.error("getSingleInputBasicOutput - some data may be missing");
    return null;
  }

  let outputOriginalShape = getWebGLOpOutputOriginalShape(node, nodeWebGLDataMap, opNodeMap);

  const output = createOutputTextureArray(gl, outputOriginalShape, options, node.name);

  return output;
}