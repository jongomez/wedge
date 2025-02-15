import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray } from "../../buffersAndTextures";
import { NodeWebGLDataMap, WebGLData, WebGLDataTextureArray, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLOpOutputOriginalShape } from "../../webGLData";

export function getArithmeticOutput(
  gl: WebGL2RenderingContext,
  inputs: (WebGLData | null)[],
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  options: WedgeOptions): WebGLDataTextureArray | null {

  if (node.inputs.length !== 2 || inputs.length !== 2) {
    throw new Error("Expected 2 inputs for Mul or AddV2");
  }

  if (inputs[0] === null || inputs[1] === null) {
    console.error("getArithmeticOutput - some data may be missing");
    return null;
  }

  let outputOriginalShape = getWebGLOpOutputOriginalShape(node, nodeWebGLDataMap, opNodeMap);

  const output = createOutputTextureArray(gl, outputOriginalShape, options, node.name);

  return output;
}
