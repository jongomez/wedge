import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray } from "../../buffersAndTextures";
import { NNShadersOptions, NodeWebGLDataMap, OpNodeWithWebGLDataMap, WebGLData, WebGLDataTextureArray } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";

export function getArithmeticOutput(
  gl: WebGL2RenderingContext,
  inputs: (WebGLData | null)[],
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: OpNodeWithWebGLDataMap,
  options: NNShadersOptions): WebGLDataTextureArray | null {
  let outputOriginalShape: number[] = [];

  if (node.inputs.length !== 2 || inputs.length !== 2) {
    throw new Error("Expected 2 inputs for Mul or AddV2");
  }

  if (inputs[0] === null || inputs[1] === null) {
    console.error("getArithmeticOutput - some data may be missing");
    return null;
  }

  for (const input of node.inputs) {
    let webGLData = getWebGLDataElseNull(input, nodeWebGLDataMap, opNodeMap);

    if (webGLData?.shape.length) {
      outputOriginalShape = webGLData.originalShape;
    }
  }



  const output = createOutputTextureArray(gl, outputOriginalShape, options, node.name);

  return output;
}
