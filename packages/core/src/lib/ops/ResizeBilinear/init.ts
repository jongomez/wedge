import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray } from "../../buffersAndTextures";
import { opNodeHasMissingData } from "../../helpers";
import { ModelType, NodeWebGLDataMap, WebGLOpNode, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { getResizeBilinearOriginalInputShape, getResizeBilinearParams } from "./helpers";
import { getResizeBilinearOutputShape } from "./output";
import { resizeBilinearWebGLShader } from "./webGLShader";

export function initResizeBilinearWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  weightMap: NamedTensorsMap,
  modelType: ModelType,
  options: WedgeOptions
): WebGLOpNode {
  const input = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);
  const originalInputShape = getResizeBilinearOriginalInputShape(node, input, nodeWebGLDataMap, opNodeMap, weightMap);
  const outputShape = getResizeBilinearOutputShape(gl, originalInputShape, node, weightMap, modelType, options);
  const output = createOutputTextureArray(gl, outputShape, options, node.name);

  const opParams = getResizeBilinearParams(node, originalInputShape, outputShape);

  const opNode: WebGLOpNode = {
    node,
    inputs: [input],
    output,
    opParams,
    weights: [],
    type: "ResizeBilinear",
    fsSource: "",
  }

  if (!opNodeHasMissingData(opNode)) {
    opNode.fsSource = resizeBilinearWebGLShader(opNode);
  }

  return opNode;
}
