import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray, createWeightDataTextureArray } from "../../buffersAndTextures";
import { checkConv2DInputs, getConv2DParams, opNodeHasMissingData } from "../../helpers";
import { biasWeightsTransform, conv2dWeightsTransform } from '../../transforms';
import { ModelType, NNShadersOptions, NodeWebGLDataMap, OpNodeWithWebGLData, OpNodeWithWebGLDataMap, WebGLData } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { getConv2DOutputShape, updateConv2DOutputDimensions } from "./output";
import { conv2DWebGLShader } from "./webGLShader";

export function getConv2DOriginalInputShape(
  node: Node,
  input: WebGLData | null,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: OpNodeWithWebGLDataMap,
  weightMap: NamedTensorsMap): number[] | null {
  if (input === null) {
    console.error("getConv2DOriginalInputShape - some data may be missing");
    return null;
  }

  checkConv2DInputs(node.inputs);

  // XXX: This should return THE ORIGINAL input shape.
  const inputShape = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap)?.originalShape;

  if (!inputShape) {
    console.error("getConv2DOriginalInputShape - some data may be missing");
    return null;
  }

  return inputShape;
}

export function initConv2DWebGLData(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: OpNodeWithWebGLDataMap,
  weightMap: NamedTensorsMap,
  modelType: ModelType,
  options: NNShadersOptions
): OpNodeWithWebGLData {
  checkConv2DInputs(node.inputs);

  const input = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);
  const originalInputShape = getConv2DOriginalInputShape(node, input, nodeWebGLDataMap, opNodeMap, weightMap);
  const outputShape = getConv2DOutputShape(gl, originalInputShape, node, weightMap, modelType, options);
  const output = createOutputTextureArray(gl, outputShape, options, node.name, updateConv2DOutputDimensions);

  const kernelWeights = createWeightDataTextureArray(
    gl,
    weightMap,
    node.inputs[1],
    "uKernelWeights",
    false,
    "HWC",
    conv2dWeightsTransform);

  const weights = [kernelWeights];

  const opParams = getConv2DParams(node, kernelWeights.originalShape, node.inputs.length);

  // If there's a 3rd input, it should be the bias weights.
  if (node.inputs.length === 3) {
    const biasWeightValues = createWeightDataTextureArray(
      gl,
      weightMap,
      node.inputs[2],
      "uBiasWeights",
      false,
      "VEC",
      biasWeightsTransform);
    weights.push(biasWeightValues);
  }

  const opNodeWithWebGLData: OpNodeWithWebGLData = {
    node,
    inputs: [input],
    output,
    opParams,
    weights,
    type: "Conv2D",
    fsSource: "",
  }

  if (!opNodeHasMissingData(opNodeWithWebGLData)) {
    opNodeWithWebGLData.fsSource = conv2DWebGLShader(opNodeWithWebGLData);
  }

  return opNodeWithWebGLData;
}