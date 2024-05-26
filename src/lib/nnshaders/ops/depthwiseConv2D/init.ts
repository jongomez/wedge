import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray, createWeightDataTextureArray } from "../../buffersAndTextures";
import { checkConv2DInputs, getConv2DParams, opNodeHasMissingData } from "../../helpers";
import { biasWeightsTransform, conv2dWeightsTransform } from '../../transforms';
import { ModelType, NNShadersOptions, NodeWebGLDataMap, OpNodeWithWebGLData, OpNodeWithWebGLDataMap } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { getConv2DOriginalInputShape } from "../conv2D/init";
import { updateConv2DOutputDimensions } from "../conv2D/output";
import { getDepthwiseConv2DOutputShape } from "./output";
import { depthwiseConv2DWebGLShader } from "./webGLShader";

export function initDepthwiseConv2DWebGLData(
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
  const outputShape = getDepthwiseConv2DOutputShape(gl, originalInputShape, node, weightMap, modelType, options);

  // TODO (maybe, think about this): New updateConv2DOutputDimensions variant for dw convs?
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
    opNodeWithWebGLData.fsSource = depthwiseConv2DWebGLShader(opNodeWithWebGLData);
  }

  return opNodeWithWebGLData;
}