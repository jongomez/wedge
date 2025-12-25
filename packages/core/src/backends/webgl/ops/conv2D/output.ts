import { maxTextureDim } from "../../../../constants";
import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { getFromWeightMap } from "../../buffersAndTextures";
import { getConv2DParams } from "../../helpers";
import { ModelType, WedgeOptions } from "../../types";

export function getConv2DOutputShape(
  gl: WebGL2RenderingContext,
  inputShape: number[] | null,
  node: Node,
  weightMap: NamedTensorsMap,
  modelType: ModelType,
  options: WedgeOptions): number[] | null {
  if (inputShape === null) {
    return null;
  }

  let kernelWeightValues = getFromWeightMap(weightMap, node.inputs[1].name);

  const conv2DParams = getConv2DParams(node, kernelWeightValues.shape, node.inputs.length);

  // Assuming inputShape is channels last - NHWC, or [batch dim, height, width, depth]
  // I believe the other common format is NCHW - aka channels first.
  // Assume the batch dim was removed, so the shape should be [height, width, depth]
  const inputWidth = inputShape[0];
  const inputHeight = inputShape[1];

  const { strides, pad, kernelX, kernelY, numFilters } = conv2DParams;

  let strideX = strides[1];
  let strideY = strides[0];

  if (modelType === "GraphModel") {
    // Ignoring strides at index 0 and 3 for now.
    strideX = strides[2];
    strideY = strides[1];
  }

  let outputHeight, outputWidth;

  // References:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py#L1138-L1161
  // https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
  // https://stackoverflow.com/questions/68035443/what-does-padding-same-exactly-mean-in-tensorflow-conv2d-is-it-minimum-paddin
  if (pad === 'valid') {
    outputWidth = Math.ceil((inputWidth - kernelX + 1) / strideX);
    outputHeight = Math.ceil((inputHeight - kernelY + 1) / strideY);
  } else { // 'same' padding
    outputWidth = Math.ceil(inputWidth / strideX);
    outputHeight = Math.ceil(inputHeight / strideY);
  }

  // Assuming the number of filters determines the depth of the output
  const outputShape = options.hasBatchDimension ? [1, outputHeight, outputWidth, numFilters] : [outputHeight, outputWidth, numFilters];

  return outputShape;
}

/*
 
For the conv output texture array, we must:
- Have the output in HWC order - so the multiple textures in the array, when concatted, are in HWC order as well.
- Have the same x y pos in each output texture of the array belong to the same filter output.
  - But, the input position used to calculate the output value should be another one. Doesn't matter which one.
  - This means every single texture size in the output texture array must be a multiple of the number of filters.
    - If this is achieved, then all the x y output positions are aligned throughout the array.
    - This is the main difference between conv2D outputs and other outputs I believe.
    
*/

export function updateConv2DOutputDimensions(
  shape: number[],
  currentOutputSingleTextureWidth: number,
  currentOutputSingleTextureHeight: number,
  numRenderTargets: number,
  nodeName: string): [number, number, number, number] {
  if (shape.length !== 3) {
    throw new Error("createConv2DOutputShape - shape must have 3 dimensions. Node name: " + nodeName);
  }

  const numFilters = shape[2];
  const numRGBAFilters = numFilters / 4;
  const singleOutputTexturePositions = currentOutputSingleTextureWidth * currentOutputSingleTextureHeight;

  if (numFilters % 4 !== 0) {
    throw new Error("createConv2DOutputShape - numFilters must be divisible by 4. Node name: " + nodeName);
  }

  let width = numRGBAFilters;
  let minimumDifference = Infinity;
  let bestWidth = width;
  let bestHeight = Math.ceil(singleOutputTexturePositions / width);

  // In the following loop, we increment the width by the number of filters in each iteration.
  // This will guarantee that the final texture size will be a multiple of the number of filters.
  // The final best width and height values will be the ones with the minimum difference between them.
  while (width <= maxTextureDim) {
    const height = Math.ceil(singleOutputTexturePositions / width);
    const difference = Math.abs(width - height);

    if (width * height >= singleOutputTexturePositions && difference < minimumDifference) {
      bestWidth = width;
      bestHeight = height;
      minimumDifference = difference;
    }

    width += numRGBAFilters;
  }

  if (bestWidth >= maxTextureDim || bestHeight >= maxTextureDim) {
    throw new Error("Shape size is too large. Width: " + bestWidth + ", Height: " + bestHeight + ". Node name: " + nodeName);
  }

  if (bestWidth * bestHeight < singleOutputTexturePositions) {
    throw new Error("Error: bestWidth * bestHeight < singleOutputTextureElementCount. Width: " + bestWidth + ", Height: " + bestHeight + ", singleOutputTexturePositions: " + singleOutputTexturePositions + ". Node name: " + nodeName);
  }

  // Final sanity test - the final texture size must be a multiple of the number of filters.
  // This is the main goal of this function.
  if ((bestWidth * bestHeight) % numRGBAFilters !== 0) {
    throw new Error("Final texture size is not a multiple of the number of filters. Width: " + bestWidth + ", Height: " + bestHeight + ", numFilters: " + numFilters + ". Node name: " + nodeName);
  }

  const originalTextureArrayElementCount = singleOutputTexturePositions * numRenderTargets * 4;
  const finalTextureArrayElementCount = bestWidth * bestHeight * numRenderTargets * 4;

  return [bestWidth, bestHeight, originalTextureArrayElementCount, finalTextureArrayElementCount];
}
