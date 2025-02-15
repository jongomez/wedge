import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { ModelType, WedgeOptions } from "../../types";
import { getConv2DOutputShape } from "../conv2D/output";

export function getDepthwiseConv2DOutputShape(
  gl: WebGL2RenderingContext,
  inputOriginalShape: number[] | null,
  node: Node,
  weightMap: NamedTensorsMap,
  modelType: ModelType,
  options: WedgeOptions): number[] | null {
  const conv2DOutputShape = getConv2DOutputShape(gl, inputOriginalShape, node, weightMap, modelType, options);

  if (conv2DOutputShape === null || inputOriginalShape === null) {
    return null;
  }

  // Convert the conv2D output shape to a valid depthwiseConv2D output shape. 
  // On regular conv2D outputs, the number of filters is the last dimension.
  // However, for depthwise conv2D outputs, the last dimension is the number of input channels * depthMultiplier.
  // (assuming depthMultiplier is always 1 for now) 
  const depthwiseConv2DOutputShape = conv2DOutputShape.slice();
  depthwiseConv2DOutputShape[2] = inputOriginalShape[2];

  return depthwiseConv2DOutputShape;
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

/*
export function updateDepthwiseConv2DOutputDimensions(
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
*/
