import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import assert from "assert";
import { getFromWeightMap } from "../../buffersAndTextures";
import { ModelType, WedgeOptions } from "../../types";

export function getResizeBilinearOutputShape(
  gl: WebGL2RenderingContext,
  inputShape: number[] | null,
  node: Node,
  weightMap: NamedTensorsMap,
  modelType: ModelType,
  options: WedgeOptions): number[] | null {

  if (inputShape === null) {
    return null;
  }

  assert(inputShape.length === 4, "getResizeBilinearOutputShape - inputShape.length should be 4. Got: " + inputShape.length)

  let kernelWeightValues = getFromWeightMap(weightMap, node.inputs[1].name);

  // I think the weights for the bilinear upsampling are the final 2d spatial dims - not 100% sure.
  const outputSpatialDims = kernelWeightValues.dataSync()

  // Assuming inputShape is channels last - NHWC, or [batch dim, height, width, depth]
  // I believe the other common format is NCHW - aka channels first.
  // Assume the batch dim was removed, so the shape should be [height, width, depth]
  const inputWidth = inputShape[0];
  const inputHeight = inputShape[1];
  const numFilters = inputShape[3]

  const outputShape = options.hasBatchDimension ? [1, outputSpatialDims[0], outputSpatialDims[1], numFilters] : [outputSpatialDims[0], outputSpatialDims[1], numFilters];

  return outputShape;
}
