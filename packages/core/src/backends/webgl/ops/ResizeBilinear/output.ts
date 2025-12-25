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

  // Handle both 3D (HWC) and 4D (NHWC) input shapes
  const is4D = inputShape.length === 4;
  const is3D = inputShape.length === 3;
  assert(is3D || is4D, "getResizeBilinearOutputShape - inputShape.length should be 3 or 4. Got: " + inputShape.length)

  let kernelWeightValues = getFromWeightMap(weightMap, node.inputs[1].name);

  // The second input contains the target spatial dimensions [height, width]
  const outputSpatialDims = kernelWeightValues.dataSync()

  // Extract number of channels from input shape
  // 4D: [batch, height, width, channels] -> channels at index 3
  // 3D: [height, width, channels] -> channels at index 2
  const numFilters = is4D ? inputShape[3] : inputShape[2];

  const outputShape = options.hasBatchDimension ? [1, outputSpatialDims[0], outputSpatialDims[1], numFilters] : [outputSpatialDims[0], outputSpatialDims[1], numFilters];

  return outputShape;
}
