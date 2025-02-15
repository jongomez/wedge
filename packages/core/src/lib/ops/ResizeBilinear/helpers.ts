import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { isWebGLDataNonTexture } from "../../helpers";
import { NodeWebGLDataMap, ResizeBilinearParams, WebGLData, WebGLOpNodeMap } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";

export function getResizeBilinearOriginalInputShape(
  node: Node,
  input: WebGLData | null,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  weightMap: NamedTensorsMap): number[] | null {
  if (input === null) {
    console.error("getResizeBilinearOriginalInputShape - some data may be missing");
    return null;
  }

  const inputWebGLData = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);

  if (isWebGLDataNonTexture(inputWebGLData)) {
    throw new Error("getResizeBilinearOriginalInputShape - inputWebGLData is not a texture");
  }

  // XXX: This should return THE ORIGINAL input shape.
  const inputShape = inputWebGLData?.originalShape;

  if (!inputShape) {
    console.error("getResizeBilinearOriginalInputShape - some data may be missing");
    return null;
  }

  return inputShape;
}



/*
alignCorners:  {value: false, type: 'bool'}
dtype: {value: 'float32', type: 'dtype'}
halfPixelCenters: {value: true, type: 'bool'}
*/

export const getResizeBilinearParams = (node: Node, inputShape: number[] | null, outputShape: number[] | null): ResizeBilinearParams | null => {
  const nodeAttrParams = node.attrParams;

  if (outputShape === null || inputShape == null) {
    return null
  }

  // Get the scaling factor:
  const size = outputShape[1] / inputShape[1] // index 0 is the batch dim.
  const alignCorners = nodeAttrParams.alignCorners.value
  const halfPixelCenters = nodeAttrParams.halfPixelCenters.value
  const dtype = nodeAttrParams.dtype.value as ResizeBilinearParams["dtype"]

  if (alignCorners != false) {
    throw new Error("getResizeBilinearParams - alignCorners value not supported: " + alignCorners);
  }

  if (halfPixelCenters != true) {
    throw new Error("getResizeBilinearParams - halfPixelCenters value not supported: " + halfPixelCenters);
  }

  return {
    size,
    alignCorners,
    halfPixelCenters,
    dtype,
  };
}