import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray, getFromWeightMap } from "../../buffersAndTextures";
import { NodeWebGLDataMap, WebGLDataTextureArray, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { isWebGLDataNonTexture, isWebGLDataTextureArray } from "../../helpers";

export function getPadOriginalInputShape(
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap): number[] | null {
  const inputWebGLData = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);

  if (inputWebGLData === null) {
    console.error("getPadOriginalInputShape - input data is missing");
    return null;
  }

  if (isWebGLDataNonTexture(inputWebGLData)) {
    throw new Error("getPadOriginalInputShape - inputWebGLData is not a texture");
  }

  const inputShape = inputWebGLData?.originalShape;

  if (!inputShape) {
    console.error("getPadOriginalInputShape - input shape is missing");
    return null;
  }

  return inputShape;
}

export function getPaddings(
  node: Node,
  weightMap: NamedTensorsMap): number[][] {
  // Case 1: LayersModel (ZeroPadding2D) - padding is in attrParams
  // Note: getAttrParams in create.ts stores it under 'pad' key
  if (node.attrParams?.pad?.value !== undefined) {
    return parsePaddingAttr(node.attrParams.pad.value);
  }

  // Case 2: GraphModel - paddings come from second input (Const node)
  if (node.inputs.length < 2) {
    throw new Error("getPaddings - no padding attribute or second input found");
  }

  const paddingsNode = node.inputs[1];

  if (paddingsNode.op !== "Const") {
    throw new Error("getPaddings - paddings input must be a Const node. Got: " + paddingsNode.op);
  }

  const paddingsTensor = getFromWeightMap(weightMap, paddingsNode.name);
  const paddingsData = paddingsTensor.dataSync();
  const paddingsShape = paddingsTensor.shape;

  // Paddings should be a 2D tensor with shape [rank, 2]
  if (paddingsShape.length !== 2 || paddingsShape[1] !== 2) {
    throw new Error("getPaddings - paddings must have shape [rank, 2]. Got: " + paddingsShape);
  }

  const rank = paddingsShape[0];
  const paddings: number[][] = [];

  for (let i = 0; i < rank; i++) {
    paddings.push([paddingsData[i * 2], paddingsData[i * 2 + 1]]);
  }

  return paddings;
}

// Parse padding attribute from ZeroPadding2D layer config
// Returns paddings in HWC format: [[h_before, h_after], [w_before, w_after], [0, 0]]
function parsePaddingAttr(padding: number | number[] | number[][]): number[][] {
  let hPad: [number, number];
  let wPad: [number, number];

  if (typeof padding === 'number') {
    // Symmetric padding: 1 → [[1,1], [1,1]]
    hPad = [padding, padding];
    wPad = [padding, padding];
  } else if (Array.isArray(padding) && padding.length === 2) {
    if (Array.isArray(padding[0])) {
      // Asymmetric: [[1,2], [3,4]] → height [1,2], width [3,4]
      hPad = padding[0] as [number, number];
      wPad = padding[1] as [number, number];
    } else {
      // Per-dimension symmetric: [1, 2] → [[1,1], [2,2]]
      hPad = [padding[0] as number, padding[0] as number];
      wPad = [padding[1] as number, padding[1] as number];
    }
  } else {
    throw new Error("parsePaddingAttr - unsupported padding format: " + JSON.stringify(padding));
  }

  // HWC format: [height, width, channels (no pad)]
  return [hPad, wPad, [0, 0]];
}

export function getConstantValue(
  node: Node,
  weightMap: NamedTensorsMap): number {
  // For PadV2, the third input is the constant value
  // For Pad, it's in the attributes with default 0
  if (node.inputs.length >= 3) {
    const constantValueNode = node.inputs[2];
    if (constantValueNode.op === "Const") {
      const constantTensor = getFromWeightMap(weightMap, constantValueNode.name);
      return constantTensor.dataSync()[0];
    }
  }

  // Check attributes for constant_value (Pad op)
  if (node.attrParams?.constantValue?.value !== undefined) {
    return node.attrParams.constantValue.value as number;
  }

  return 0; // Default padding value
}

export function getPadOutputShape(
  inputShape: number[] | null,
  paddings: number[][],
  options: WedgeOptions): number[] | null {
  if (inputShape === null) {
    return null;
  }

  // For 3D input (HWC), paddings should be [[h_before, h_after], [w_before, w_after], [c_before, c_after]]
  // For 4D input (NHWC), paddings should be [[n_before, n_after], [h_before, h_after], [w_before, w_after], [c_before, c_after]]

  const outputShape: number[] = [];

  for (let i = 0; i < inputShape.length; i++) {
    if (i < paddings.length) {
      outputShape.push(inputShape[i] + paddings[i][0] + paddings[i][1]);
    } else {
      outputShape.push(inputShape[i]);
    }
  }

  return outputShape;
}

export function getPadOutput(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  weightMap: NamedTensorsMap,
  options: WedgeOptions): WebGLDataTextureArray | null {
  const inputShape = getPadOriginalInputShape(node, nodeWebGLDataMap, opNodeMap);
  let paddings = getPaddings(node, weightMap);

  if (!inputShape) {
    console.error("getPadOutput - could not determine input shape");
    return null;
  }

  // Adjust paddings to match input dimensions
  // parsePaddingAttr returns HWC format, but input might be 4D (NHWC)
  if (inputShape.length === 4 && paddings.length === 3) {
    // Prepend [0, 0] for batch dimension
    paddings = [[0, 0], ...paddings];
  }

  const outputShape = getPadOutputShape(inputShape, paddings, options);

  if (!outputShape) {
    console.error("getPadOutput - could not determine output shape");
    return null;
  }

  // For 4D input with hasBatchDimension: false, we need to output 3D shape
  // (this matches how conv2D handles it - the batch dim is squeezed internally)
  let finalOutputShape = outputShape;
  if (outputShape.length === 4 && !options.hasBatchDimension) {
    // Remove the batch dimension for the output texture
    finalOutputShape = outputShape.slice(1);
  }

  const output = createOutputTextureArray(gl, finalOutputShape, options, node.name);
  return output;
}
