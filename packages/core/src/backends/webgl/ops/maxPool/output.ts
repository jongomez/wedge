import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray } from "../../buffersAndTextures";
import { MaxPoolParams, ModelType, NodeWebGLDataMap, WebGLDataTextureArray, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";
import { isWebGLDataTextureArray } from "../../helpers";

// Helper to normalize scalar or array to [H, W] array
function normalizeToArray(value: number | number[] | undefined, defaultValue: number[]): number[] {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value === 'number') {
    return [value, value];
  }
  if (Array.isArray(value)) {
    if (value.length === 1) {
      return [value[0], value[0]];
    }
    return value;
  }
  return defaultValue;
}

export function getMaxPoolParams(node: Node, modelType: ModelType): MaxPoolParams {
  const nodeAttrParams = node.attrParams;

  let poolSize: number[];
  let strides: number[];
  let pad: "same" | "valid";

  if (modelType === "GraphModel") {
    // GraphModel format: ksize and strides are [1, H, W, 1] (NHWC format)
    const ksize = nodeAttrParams.filterSize?.value as number[] || [1, 2, 2, 1];
    const stridesRaw = nodeAttrParams.strides?.value as number[] || [1, 2, 2, 1];
    pad = (nodeAttrParams.pad?.value as string)?.toLowerCase() as "same" | "valid" || "valid";

    // Extract H, W from NHWC format
    poolSize = [ksize[1], ksize[2]];
    strides = [stridesRaw[1], stridesRaw[2]];
  } else {
    // LayersModel format - poolSize and strides can be scalar or array
    const poolSizeRaw = nodeAttrParams.poolSize?.value;
    const stridesRaw = nodeAttrParams.strides?.value;

    poolSize = normalizeToArray(poolSizeRaw as number | number[], [2, 2]);
    strides = normalizeToArray(stridesRaw as number | number[], poolSize);

    pad = (nodeAttrParams.padding?.value as string)?.toLowerCase() as "same" | "valid" ||
          (nodeAttrParams.pad?.value as string)?.toLowerCase() as "same" | "valid" || "valid";
  }

  return {
    poolSize,
    strides,
    pad
  };
}

export function getMaxPoolOutputShape(
  inputShape: number[],
  params: MaxPoolParams,
  options: WedgeOptions
): number[] {
  // Input shape is [H, W, C] or [N, H, W, C]
  const inputIs4D = inputShape.length === 4;
  const inputH = inputIs4D ? inputShape[1] : inputShape[0];
  const inputW = inputIs4D ? inputShape[2] : inputShape[1];
  const inputC = inputIs4D ? inputShape[3] : inputShape[2];

  const { poolSize, strides, pad } = params;
  const poolH = poolSize[0];
  const poolW = poolSize[1];
  const strideH = strides[0];
  const strideW = strides[1];

  let outputH: number;
  let outputW: number;

  if (pad === 'valid') {
    outputH = Math.ceil((inputH - poolH + 1) / strideH);
    outputW = Math.ceil((inputW - poolW + 1) / strideW);
  } else {
    // 'same' padding
    outputH = Math.ceil(inputH / strideH);
    outputW = Math.ceil(inputW / strideW);
  }

  // Channels remain unchanged in pooling
  const outputC = inputC;

  return options.hasBatchDimension
    ? [1, outputH, outputW, outputC]
    : [outputH, outputW, outputC];
}

export function getMaxPoolOutput(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  modelType: ModelType,
  options: WedgeOptions
): { output: WebGLDataTextureArray | null; params: MaxPoolParams } {
  const inputData = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);

  if (!inputData || !isWebGLDataTextureArray(inputData)) {
    console.error("getMaxPoolOutput - input data is missing or not a texture array");
    return { output: null, params: { poolSize: [2, 2], strides: [2, 2], pad: "valid" } };
  }

  const params = getMaxPoolParams(node, modelType);
  const outputShape = getMaxPoolOutputShape(inputData.originalShape, params, options);

  const output = createOutputTextureArray(gl, outputShape, options, node.name);

  return { output, params };
}
