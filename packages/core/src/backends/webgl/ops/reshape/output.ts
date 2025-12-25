import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { createOutputTextureArray, getFromWeightMap } from "../../buffersAndTextures";
import { isWebGLDataNonTexture, isWebGLDataTextureArray } from "../../helpers";
import { NodeWebGLDataMap, WebGLDataTextureArray, WebGLOpNodeMap, WedgeOptions } from "../../types";
import { getWebGLDataElseNull } from "../../webGLData";

export function getNewShape(
  node: Node,
  weightMap: NamedTensorsMap
): number[] {
  // Case 1: LayersModel - targetShape is in attrParams
  if (node.attrParams?.targetShape?.value !== undefined) {
    const targetShape = node.attrParams.targetShape.value as number[];
    return [...targetShape];
  }

  // Case 2: GraphModel - the second input is a Const node containing the new shape
  if (node.inputs.length < 2) {
    throw new Error("Reshape requires a shape input (second input or targetShape attribute)");
  }

  const shapeNode = node.inputs[1];

  if (shapeNode.op !== "Const") {
    throw new Error("Reshape shape input must be a Const node. Got: " + shapeNode.op);
  }

  const shapeTensor = getFromWeightMap(weightMap, shapeNode.name);
  const shapeData = shapeTensor.dataSync();

  // Convert to number array
  const newShape: number[] = Array.from(shapeData).map(v => Number(v));

  return newShape;
}

export function getReshapeOutput(
  gl: WebGL2RenderingContext,
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
  newShape: number[],
  options: WedgeOptions
): WebGLDataTextureArray | null {
  const inputWebGLData = getWebGLDataElseNull(node.inputs[0], nodeWebGLDataMap, opNodeMap);

  if (inputWebGLData === null) {
    console.error("getReshapeOutput - input data is missing");
    return null;
  }

  if (isWebGLDataNonTexture(inputWebGLData)) {
    throw new Error("getReshapeOutput - inputWebGLData is not a texture");
  }

  const inputShape = inputWebGLData.originalShape;
  const inputElementCount = inputWebGLData.originalElementCount;

  // Handle -1 in newShape (infer dimension)
  let resolvedShape = resolveShape(newShape, inputElementCount, options);

  // For 4D output with hasBatchDimension: false, we may need to handle this differently
  // For now, just use the resolved shape directly
  const output = createOutputTextureArray(gl, resolvedShape, options, node.name);
  return output;
}

function resolveShape(newShape: number[], inputElementCount: number, options: WedgeOptions): number[] {
  let resolvedShape = [...newShape];

  // Find the index of -1 (if any)
  const inferIndex = resolvedShape.indexOf(-1);

  if (inferIndex !== -1) {
    // Calculate the inferred dimension
    let knownProduct = 1;
    for (let i = 0; i < resolvedShape.length; i++) {
      if (i !== inferIndex) {
        knownProduct *= resolvedShape[i];
      }
    }
    resolvedShape[inferIndex] = inputElementCount / knownProduct;
  }

  // Handle batch dimension (first dimension often 1 or -1)
  // If the shape starts with 1 and hasBatchDimension is false, remove it for texture creation
  if (resolvedShape.length === 4 && resolvedShape[0] === 1 && !options.hasBatchDimension) {
    resolvedShape = resolvedShape.slice(1);
  }

  // Handle 2D shapes by converting to 3D (add channel dimension)
  // e.g., [1, 117] -> [1, 117, 1] for texture compatibility
  if (resolvedShape.length === 2) {
    resolvedShape = [resolvedShape[0], resolvedShape[1], 1];
  }

  // Handle 1D shapes by converting to 3D
  // e.g., [117] -> [1, 117, 1]
  if (resolvedShape.length === 1) {
    resolvedShape = [1, resolvedShape[0], 1];
  }

  return resolvedShape;
}
