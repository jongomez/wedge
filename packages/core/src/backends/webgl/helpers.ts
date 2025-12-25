import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { Conv2DParams, ModelConfig, ModelType, WebGLData, WebGLDataNonTexture, WebGLDataTexture, WebGLDataTextureArray, WebGLOpNode, WedgeOptions } from "./types";

export const getShapeUniformName = (nodeDataUniformName: string) => {
  return nodeDataUniformName + "Shape";
}

export const getSquareShapeUniformName = (nodeDataUniformName: string) => {
  return nodeDataUniformName + "SquareShape";
}

export const getConv2DParams = (node: Node, kernelWeightsShape: number[], numInputs: number): Conv2DParams => {
  if (kernelWeightsShape.length !== 4) {
    throw new Error("Conv2D kernelWeightsShape.length must be 4. Shape is: " + kernelWeightsShape);
  }

  const nodeAttrParams = node.attrParams;

  const pad = nodeAttrParams.pad.value as "same" | "valid";
  const strides = nodeAttrParams.strides.value as number[];
  let activation: 'relu' | null = null;

  if (nodeAttrParams.activation) {
    activation = nodeAttrParams.activation.value as 'relu';
  }

  const kernelX = kernelWeightsShape[0];
  const kernelY = kernelWeightsShape[1];
  const kernelDepth = kernelWeightsShape[2];

  const numFilters = kernelWeightsShape[3];
  const hasBias = numInputs === 2;

  return {
    strides,
    pad,
    kernelX,
    kernelY,
    kernelDepth,
    numFilters,
    activation,
    hasBias
  };
}

export const checkConv2DInputs = (inputs: Node[]) => {
  if (inputs.length > 4) {
    throw new Error("Expected 4 or fewer inputs for Conv2D. Got: " + inputs.length);
  }

  if (inputs[1].op !== "Const") {
    throw new Error("Second input to Conv2D must be a Const node (the kernel weights). Got: " + inputs[1].op);
  }

  if (inputs.length === 3 && inputs[2].op !== "Const") {
    throw new Error("Third input to Conv2D must be a Const node (the bias weights). Got: " + inputs[2].op);
  }

  if (inputs.length === 4 && inputs[3].op !== "Const") {
    throw new Error("Fourth input to Conv2D must be a Const node (params for the activation). Got: " + inputs[2].op);
  }
}

export const getDoubleInputWebGLData = (inputs: WebGLDataTexture[], name: string): [WebGLData, WebGLData] => {
  if (inputs.length !== 2) {
    throw new Error("Required exactly 2 inputs for: " + name + ". Got: " + inputs.length);
  }

  return [inputs[0], inputs[1]];
}

export function checkFloatTextureSupport(gl: WebGL2RenderingContext): boolean {
  const testTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, testTexture);
  const height = 1;
  const width = 1;
  const floatTestArray = new Float32Array(height * width * 4);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, floatTestArray);

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    console.error("32-bit floating point textures are not fully supported in this context");
    return false;
  }

  // Clean up
  gl.deleteTexture(testTexture);

  console.log("32-bit floating point textures supported");
  return true;
}

export function isWebGLDataTextureArray(data: WebGLData | null): data is WebGLDataTextureArray {
  return data?.webGLType === "sampler2DArray";
}

export function isWebGLDataTexture(data: WebGLData | null): data is WebGLDataTexture {
  return data?.webGLType === "sampler2D";
}

export function isWebGLDataNonTexture(data: WebGLData | null): data is WebGLDataNonTexture {
  return data?.webGLType === "float" || data?.webGLType === "vec2" || data?.webGLType === "vec3" || data?.webGLType === "vec4";
}

export function findRecommendedRenderTargets(
  outputTextureElementCount: number,
  renderTargetBreakpoints: WedgeOptions["renderTargetBreakpoints"]): number {

  // Assuming the array is sorted by outputCount.
  for (let i = renderTargetBreakpoints.length - 1; i >= 0; i--) {
    if (outputTextureElementCount >= renderTargetBreakpoints[i].outputTextureElementCount) {
      return renderTargetBreakpoints[i].numberOfRenderTargets;
    }
  }

  // Default to 1 if no higher recommendation exists.
  return 1;
}

export const opNodeHasMissingInputs = (opNode: WebGLOpNode): boolean => {
  return opNode.inputs.some(input => input === null);
}

export const opNodeHasMissingOutput = (opNode: WebGLOpNode): boolean => {
  return opNode.output === null;
}

export function opNodeHasMissingData(opNode: WebGLOpNode): boolean {
  const hasMissingInputs = opNodeHasMissingInputs(opNode);
  const hasMissingOutput = opNodeHasMissingOutput(opNode);

  return hasMissingInputs || hasMissingOutput;
}

export function loop(numberUntilWeWantToLoop: number, callback: (index: number) => string): string {
  let result = "";
  for (let i = 0; i < numberUntilWeWantToLoop; i++) {
    result += callback(i);
  }
  return result;
}

export function getValidShape(shape: number[]): number[] {
  const shapeCopy = shape.slice();

  if (shapeCopy[0] === -1) {
    shapeCopy[0] = 1;
  }

  return shapeCopy;
}

export function getModelType(modelConfig: ModelConfig): ModelType {
  const { url } = modelConfig;

  if (url.endsWith(".tflite")) {
    return "tflite";
  } else if (url.endsWith(".json")) {
    return "GraphModel";
  } else {
    throw new Error(`Unknown model type for url: ${url}`);
  }

}

