import { Conv2DParams, WebGLDataTextureArray } from "./types";

export const outputVar = 'fragColor';

export const outputVarName = 'outColor';
export const resultVarName = 'result';
export const weightsVarName = 'weights';

export const fsHeader = `#version 300 es
precision mediump float;
precision mediump int;
precision mediump sampler2D;`

export const defaultVsSource = `#version 300 es
in vec2 aVertexPosition; // Input attribute for vertex positions
void main() {
    gl_Position = vec4(aVertexPosition, 0.0, 1.0); // Convert to vec4, setting z = 0.0 and w = 1.0
}
`;


export const createRenderTargetLayoutsCode = (numberOfTextures: number): string => {
  let renderTargetLayouts = "";
  for (let i = 0; i < numberOfTextures; i++) {
    const isLast = i === numberOfTextures - 1;
    renderTargetLayouts += `layout(location = ${i}) out mediump vec4 ${outputVarName}${i};${isLast ? '' : '\n'}`;
  }

  return renderTargetLayouts;
}

export const createRenderTargetFinalCode = (numberOfTextures: number): string => {
  let finalCode = `\n`;
  for (let i = 0; i < numberOfTextures; i++) {
    const isLast = i === numberOfTextures - 1;
    finalCode += `  ${outputVarName}${i} = ${resultVarName}${i};${isLast ? '' : '\n'}`;
  }

  return finalCode;
}

export const createInitResultsCode = (numberOfTextures: number): string => {
  let finalCode = `\n`;
  for (let i = 0; i < numberOfTextures; i++) {
    const isLast = i === numberOfTextures - 1;
    finalCode += `  vec4 ${resultVarName}${i} = vec4(0.0);${isLast ? '' : '\n'}`;
  }

  return finalCode;
}

export const createAllOutputRealXYZCode = (outputNumberOfTextures: number, output: WebGLDataTextureArray): string => {
  let allOutputRealXYZ = "";

  for (let i = 0; i < outputNumberOfTextures; i++) {
    const flatCoordOffset = i * output.RGBATextureShape[0] * output.RGBATextureShape[1] * 4;
    allOutputRealXYZ += `ivec3 target${i}Output0RealXYZ = convertFlatToHWC3D(outputCoordFlat + ${flatCoordOffset}, outputDims.x, outputDims.y, paddedOutputDepth);
    `;
  }

  return allOutputRealXYZ;
}

export const createActivationCode = (numberOfTextures: number, activation: Conv2DParams["activation"]): string => {
  let finalActivationCode = "";

  if (activation === "relu") {
    for (let i = 0; i < numberOfTextures; i++) {
      finalActivationCode += `${resultVarName}${i} = max(${resultVarName}${i}, vec4(0.0));`
    }
  } else if (activation === "linear") {
    // Do nothing.
  } else {
    throw new Error("createActivationCode - Activation not supported: " + activation);
  }

  return finalActivationCode;
}
