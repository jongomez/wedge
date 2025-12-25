import { isWebGLDataTextureArray } from "../../helpers";
import { createRenderTargetFinalCode, createRenderTargetLayoutsCode, fsHeader } from "../../shaderHelpers";
import { PadParams, WebGLDataTextureArray, WebGLOpNode } from "../../types";

export const padWebGLShader = (opNode: WebGLOpNode): string => {
  if (opNode.inputs.length !== 1) {
    throw new Error(`Pad operation expects 1 input, got ${opNode.inputs.length}`);
  }

  const input = opNode.inputs[0];
  const output = opNode.output;
  const params = opNode.opParams as PadParams;

  if (!input) {
    throw new Error("Pad operation requires an input");
  }

  if (!output) {
    throw new Error("Pad operation requires an output");
  }

  if (!isWebGLDataTextureArray(input)) {
    throw new Error("Pad operation requires input to be a texture array. Got: " + input.webGLType);
  }

  const inputNumberOfTextures = input.RGBATextureShape[2];
  if (inputNumberOfTextures === 0) {
    throw new Error("Pad operation requires at least one input texture.");
  }

  // Uniforms
  const textureArrayUniform = `uniform mediump sampler2DArray ${input.uniformName};`;

  const numberOfTextures = output.RGBATextureShape[2];
  const renderTargetLayoutsCode = createRenderTargetLayoutsCode(numberOfTextures);
  const finalResultCode = createRenderTargetFinalCode(numberOfTextures);

  // Extract padding values - paddings are always in HWC format (normalized in init.ts)
  // paddings[0] = [height_before, height_after]
  // paddings[1] = [width_before, width_after]
  // paddings[2] = [channels_before, channels_after] - usually [0, 0]
  const { paddings, constantValue } = params;

  const padHeightBefore = paddings.length > 0 ? paddings[0][0] : 0;
  const padWidthBefore = paddings.length > 1 ? paddings[1][0] : 0;
  const padChannelsBefore = paddings.length > 2 ? paddings[2][0] : 0;

  // Handle both 3D [H, W, C] and 4D [N, H, W, C] input shapes
  // Extract HWC dimensions, skipping batch dim if present
  const inputIs4D = input.originalShape.length === 4;
  const inputH = inputIs4D ? input.originalShape[1] : input.originalShape[0];
  const inputW = inputIs4D ? input.originalShape[2] : input.originalShape[1];
  const inputC = inputIs4D ? input.originalShape[3] : input.originalShape[2];

  const constants = `
const ivec3 inputDims = ivec3(${inputH}, ${inputW}, ${inputC});
const ivec3 inputTextureArrayDims = ivec3(${input.RGBATextureShape[0]}, ${input.RGBATextureShape[1]}, ${input.RGBATextureShape[2]});
const ivec3 outputDims = ivec3(${output.originalShape[0]}, ${output.originalShape[1]}, ${output.originalShape[2]});
const ivec3 outputTextureArrayDims = ivec3(${output.RGBATextureShape[0]}, ${output.RGBATextureShape[1]}, ${output.RGBATextureShape[2]});

// padBefore order must match outputRealXYZ from convertFlatToHWC3D which returns (w, h, c)
const ivec3 padBefore = ivec3(${padWidthBefore}, ${padHeightBefore}, ${padChannelsBefore});
const float constantValue = ${constantValue.toFixed(6)};

const int inputTextureSpatialSize = inputTextureArrayDims.x * inputTextureArrayDims.y;
const int paddedInputDepth = ((inputDims.z + 3) / 4) * 4;
const int paddedOutputDepth = ((outputDims.z + 3) / 4) * 4;
`;

  return `${fsHeader}

${textureArrayUniform}

${constants}
${renderTargetLayoutsCode}

ivec3 convertFlatToTextureArray3D(int flatIndex, int width, int height, int textureSpatialSize) {
  int textureArrayFlatIndex = flatIndex / 4;
  int textureArrayFlatIndexModulo = textureArrayFlatIndex % textureSpatialSize;

  int x = textureArrayFlatIndexModulo % width;
  int y = textureArrayFlatIndexModulo / width;
  int z = textureArrayFlatIndex / textureSpatialSize;

  return ivec3(x, y, z);
}

// Convert flat HWC index to 3D coordinates
ivec3 convertFlatToHWC3D(int flatIndex, int width, int height, int depth) {
  int z = flatIndex % depth;
  int x = flatIndex / depth % width;
  int y = flatIndex / (depth * width);

  return ivec3(x, y, z);
}

void main() {
  // Each output spatial xy position can output 4 * numberOfTextures different values
  int outputCoordFlat = (int(gl_FragCoord.x) + int(gl_FragCoord.y) * outputTextureArrayDims.x) * 4;

  ${function() {
    let code = '';
    for (let i = 0; i < numberOfTextures; i++) {
      const flatCoordOffset = i * output.RGBATextureShape[0] * output.RGBATextureShape[1] * 4;
      code += `
  // Render target ${i}
  // convertFlatToHWC3D returns ivec3(x, y, z) matching the convention used in conv2D
  // outputDims = (H, W, C), pass (H, W) as (width, height) to match conv2D
  ivec3 output${i}RealXYZ = convertFlatToHWC3D(outputCoordFlat + ${flatCoordOffset}, outputDims.x, outputDims.y, paddedOutputDepth);

  // Calculate corresponding input position by subtracting padding
  // padBefore = (padH, padW, padC), subtract directly
  ivec3 input${i}RealXYZ = output${i}RealXYZ - padBefore;

  // Check if we're in the valid input region (matches conv2D convention)
  bool isInBounds${i} = input${i}RealXYZ.x >= 0 && input${i}RealXYZ.x < inputDims.x &&
                        input${i}RealXYZ.y >= 0 && input${i}RealXYZ.y < inputDims.y &&
                        input${i}RealXYZ.z >= 0 && input${i}RealXYZ.z < inputDims.z;

  vec4 result${i};

  if (isInBounds${i}) {
    // Fetch from input texture (matches conv2D flat index formula)
    int input${i}FlatIndex = input${i}RealXYZ.y * inputDims.x * paddedInputDepth +
                             input${i}RealXYZ.x * paddedInputDepth +
                             input${i}RealXYZ.z;
    ivec3 input${i}TextureXYZ = convertFlatToTextureArray3D(input${i}FlatIndex, inputTextureArrayDims.x, inputTextureArrayDims.y, inputTextureSpatialSize);
    result${i} = texelFetch(${input.uniformName}, input${i}TextureXYZ, 0);
  } else {
    // Use constant value for padding
    result${i} = vec4(constantValue);
  }
`;
    }
    return code;
  }()}

  ${finalResultCode}
}
`;
}
