import { isWebGLDataTextureArray } from "../../helpers";
import { createRenderTargetFinalCode, createRenderTargetLayoutsCode, fsHeader } from "../../shaderHelpers";
import { MaxPoolParams, WebGLOpNode } from "../../types";

export const maxPoolWebGLShader = (opNode: WebGLOpNode): string => {
  if (opNode.inputs.length !== 1) {
    throw new Error(`MaxPool node ${opNode.node.name} has ${opNode.inputs.length} inputs, expected 1`);
  }

  const input = opNode.inputs[0];
  const output = opNode.output;
  const params = opNode.opParams as MaxPoolParams;

  if (!input || !output || !params) {
    throw new Error("MaxPool requires input, output, and params");
  }

  if (!isWebGLDataTextureArray(input)) {
    throw new Error('MaxPool requires the input to be a texture array. Got: ' + input.webGLType);
  }

  const { poolSize, strides, pad } = params;

  // Handle 4D input shapes (N, H, W, C)
  const inputIs4D = input.originalShape.length === 4;
  const inputH = inputIs4D ? input.originalShape[1] : input.originalShape[0];
  const inputW = inputIs4D ? input.originalShape[2] : input.originalShape[1];
  const inputC = inputIs4D ? input.originalShape[3] : input.originalShape[2];

  const outputIs4D = output.originalShape.length === 4;
  const outputH = outputIs4D ? output.originalShape[1] : output.originalShape[0];
  const outputW = outputIs4D ? output.originalShape[2] : output.originalShape[1];
  const outputC = outputIs4D ? output.originalShape[3] : output.originalShape[2];

  // Pad channels to multiple of 4 for RGBA texture storage
  const paddedInputC = Math.ceil(inputC / 4) * 4;
  const paddedOutputC = Math.ceil(outputC / 4) * 4;

  // Calculate padding for 'same' mode
  let padTop = 0;
  let padLeft = 0;

  if (pad === 'same') {
    const padH = Math.max(0, (outputH - 1) * strides[0] + poolSize[0] - inputH);
    const padW = Math.max(0, (outputW - 1) * strides[1] + poolSize[1] - inputW);
    padTop = Math.floor(padH / 2);
    padLeft = Math.floor(padW / 2);
  }

  const numberOfTextures = output.RGBATextureShape[2];
  const renderTargetLayoutsCode = createRenderTargetLayoutsCode(numberOfTextures);
  const finalResultCode = createRenderTargetFinalCode(numberOfTextures);

  const uniforms = `uniform mediump sampler2DArray ${input.uniformName};`;

  const constants = `
const ivec3 inputDims = ivec3(${inputH}, ${inputW}, ${inputC});
const ivec3 outputDims = ivec3(${outputH}, ${outputW}, ${outputC});
const int paddedInputDepth = ${paddedInputC};
const int paddedOutputDepth = ${paddedOutputC};
const ivec2 poolSize = ivec2(${poolSize[0]}, ${poolSize[1]});
const ivec2 strides = ivec2(${strides[0]}, ${strides[1]});
const ivec2 padOffset = ivec2(${padTop}, ${padLeft});
const int inputTextureWidth = ${input.RGBATextureShape[0]};
const int inputTextureHeight = ${input.RGBATextureShape[1]};
const int outputTextureWidth = ${output.RGBATextureShape[0]};
const int outputTextureHeight = ${output.RGBATextureShape[1]};
`;

  // Helper functions for coordinate conversion
  const helperFunctions = `
// Convert flat index to HWC coordinates
ivec3 convertFlatToHWC3D(int flatIndex, int height, int width, int depth) {
  int z = flatIndex % depth;
  int temp = flatIndex / depth;
  int x = temp % width;
  int y = temp / width;
  return ivec3(x, y, z);
}

// Convert HWC coordinates to flat index
int convertHWC3DToFlat(ivec3 hwc, int height, int width, int depth) {
  return hwc.y * width * depth + hwc.x * depth + hwc.z;
}

// Convert flat index to texture array coordinates
ivec3 convertFlatToTextureArray3D(int flatIndex, int textureWidth, int textureHeight) {
  int pixelIndex = flatIndex / 4;
  int textureIndex = pixelIndex / (textureWidth * textureHeight);
  int localPixelIndex = pixelIndex % (textureWidth * textureHeight);
  int x = localPixelIndex % textureWidth;
  int y = localPixelIndex / textureWidth;
  return ivec3(x, y, textureIndex);
}
`;

  return `${fsHeader}

${uniforms}

${constants}

${renderTargetLayoutsCode}

${helperFunctions}

void main() {
  vec2 coords = gl_FragCoord.xy;
  int texelX = int(coords.x);
  int texelY = int(coords.y);

  ${function () {
    let result = '';
    for (let renderTargetNum = 0; renderTargetNum < numberOfTextures; renderTargetNum++) {
      const flatOffset = renderTargetNum * output.RGBATextureShape[0] * output.RGBATextureShape[1] * 4;

      result += `
  // Render target ${renderTargetNum}
  int outputFlatIndex${renderTargetNum} = (texelY * outputTextureWidth + texelX) * 4 + ${flatOffset};

  vec4 result${renderTargetNum} = vec4(-1.0e38); // Initialize to very small value for max operation

  // Process 4 channels (RGBA) per texel
  for (int channelOffset = 0; channelOffset < 4; channelOffset++) {
    int currentFlatIndex = outputFlatIndex${renderTargetNum} + channelOffset;

    // Convert flat index to output HWC coordinates
    // Convention (from Pad shader): .x = w (width), .y = h (height), .z = c (channel)
    ivec3 outputHWC = convertFlatToHWC3D(currentFlatIndex, outputDims.x, outputDims.y, paddedOutputDepth);
    int outH = outputHWC.y;  // .y is height coordinate
    int outW = outputHWC.x;  // .x is width coordinate
    int outC = outputHWC.z;  // .z is channel

    // Check if this is a valid output position (not padding)
    if (outH >= outputDims.x || outW >= outputDims.y || outC >= outputDims.z) {
      if (channelOffset == 0) result${renderTargetNum}.x = 0.0;
      else if (channelOffset == 1) result${renderTargetNum}.y = 0.0;
      else if (channelOffset == 2) result${renderTargetNum}.z = 0.0;
      else result${renderTargetNum}.w = 0.0;
      continue;
    }

    // Calculate input window start position
    int inputStartH = outH * strides.x - padOffset.x;
    int inputStartW = outW * strides.y - padOffset.y;

    float maxVal = -1.0e38;

    // Iterate over the pooling window
    for (int ph = 0; ph < poolSize.x; ph++) {
      for (int pw = 0; pw < poolSize.y; pw++) {
        int inputH = inputStartH + ph;
        int inputW = inputStartW + pw;

        // Check bounds
        if (inputH >= 0 && inputH < inputDims.x && inputW >= 0 && inputW < inputDims.y) {
          // Convert to flat index for input (convention: h * H * paddedC + w * paddedC + c)
          int inputFlatIndex = inputH * inputDims.x * paddedInputDepth + inputW * paddedInputDepth + outC;

          // Convert to texture coordinates
          ivec3 inputTexCoord = convertFlatToTextureArray3D(inputFlatIndex, inputTextureWidth, inputTextureHeight);
          int componentIndex = inputFlatIndex % 4;

          vec4 texelValue = texelFetch(${input.uniformName}, inputTexCoord, 0);
          float val;
          if (componentIndex == 0) val = texelValue.x;
          else if (componentIndex == 1) val = texelValue.y;
          else if (componentIndex == 2) val = texelValue.z;
          else val = texelValue.w;

          maxVal = max(maxVal, val);
        }
      }
    }

    // Handle case where no valid input was found (shouldn't happen in valid configs)
    if (maxVal < -1.0e37) maxVal = 0.0;

    if (channelOffset == 0) result${renderTargetNum}.x = maxVal;
    else if (channelOffset == 1) result${renderTargetNum}.y = maxVal;
    else if (channelOffset == 2) result${renderTargetNum}.z = maxVal;
    else result${renderTargetNum}.w = maxVal;
  }
`;
    }
    return result;
  }()}

  ${finalResultCode}
}
`;
};
