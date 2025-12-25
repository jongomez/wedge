import { isWebGLDataTextureArray } from "../../helpers";
import { createRenderTargetFinalCode, createRenderTargetLayoutsCode, fsHeader } from "../../shaderHelpers";
import { ReshapeParams, WebGLOpNode } from "../../types";

export const reshapeWebGLShader = (opNode: WebGLOpNode): string => {
  if (opNode.inputs.length !== 1) {
    throw new Error(`Reshape operation expects 1 input, got ${opNode.inputs.length}`);
  }

  const input = opNode.inputs[0];
  const output = opNode.output;
  const params = opNode.opParams as ReshapeParams;

  if (!input) {
    throw new Error("Reshape operation requires an input");
  }

  if (!output) {
    throw new Error("Reshape operation requires an output");
  }

  if (!isWebGLDataTextureArray(input)) {
    throw new Error("Reshape operation requires input to be a texture array. Got: " + input.webGLType);
  }

  const inputNumberOfTextures = input.RGBATextureShape[2];
  if (inputNumberOfTextures === 0) {
    throw new Error("Reshape operation requires at least one input texture.");
  }

  // Uniforms
  const textureArrayUniform = `uniform mediump sampler2DArray ${input.uniformName};`;

  const numberOfTextures = output.RGBATextureShape[2];
  const renderTargetLayoutsCode = createRenderTargetLayoutsCode(numberOfTextures);
  const finalResultCode = createRenderTargetFinalCode(numberOfTextures);

  // Handle both 3D [H, W, C] and 4D [N, H, W, C] input shapes
  const inputIs4D = input.originalShape.length === 4;
  const inputH = inputIs4D ? input.originalShape[1] : input.originalShape[0];
  const inputW = inputIs4D ? input.originalShape[2] : input.originalShape[1];
  const inputC = inputIs4D ? input.originalShape[3] : input.originalShape[2];

  // Handle both 3D [H, W, C] and 4D [N, H, W, C] output shapes
  const outputIs4D = output.originalShape.length === 4;
  const outputH = outputIs4D ? output.originalShape[1] : output.originalShape[0];
  const outputW = outputIs4D ? output.originalShape[2] : output.originalShape[1];
  const outputC = outputIs4D ? output.originalShape[3] : output.originalShape[2];

  const constants = `
const ivec3 inputDims = ivec3(${inputH}, ${inputW}, ${inputC});
const ivec3 inputTextureArrayDims = ivec3(${input.RGBATextureShape[0]}, ${input.RGBATextureShape[1]}, ${input.RGBATextureShape[2]});
const ivec3 outputDims = ivec3(${outputH}, ${outputW}, ${outputC});
const ivec3 outputTextureArrayDims = ivec3(${output.RGBATextureShape[0]}, ${output.RGBATextureShape[1]}, ${output.RGBATextureShape[2]});

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

// Fetch a single value from input texture at the given flat padded index
float fetchInputValue(int flatPaddedIndex) {
  ivec3 texCoord = convertFlatToTextureArray3D(flatPaddedIndex, inputTextureArrayDims.x, inputTextureArrayDims.y, inputTextureSpatialSize);
  vec4 texel = texelFetch(${input.uniformName}, texCoord, 0);
  int component = flatPaddedIndex - (flatPaddedIndex / 4) * 4;
  if (component == 0) return texel.r;
  if (component == 1) return texel.g;
  if (component == 2) return texel.b;
  return texel.a;
}

void main() {
  // Each output spatial xy position outputs 4 values (RGBA)
  // outputCoordFlat is the base flat index in padded HWC layout
  int outputCoordFlat = (int(gl_FragCoord.x) + int(gl_FragCoord.y) * outputTextureArrayDims.x) * 4;

  ${function() {
    let code = '';
    for (let i = 0; i < numberOfTextures; i++) {
      const flatCoordOffset = i * output.RGBATextureShape[0] * output.RGBATextureShape[1] * 4;
      code += `
  // Render target ${i}
  vec4 result${i};

  // Process each of the 4 RGBA channels separately
  // For reshape, consecutive output channels may map to non-consecutive input positions
`;
      // Generate code for each of the 4 channels
      for (let ch = 0; ch < 4; ch++) {
        const component = ['r', 'g', 'b', 'a'][ch];
        code += `
  {
    int outPaddedFlat${ch} = outputCoordFlat + ${flatCoordOffset} + ${ch};

    // Convert padded flat index to HWC coordinates
    // paddedFlat = h * W * paddedC + w * paddedC + c
    int outC${ch} = outPaddedFlat${ch} % paddedOutputDepth;
    int outW${ch} = (outPaddedFlat${ch} / paddedOutputDepth) % outputDims.y;
    int outH${ch} = outPaddedFlat${ch} / (paddedOutputDepth * outputDims.y);

    // Check if this output position is valid (within actual output bounds)
    bool inBounds${ch} = outH${ch} < outputDims.x && outW${ch} < outputDims.y && outC${ch} < outputDims.z;

    if (inBounds${ch}) {
      // Compute the DATA flat index (using actual dimensions, not padded)
      // dataFlat = h * W * C + w * C + c
      int dataFlat${ch} = outH${ch} * outputDims.y * outputDims.z + outW${ch} * outputDims.z + outC${ch};

      // For reshape, the data flat index is preserved - map to input coordinates
      // dataFlat = inH * inW * inC + inW * inC + inC
      int inC${ch} = dataFlat${ch} % inputDims.z;
      int inW${ch} = (dataFlat${ch} / inputDims.z) % inputDims.y;
      int inH${ch} = dataFlat${ch} / (inputDims.z * inputDims.y);

      // Convert to input padded flat index
      // paddedFlat = h * W * paddedC + w * paddedC + c
      int inPaddedFlat${ch} = inH${ch} * inputDims.y * paddedInputDepth + inW${ch} * paddedInputDepth + inC${ch};

      result${i}.${component} = fetchInputValue(inPaddedFlat${ch});
    } else {
      result${i}.${component} = 0.0;
    }
  }
`;
      }
    }
    return code;
  }()}

  ${finalResultCode}
}
`;
}
