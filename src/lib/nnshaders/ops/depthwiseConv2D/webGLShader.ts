import { loop } from "../../helpers";
import { createActivationCode, createAllOutputRealXYZCode, createInitResultsCode, createRenderTargetFinalCode, createRenderTargetLayoutsCode, fsHeader, resultVarName, weightsVarName } from "../../shaderHelpers";
import { Conv2DParams, WebGLDataTextureArray, WebGLOpNode } from "../../types";
import { getConvPadding } from "../conv2D/helpers";

export const depthwiseConv2DWebGLShader = (opNode: WebGLOpNode): string => {
  const {
    strides,
    pad,
    kernelX,
    kernelY,
    kernelDepth,
    numFilters,
    activation
  } = opNode.opParams as Conv2DParams;

  const input = opNode.inputs[0] as WebGLDataTextureArray;
  const weights = opNode.weights;
  const output = opNode.output;

  if (!weights || weights.length < 1) {
    throw new Error("Depthwise conv2D operation must have kernel weights");
  }

  if (!output) {
    throw new Error("Depthwise conv2D operation must have an output");
  }

  const strideX = strides[0];
  const strideY = strides[1];
  const [padX, padY] = getConvPadding(output, input, kernelX, kernelY, strideX, strideY, pad);

  const inputUniform = `uniform mediump sampler2DArray ${input.uniformName};`;
  const weightsUniform = `uniform mediump sampler2DArray ${weights[0].uniformName};`
  const biasWeightsUniform = weights.length > 1 ? `uniform mediump sampler2DArray ${weights[1].uniformName};` : '';

  const outputNumberOfTextures = output.RGBATextureShape[2];

  let biasAddCode = "";
  if (weights.length > 1) {
    biasAddCode = `// NOTE: target0Output0RealXYZ.z is the 1st filter's index. The following filters should be sequential. 
  ivec3 biasXYZ = ivec3(target0Output0RealXYZ.z / 4, 0, 0); // assuming the bias texture is 1D, with values in the x direction.
  vec4 biasValues = texelFetch(${weights[1].uniformName}, biasXYZ, 0);`;

    for (let i = 0; i < outputNumberOfTextures; i++) {
      biasAddCode += `
  ${resultVarName}${i} += biasValues;`;
    }
  }

  const paddedKernelDepth = Math.ceil(kernelDepth / 4) * 4;
  const paddedOutputDepth = Math.ceil(output.originalShape[2] / 4) * 4;
  const paddedInputDepth = Math.ceil(input.originalShape[2] / 4) * 4;
  // const singlePaddedKernelElementCount = kernelX * kernelY * paddedKernelDepth;
  // const allPaddedKernelsElementCount = singlePaddedKernelElementCount * numFilters;

  // The kernel weights are concatted in the x direction.
  // So if we multiply the kernel's x dimension by the number of filters, we should get the total width of the kernel texture.
  if (kernelX * numFilters != weights[0].RGBATextureShape[0]) {
    throw new Error("kernelX * numFilters != weights[0].RGBATextureShape[0]");
  }

  const hasBias = weights.length > 1;

  const constants = `const ivec3 inputDims = ivec3(${input.originalShape[0]}, ${input.originalShape[1]}, ${input.originalShape[2]}); 
const ivec3 inputTextureArrayDims = ivec3(${input.RGBATextureShape[0]}, ${input.RGBATextureShape[1]}, ${input.RGBATextureShape[2]});
const ivec3 paddedKernelDims = ivec3(${kernelX}, ${kernelY}, ${paddedKernelDepth});
const ivec3 singleKernelTextureDims = ivec3(${weights[0].RGBATextureShape[0]}, ${weights[0].RGBATextureShape[1]}, ${weights[0].RGBATextureShape[2] / numFilters});
const ivec3 biasTextureDims = ivec3(${hasBias ? weights[1].RGBATextureShape[0] : 0}, ${hasBias ? weights[1].RGBATextureShape[1] : 0}, 1);
    
// XXX: NOTE: outputDepth MAY BE different from the number of filters.
// This is because, when creating the output texture array, the textures are channel padded to be divisible by 4.
const ivec3 outputDims = ivec3(${output.originalShape[0]}, ${output.originalShape[1]}, ${output.originalShape[2]});
const ivec3 outputTextureArrayDims = ivec3(${output.RGBATextureShape[0]}, ${output.RGBATextureShape[1]}, ${output.RGBATextureShape[2]});

const ivec2 strides = ivec2(${strides[0]}, ${strides[1]});
const ivec2 padding = ivec2(${padX}, ${padY});

const int inputTextureSpatialSize = inputTextureArrayDims.x * inputTextureArrayDims.y;
const int numFilters = ${numFilters};
const int paddedOutputDepth = ${paddedOutputDepth};
const int paddedInputDepth = ${paddedInputDepth};
`;

  return `${fsHeader}
  
${inputUniform}
${weightsUniform}
${biasWeightsUniform}

${constants}
${createRenderTargetLayoutsCode(outputNumberOfTextures)}

ivec3 convertFlatToTextureArray3D(int flatIndex, int width, int height, int textureSpatialSize) {
  int textureArrayFlatIndex = flatIndex / 4;
  int textureArrayFlatIndexModulo = textureArrayFlatIndex % textureSpatialSize;

  int x = textureArrayFlatIndexModulo % width;
  int y = textureArrayFlatIndexModulo / width;
  int z = textureArrayFlatIndex / textureSpatialSize;

  return ivec3(x, y, z);
}

// The following convertFlatHWCTo3D function is for flat data arranged as follows (HWC format):
// [(0,0,0), ..., (0,0,depth-1), (0,1,0), ..., (0,1,depth-1), ..., (height-1, width-1, depth-1)]
ivec3 convertFlatToHWC3D(int flatIndex, int width, int height, int depth) {
  int z = flatIndex % depth;
  int x = flatIndex / depth % width;
  int y = flatIndex / (depth * width);

  return ivec3(x, y, z);
}

/*

There are basically 3 spaces to consider:
1 - Real space - this is where the real x y and z values live. 
  - The input / output shape values are used here.
2 - Texture space - this is where the texture coordinates live. 
  - The textures have 4 values per pixel. So if texture array depth is 4, then the real depth is 4 * texture array depth.
3 - Flat space - Before converting to texture space, the real data from real space is flattened to a 1D array.
  - Note: the flat indices should be in real coordinates - unless the var name specifies otherwise.

*/

void main() {
  // Each output spatial xy position can output 4 * outputNumberOfTextures different values.
  int outputCoordFlat = (int(gl_FragCoord.x) + int(gl_FragCoord.y) * outputTextureArrayDims.x) * 4;

  // Calculate the real 3D coordinates from the flat index
  ${createAllOutputRealXYZCode(outputNumberOfTextures, output)}
  ${createInitResultsCode(outputNumberOfTextures)}

  int weightsZ = target0Output0RealXYZ.z/4;

  for(int x = 0; x < paddedKernelDims.x; x++) {
    ${loop(outputNumberOfTextures, (i: number) => `
    int input${i}RealX = target${i}Output0RealXYZ.x * strides.x - padding.x + x;
    int input${i}FlatXOffset = input${i}RealX * paddedInputDepth;
    bool is${i}InBoundsX = input${i}RealX >= 0 && input${i}RealX < inputDims.x;
    `)}

    for(int y = 0; y < paddedKernelDims.y; y++) {
      ${loop(outputNumberOfTextures, (i: number) => `
      int input${i}RealY = target${i}Output0RealXYZ.y * strides.y - padding.y + y;
      int input${i}FlatIndex = input${i}RealY * inputDims.x * paddedInputDepth + input${i}FlatXOffset;
      float is${i}InBounds = float(is${i}InBoundsX && input${i}RealY >= 0 && input${i}RealY < inputDims.y);
      `)}

      // Step 1 - Fetch the weights. These weights will be applied across multiple input values.
      // XXX: These weight fetches should belong to different filters. The output ALWAYS HAS the result of 4 filters.
      // If outputNumberOfTextures is > 1, we will apply these filters to different input values.
      vec4 ${weightsVarName} = texelFetch(${weights[0].uniformName}, ivec3(x, y, weightsZ), 0);

      // Step 2 - Fetch the inputs and multiply them by the weights.
      // Each render target block should output 4 values. 
      // numFilters should be a multiple of 4, so each value will belong to a different conv filter. 
      // This is because we're working with channels last data - HWC format.

      ${loop(outputNumberOfTextures, (i: number) => `
      ivec3 input${i}TextureXYZ = convertFlatToTextureArray3D(input${i}FlatIndex + weightsZ*4, inputTextureArrayDims.x, inputTextureArrayDims.y, inputTextureSpatialSize);
      vec4 input${i}Values = is${i}InBounds * texelFetch(${input.uniformName}, input${i}TextureXYZ, 0);

      ${resultVarName}${i} += input${i}Values * ${weightsVarName};

      // For debugging:
      // result0 = ${weightsVarName};
      // break;
      `)}
    }
  // break;
  }

  ${biasAddCode}
  ${activation ? createActivationCode(outputNumberOfTextures, activation) : ""}
  ${createRenderTargetFinalCode(outputNumberOfTextures)}
}`;
}