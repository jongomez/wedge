import { isWebGLDataTextureArray } from "../../helpers";
import { fsHeader, outputVarName } from "../../shaderHelpers";
import { ArithmeticOpName, WebGLOpNode } from "../../types";

export const arithmeticWebGLShader = (
  opNode: WebGLOpNode,
  operation: ArithmeticOpName): string => {
  const input1 = opNode.inputs[0];
  const input2 = opNode.inputs[1];

  if (input1?.webGLType === 'sampler2D' || input2?.webGLType === 'sampler2D') {
    throw new Error('Arithmetic operations do not support texture inputs - only texture arrays or float inputs.');
  }

  let inputNumberOfTextures = 0;
  if (isWebGLDataTextureArray(input1)) {
    inputNumberOfTextures = input1.RGBATextureShape[2];
  } else if (isWebGLDataTextureArray(input2)) {
    inputNumberOfTextures = input2.RGBATextureShape[2];
  }

  if (inputNumberOfTextures === 0) {
    throw new Error('Arithmetic operations require at least one input to be a texture array.');
  }

  // if (inputNumberOfTextures !== opNode.output.RGBATextureShape[2]) {
  //   throw new Error('Arithmetic operations require the input and output texture array lengths to match.');
  // }

  // Determine the types of inputs
  const input1IsTextureArray = input1?.webGLType === 'sampler2DArray';
  const input2IsTextureArray = input2?.webGLType === 'sampler2DArray';

  const textureArrayUniform2 = input2IsTextureArray ? `uniform mediump sampler2DArray ${input2?.uniformName};` : '';
  const textureArrayUniform1 = input1IsTextureArray ? `uniform mediump sampler2DArray ${input1?.uniformName};` : '';

  const floatUniform1 = !input1IsTextureArray ? `uniform float ${input1?.uniformName};` : '';
  const floatUniform2 = !input2IsTextureArray ? `uniform float ${input2?.uniformName};` : '';

  // Setting the output square size as a constant. No need for uniform in this case - at least for now.
  // const outputHeight = `const float outputHeight = ${opNode.output.height}.0;`;
  // const outputWidth = `const float outputWidth = ${opNode.output.width}.0;`;
  // const numLayers = `const float numLayers = ${opNode.output.numberOfTextures}.0;`;

  // Determine the shader operation based on the operation argument
  const shaderOperation = (() => {
    switch (operation) {
      case 'AddV2': return '+';
      case 'Mul': return '*';
      default: throw new Error(`Unsupported operation: ${operation}`);
    }
  })();

  const finalNumberOfElements = opNode.output?.RGBATextureElementCount || 0;

  // if (input1IsTextureArray && input1.elementCount !== finalNumberOfElements) {
  //   throw new Error('Input 1 element count does not match output element count');
  // }
  // if (input2IsTextureArray && input2.elementCount !== finalNumberOfElements) {
  //   throw new Error('Input 2 element count does not match output element count');
  // }
  // if (finalNumberOfElements % 4 !== 0) {
  //   throw new Error('Output element count must be a multiple of 4');
  // }

  const finalNumberOfRGBAPositions = finalNumberOfElements / 4;


  return `${fsHeader}

const int finalNumberOfRGBAPositions = ${finalNumberOfRGBAPositions};
const ivec3 input1Dims = ${input1IsTextureArray ? `ivec3(${input1.RGBATextureShape[0]}, ${input1.RGBATextureShape[1]}, ${input1.RGBATextureShape[2]})` : 'ivec3(1, 1, 1)'};
const ivec3 input2Dims = ${input2IsTextureArray ? `ivec3(${input2.RGBATextureShape[0]}, ${input2.RGBATextureShape[1]}, ${input2.RGBATextureShape[2]})` : 'ivec3(1, 1, 1)'};
const int input1SpatialSize = input1Dims.x * input1Dims.y;
const int input2SpatialSize = input2Dims.x * input2Dims.y;

${textureArrayUniform1 || floatUniform1}
${textureArrayUniform2 || floatUniform2}

out mediump vec4 ${outputVarName};

ivec3 convertFlatToTextureArray3D(int flatIndex, int width, int height, int textureSpatialSize) {
  int textureArrayFlatIndexModulo = flatIndex % textureSpatialSize;

  int x = textureArrayFlatIndexModulo % width;
  int y = textureArrayFlatIndexModulo / width;
  int z = flatIndex / textureSpatialSize;

  return ivec3(x, y, z);
}

void main() {
  vec2 outputXY = gl_FragCoord.xy;
  int outputIndex = int(outputXY.x) + int(outputXY.y) * ${opNode.output?.RGBATextureShape[0]};

  ivec3 value1XYZ = ${input1IsTextureArray ? `convertFlatToTextureArray3D(outputIndex, input1Dims.x, input1Dims.y, input1SpatialSize)` : 'ivec3(0, 0, 0)'};
  ivec3 value2XYZ = ${input2IsTextureArray ? `convertFlatToTextureArray3D(outputIndex, input2Dims.x, input2Dims.y, input2SpatialSize)` : 'ivec3(0, 0, 0)'};

  ${input1IsTextureArray ? `vec4 value1 = texelFetch(${input1.uniformName}, value1XYZ, 0)` : `float value1 = ${input1?.uniformName}`};
  ${input2IsTextureArray ? `vec4 value2 = texelFetch(${input2.uniformName}, value2XYZ, 0)` : `float value2 = ${input2?.uniformName}`};

  // Perform the specified operation
  ${outputVarName} = value1 ${shaderOperation} value2;
}
`;
}