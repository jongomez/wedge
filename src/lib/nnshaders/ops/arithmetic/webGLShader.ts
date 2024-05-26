import { isWebGLDataTextureArray } from "../../helpers";
import { fsHeader, outputVarName } from "../../shaderHelpers";
import { ArithmeticOpName, OpNodeWithWebGLData } from "../../types";

export const arithmeticWebGLShader = (opNode: OpNodeWithWebGLData, operation: ArithmeticOpName): string => {
  if (opNode.inputs[0].type === 'sampler2D' || opNode.inputs[1].type === 'sampler2D') {
    throw new Error('Arithmetic operations do not support texture inputs - only texture arrays or float inputs.');
  }

  let inputNumberOfTextures = 0;
  if (isWebGLDataTextureArray(opNode.inputs[0])) {
    inputNumberOfTextures = opNode.inputs[0].numberOfTextures;
  } else if (isWebGLDataTextureArray(opNode.inputs[1])) {
    inputNumberOfTextures = opNode.inputs[1].numberOfTextures;
  }

  if (inputNumberOfTextures === 0) {
    throw new Error('Arithmetic operations require at least one input to be a texture array.');
  }

  // if (inputNumberOfTextures !== opNode.output.numberOfTextures) {
  //   throw new Error('Arithmetic operations require the input and output texture array lengths to match.');
  // }

  // Determine the types of inputs
  const input1IsTextureArray = opNode.inputs[0].type === 'sampler2DArray';
  const input2IsTextureArray = opNode.inputs[1].type === 'sampler2DArray';

  const textureArrayUniform2 = input2IsTextureArray ? `uniform mediump sampler2DArray ${opNode.inputs[1].uniformName};` : '';
  const textureArrayUniform1 = input1IsTextureArray ? `uniform mediump sampler2DArray ${opNode.inputs[0].uniformName};` : '';

  const floatUniform1 = !input1IsTextureArray ? `uniform float ${opNode.inputs[0].uniformName};` : '';
  const floatUniform2 = !input2IsTextureArray ? `uniform float ${opNode.inputs[1].uniformName};` : '';

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

  const finalNumberOfElements = opNode.output.elementCount;

  // if (input1IsTextureArray && opNode.inputs[0].elementCount !== finalNumberOfElements) {
  //   throw new Error('Input 1 element count does not match output element count');
  // }
  // if (input2IsTextureArray && opNode.inputs[1].elementCount !== finalNumberOfElements) {
  //   throw new Error('Input 2 element count does not match output element count');
  // }
  // if (finalNumberOfElements % 4 !== 0) {
  //   throw new Error('Output element count must be a multiple of 4');
  // }

  const finalNumberOfRGBAPositions = finalNumberOfElements / 4;


  return `${fsHeader}

const int finalNumberOfRGBAPositions = ${finalNumberOfRGBAPositions};
const ivec3 input1Dims = ${input1IsTextureArray ? `ivec3(${opNode.inputs[0].shape[0]}, ${opNode.inputs[0].shape[1]}, ${opNode.inputs[0].shape[2]})` : 'ivec3(1, 1, 1)'};
const ivec3 input2Dims = ${input2IsTextureArray ? `ivec3(${opNode.inputs[1].shape[0]}, ${opNode.inputs[1].shape[1]}, ${opNode.inputs[1].shape[2]})` : 'ivec3(1, 1, 1)'};
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
  int outputIndex = int(outputXY.x) + int(outputXY.y) * ${opNode.output.width};

  ivec3 value1XYZ = ${input1IsTextureArray ? `convertFlatToTextureArray3D(outputIndex, input1Dims.x, input1Dims.y, input1SpatialSize)` : 'ivec3(0, 0, 0)'};
  ivec3 value2XYZ = ${input2IsTextureArray ? `convertFlatToTextureArray3D(outputIndex, input2Dims.x, input2Dims.y, input2SpatialSize)` : 'ivec3(0, 0, 0)'};

  vec4 value1 = ${input1IsTextureArray ? `texelFetch(${opNode.inputs[0].uniformName}, value1XYZ, 0)` : `${opNode.inputs[0].uniformName}`};
  vec4 value2 = ${input2IsTextureArray ? `texelFetch(${opNode.inputs[1].uniformName}, value2XYZ, 0)` : `${opNode.inputs[1].uniformName}`};

  // Perform the specified operation
  ${outputVarName} = value1 ${shaderOperation} value2;
}
`;
}