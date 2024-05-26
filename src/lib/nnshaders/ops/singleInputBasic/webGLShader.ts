import { createRenderTargetFinalCode, createRenderTargetLayoutsCode, fsHeader } from "../../shaderHelpers";
import { OpNodeWithWebGLData, SingleInputBasicOpName } from "../../types";

export const singleInputBasicWebGLShader = (opNode: OpNodeWithWebGLData, operation: SingleInputBasicOpName): string => {
  if (opNode.inputs.length !== 1) {
    throw new Error(`Node ${opNode.node.name} has ${opNode.inputs.length} inputs, expected 1`);
  }

  if (opNode.inputs[0].type !== 'sampler2DArray') {
    throw new Error('Single input basic operations require the input to be a texture array. Got: ' + opNode.inputs[0].type);
  }

  let inputNumberOfTextures = opNode.inputs[0].numberOfTextures;
  if (inputNumberOfTextures === 0) {
    throw new Error('Arithmetic operations require at least one input to be a texture array.');
  }

  // Uniforms
  const textureArrayUniform = `uniform mediump sampler2DArray ${opNode.inputs[0].uniformName};`;

  // Setting the output dimensions as constants
  const outputHeight = `const float outputHeight = ${opNode.output.height}.0;`;
  const outputWidth = `const float outputWidth = ${opNode.output.width}.0;`;

  // Determine the shader operation based on the operation argument
  const shaderOperation = (() => {
    switch (operation) {
      case 'Relu': return 'max(value, 0.0)';  // Relu operation
      default: throw new Error(`Unsupported operation: ${operation}`);
    }
  })();

  const numberOfTextures = opNode.output.numberOfTextures;
  const renderTargetLayoutsCode = createRenderTargetLayoutsCode(numberOfTextures);
  const finalResultCode = createRenderTargetFinalCode(numberOfTextures);

  const constants = `int outputHeight = ${opNode.output.height};
int outputWidth = ${opNode.output.width};`;

  return `${fsHeader}

${textureArrayUniform}

${constants}

${renderTargetLayoutsCode}

void main() {
  vec2 coords = gl_FragCoord.xy;
  
  ${function () {
      let result = '';
      for (let renderTargetNum = 0; renderTargetNum < numberOfTextures; renderTargetNum++) {
        result += `
        ivec3 inputXYZ = ivec3(coords, ${renderTargetNum});
        vec4 value = texelFetch(${opNode.inputs[0].uniformName}, inputXYZ, 0);

        // Perform the specified operation
        vec4 result${renderTargetNum} = ${shaderOperation};
        `
      }
      return result;
    }()}
  
  ${finalResultCode}
}
`;
}