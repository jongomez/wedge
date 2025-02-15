import { isWebGLDataTextureArray } from "../../helpers";
import { createRenderTargetFinalCode, createRenderTargetLayoutsCode, fsHeader } from "../../shaderHelpers";
import { SingleInputBasicOpName, WebGLOpNode } from "../../types";

export const singleInputBasicWebGLShader = (opNode: WebGLOpNode, operation: SingleInputBasicOpName): string => {
  if (opNode.inputs.length !== 1) {
    throw new Error(`Node ${opNode.node.name} has ${opNode.inputs.length} inputs, expected 1`);
  }

  const input = opNode.inputs[0];
  const output = opNode.output;

  if (!input) {
    throw new Error("Single input basic operations require an input");
  }

  if (!output) {
    throw new Error("Single input basic operations require an output");
  }

  if (!isWebGLDataTextureArray(input)) {
    throw new Error('Single input basic operations require the input to be a texture array. Got: ' + input.webGLType);
  }

  let inputNumberOfTextures = input.RGBATextureShape[2];
  if (inputNumberOfTextures === 0) {
    throw new Error('Arithmetic operations require at least one input to be a texture array.');
  }

  // Uniforms
  const textureArrayUniform = `uniform mediump sampler2DArray ${input.uniformName};`;

  // Determine the shader operation based on the operation argument
  const shaderOperation = (renderTargetNum: number) => {
    switch (operation) {
      case 'Relu': return `max(value${renderTargetNum}, 0.0)`;  // Relu operation
      default: throw new Error(`Unsupported operation: ${operation}`);
    }
  };

  const numberOfTextures = output.RGBATextureShape[2];
  const renderTargetLayoutsCode = createRenderTargetLayoutsCode(numberOfTextures);
  const finalResultCode = createRenderTargetFinalCode(numberOfTextures);

  const constants = `int outputWidth = ${output.RGBATextureShape[0]};
int outputHeight = ${output.RGBATextureShape[1]};`;

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
  ivec3 inputXYZ${renderTargetNum} = ivec3(coords, ${renderTargetNum});
  vec4 value${renderTargetNum} = texelFetch(${input.uniformName}, inputXYZ${renderTargetNum}, 0);

  // Perform the specified operation
  vec4 result${renderTargetNum} = ${shaderOperation(renderTargetNum)};
        `
      }
      return result;
    }()}
  
  ${finalResultCode}
}
`;
}