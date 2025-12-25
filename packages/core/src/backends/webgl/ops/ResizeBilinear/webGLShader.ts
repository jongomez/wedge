import { isWebGLDataTextureArray } from "../../helpers";
import { createRenderTargetFinalCode, createRenderTargetLayoutsCode, fsHeader } from "../../shaderHelpers";
import { ResizeBilinearParams, WebGLOpNode } from "../../types";

export const resizeBilinearWebGLShader = (opNode: WebGLOpNode): string => {
  const input = opNode.inputs[0];
  const output = opNode.output;
  const params = opNode.opParams as ResizeBilinearParams;

  if (!input) {
    throw new Error("ResizeBilinear operation requires an input");
  }

  if (!output) {
    throw new Error("ResizeBilinear operation requires an output");
  }

  if (!isWebGLDataTextureArray(input)) {
    throw new Error('ResizeBilinear requires the input to be a texture array. Got: ' + input.webGLType);
  }

  // Input and output dimensions (HWC format, assuming batch dimension removed)
  const inputH = input.originalShape.length === 4 ? input.originalShape[1] : input.originalShape[0];
  const inputW = input.originalShape.length === 4 ? input.originalShape[2] : input.originalShape[1];
  const outputH = output.originalShape.length === 4 ? output.originalShape[1] : output.originalShape[0];
  const outputW = output.originalShape.length === 4 ? output.originalShape[2] : output.originalShape[1];

  // Uniforms
  const textureArrayUniform = `uniform mediump sampler2DArray ${input.uniformName};`;

  const numberOfTextures = output.RGBATextureShape[2];
  const renderTargetLayoutsCode = createRenderTargetLayoutsCode(numberOfTextures);
  const finalResultCode = createRenderTargetFinalCode(numberOfTextures);

  // Scale factors for coordinate calculation
  // With halfPixelCenters=true: src = (dst + 0.5) * (srcSize / dstSize) - 0.5
  const scaleH = inputH / outputH;
  const scaleW = inputW / outputW;

  const constants = `
const int inputH = ${inputH};
const int inputW = ${inputW};
const int outputH = ${outputH};
const int outputW = ${outputW};
const float scaleH = ${scaleH.toFixed(10)};
const float scaleW = ${scaleW.toFixed(10)};
const int inputTextureW = ${input.RGBATextureShape[0]};
const int inputTextureH = ${input.RGBATextureShape[1]};
`;

  return `${fsHeader}

${textureArrayUniform}

${constants}

${renderTargetLayoutsCode}

// Convert output (x, y) to input source coordinates using halfPixelCenters formula
vec2 getSourceCoords(vec2 outputCoords) {
  // halfPixelCenters=true: src = (dst + 0.5) * scale - 0.5
  float srcY = (outputCoords.y + 0.5) * scaleH - 0.5;
  float srcX = (outputCoords.x + 0.5) * scaleW - 0.5;
  return vec2(srcX, srcY);
}

// Bilinear interpolation between 4 pixels
vec4 bilinearSample(int textureLayer, vec2 srcCoords) {
  // Clamp source coordinates to valid range
  float srcX = clamp(srcCoords.x, 0.0, float(inputW - 1));
  float srcY = clamp(srcCoords.y, 0.0, float(inputH - 1));

  // Get the 4 neighboring pixels
  int x0 = int(floor(srcX));
  int x1 = min(x0 + 1, inputW - 1);
  int y0 = int(floor(srcY));
  int y1 = min(y0 + 1, inputH - 1);

  // Interpolation weights
  float xLerp = srcX - float(x0);
  float yLerp = srcY - float(y0);

  // Fetch the 4 neighboring texels
  vec4 topLeft = texelFetch(${input.uniformName}, ivec3(x0, y0, textureLayer), 0);
  vec4 topRight = texelFetch(${input.uniformName}, ivec3(x1, y0, textureLayer), 0);
  vec4 bottomLeft = texelFetch(${input.uniformName}, ivec3(x0, y1, textureLayer), 0);
  vec4 bottomRight = texelFetch(${input.uniformName}, ivec3(x1, y1, textureLayer), 0);

  // Bilinear interpolation
  vec4 top = mix(topLeft, topRight, xLerp);
  vec4 bottom = mix(bottomLeft, bottomRight, xLerp);
  return mix(top, bottom, yLerp);
}

void main() {
  vec2 outCoords = gl_FragCoord.xy;

  // Calculate source coordinates for this output pixel
  vec2 srcCoords = getSourceCoords(outCoords);

  ${function () {
      let result = '';
      for (let renderTargetNum = 0; renderTargetNum < numberOfTextures; renderTargetNum++) {
        result += `
  // Sample from texture layer ${renderTargetNum}
  vec4 result${renderTargetNum} = bilinearSample(${renderTargetNum}, srcCoords);
`;
      }
      return result;
    }()}

  ${finalResultCode}
}
`;
};
