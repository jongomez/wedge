import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TensorType = any;
import { maxTextureDim } from "../../constants";
import { findRecommendedRenderTargets, isWebGLDataTextureArray } from "./helpers";
import { getChannelPaddedShape } from "../../transforms";
import { ProgramInfo, WebGLDataTexture, WebGLDataTextureArray, WedgeOptions } from "./types";
import { CustomShapeUpdate, DataFormat } from "../../types";

export function initVertexShaderBuffer(gl: WebGL2RenderingContext): WebGLBuffer {
  // Create a buffer for the square's positions.
  var positionBuffer = gl.createBuffer();

  if (!positionBuffer) {
    throw new Error("Error creating positionBuffer");
  }

  // Select the positionBuffer as the one to apply buffer operations to from here out.
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

  var positions = [
    // First triangle
    -1.0, 1.0,  // Top left
    -1.0, -1.0, // Bottom left
    1.0, -1.0,  // Bottom right
    // Second triangle
    -1.0, 1.0,  // Top left (repeated from first triangle)
    1.0, -1.0,  // Bottom right (repeated from first triangle)
    1.0, 1.0    // Top right
  ];

  // Now pass the list of positions into WebGL to build the shape. We do this by creating a Float32Array from the JavaScript array,
  // then use it to fill the current buffer.
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  return positionBuffer;
}

// This is the output from the WebGL program - framebuffers linked to texture arrays.
export function updateFramebufferTextureLayer(
  gl: WebGL2RenderingContext,
  textureArrayData: WebGLDataTextureArray): void {
  const numberOfTextures = textureArrayData.RGBATextureShape[2];

  let attachments = [];

  for (let i = 0; i < 8; i++) {
    // Unbind all textures from the framebuffer. Prevents the following warning / error:
    // GL_INVALID_FRAMEBUFFER_OPERATION: Framebuffer is incomplete: Attachments are not all the same size.
    // TODO (maybe?): would it be better to have multiple framebuffers, one for each output texture?
    // XXX: Multiple framebuffers would require multiple bindFramebuffer calls. Is it worth it?
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, null, 0);
  }

  for (let layerIndex = 0; layerIndex < numberOfTextures; layerIndex++) {
    let attachmentPoint = gl.COLOR_ATTACHMENT0 + layerIndex;

    // Bind each layer to a different color attachment
    gl.framebufferTextureLayer(
      gl.FRAMEBUFFER,
      attachmentPoint,
      textureArrayData.texture,
      0,  // mipmap level
      layerIndex
    );

    attachments.push(attachmentPoint);
  }

  // Tell WebGL the destinations to which we will draw.
  // XXX: Is it 100% necessary to call this every time?
  gl.drawBuffers(attachments);

  // if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
  //   throw new Error("Error: Framebuffer is not complete.");
  // }

}

// function createDummyImage(width: number, height: number): ImageData {
//   const buffer = new ArrayBuffer(width * height * 4);
//   const ui8ca = new Uint8ClampedArray(buffer);
//   const imageData = new ImageData(ui8ca, width, height);
//   return imageData;
// }

export function getElementCount(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export function convertShapeToTexture2DShape(shape: number[], nodeName: string, divideBy = 1): [number, number, number, number] {
  if (shape.length === 0) {
    throw new Error("Shape must have at least one dimension. Node name: " + nodeName);
  }

  // Calculate total size of the shape
  const originalElementCount = getElementCount(shape);

  // Adjust the total size for a texture to account for 4 elements per position (RGBA)
  if (originalElementCount % 4 !== 0) {
    throw new Error("convertShapeToTexture2DShape - originalElementCount must be divisible by 4. Node name: " + nodeName);
  }

  // Adjust the total size for a texture to account for 4 elements per position (RGBA)
  const totalTexturePositions = originalElementCount / (4 * divideBy);

  // Initialize height
  let height = Math.ceil(Math.sqrt(totalTexturePositions));

  // Calculate the width based on the height
  let width = Math.floor(totalTexturePositions / height);

  // Check if the calculated area is enough, if not, adjust the width
  if (width * height < totalTexturePositions) {
    width++;
  }

  // Check if the resulting dimensions are excessively large
  if (width >= maxTextureDim || height >= maxTextureDim) {
    throw new Error("Shape size is too large. Width: " + width + ", Height: " + height + ". Node name: " + nodeName);
  }

  if (width * height < totalTexturePositions) {
    throw new Error("Error: width * height < totalTexturePositions. width: " + width + ", height: " + height + ", totalTexturePositions: " + totalTexturePositions + ". Node name: " + nodeName);
  }

  // Calculate the textureElementCount
  const textureElementCount = width * height * divideBy * 4; // Now considering the 4 elements per position

  return [width, height, originalElementCount, textureElementCount];
}

export function createTexture(gl: WebGL2RenderingContext, width: number, height: number, nodeName: string, srcData: Float32Array | null): WebGLTexture {
  let texture = gl.createTexture();
  let paddedData = new Float32Array(width * height * 4);

  if (width > maxTextureDim || height > maxTextureDim) {
    throw new Error("Texture size is too large. Width: " + width + ", Height: " + height + ". Node name: " + nodeName);
  }

  if (srcData) {
    paddedData.set(srcData, 0);
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, paddedData);

  if (!texture) {
    throw new Error("Error creating WebGL texture");
  }

  return texture;
}

export function createTextureArray(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  depth: number,
  nodeName: string,
  srcData: Float32Array | null // Single Float32Array containing RGBA data for all layers
): WebGLTexture {
  let texture = gl.createTexture();
  let paddedData = new Float32Array(width * height * depth * 4);

  if (width > maxTextureDim || height > maxTextureDim) {
    throw new Error("Texture size is too large. Width: " + width + ", Height: " + height + ". Node name: " + nodeName);
  }

  try {
    if (srcData) {
      paddedData.set(srcData, 0);
    }
  }
  catch (e) {
    // debugger;
  }

  gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);

  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  // Initialize texture with data for all layers
  gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA32F, width, height, depth, 0, gl.RGBA, gl.FLOAT, paddedData);

  if (!texture) {
    throw new Error("Error creating WebGL texture array");
  }

  return texture;
}


export function getFromWeightMap(
  weightMap: NamedTensorsMap,
  nodeName: string): TensorType {
  const weightMapValue = weightMap[nodeName];

  if (!weightMapValue) {
    throw new Error("Weight map value not found for node: " + nodeName);
  }

  if (weightMapValue.length !== 1) {
    throw new Error("Expected weight map value to be a single tensor. Node name: " + nodeName);
  }

  let weightMapTensor: TensorType = weightMapValue[0];

  return weightMapTensor;
}

export function createWeightDataTextureArray(
  gl: WebGL2RenderingContext,
  weightMap: NamedTensorsMap,
  node: Node,
  uniformName: string,
  convertToTextureShape: boolean = false,
  dataFormat: DataFormat,
  transform?: (weights: TensorType, node: Node) => TensorType): WebGLDataTextureArray {
  if (node.op !== "Const") {
    throw new Error("createWeightDataTextureArray - node is not a Const node. Node name: " + node.name);
  }

  let weightMapTensor = getFromWeightMap(weightMap, node.name);
  let originalShape = weightMapTensor.shape;
  let shape = originalShape;

  if (transform) {
    weightMapTensor = transform(weightMapTensor, node);
    shape = weightMapTensor.shape;
    // console.log("weightMapTensor:")
    // print(weightMapTensor);
  }

  const data = weightMapTensor.dataSync() as Float32Array;

  let depth: number | undefined;
  let width: number | undefined;
  let height: number | undefined;
  let originalElementCount: number | undefined;

  if (dataFormat === "NHWC") {
    depth = shape[3];
    width = shape[2];
    height = shape[1];
  } else if (dataFormat === "HWC") {
    depth = shape[2];
    width = shape[1];
    height = shape[0];
  } else if (dataFormat === "VEC") {
    depth = 1;
    width = shape[0];
    height = 1;
  }

  if (!depth || !width || !height) {
    throw new Error("Error: all of depth, width, or height should be != 0 and != undefined. Node name: " + node.name);
  }

  if (convertToTextureShape) {
    [width, height, originalElementCount] = convertShapeToTexture2DShape(shape, node.name);
  }

  const RGBADepth = Math.ceil(depth / 4);
  const weightTextureArray = createTextureArray(gl, width, height, RGBADepth, node.name, data);

  const webGLDataTextureArray: WebGLDataTextureArray = {
    webGLType: "sampler2DArray",
    nodeName: node.name,
    texture: weightTextureArray,
    RGBATextureShape: [width, height, RGBADepth, 4],
    RGBATextureElementCount: width * height * RGBADepth * 4,
    uniformName,
    originalShape,
    originalElementCount: getElementCount(originalShape),
  };

  return webGLDataTextureArray;
}


export function removeBatchDimension(shape: number[], nodeName: string): number[] {
  if (shape.length === 0) {
    throw new Error("Shape must have at least one dimension");
  }

  if (shape.length === 1) {
    return shape;
  }

  // Remove the first dimension.
  const removedDimension = shape.shift();

  if (removedDimension !== 1) {
    throw new Error("Expected batch dimension to be 1. Node name: " + nodeName);
  }

  return shape;
}

export function createAndBindFramebuffer(gl: WebGL2RenderingContext): WebGLFramebuffer {
  const frameBuffer = gl.createFramebuffer();

  if (!frameBuffer) {
    throw new Error("Error creating WebGL framebuffer");
  }

  // I think we only need to bind the framebuffer once? I could be mistaken though.
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);

  return frameBuffer;
}


// Function to handle texture binding
export function handleTextureUniforms(
  gl: WebGL2RenderingContext,
  webGLDataTexture: WebGLDataTexture | WebGLDataTextureArray,
  textureUnitIndex: number,
  programInfo: ProgramInfo,
  dataToUpdate: ArrayBufferView | null) {

  // console.log("handleTextureUniforms:\
  //   webGLDataTexture node name: " + webGLDataTexture.nodeName + "\
  //   textureUnitIndex: " + textureUnitIndex + "\
  //   uniformName: " + webGLDataTexture.uniformName + "\
  //   isWebGLDataTextureArray: " + isWebGLDataTextureArray(webGLDataTexture));

  // 1st step - handle texture uniforms.
  const location = gl.getUniformLocation(programInfo.program, webGLDataTexture.uniformName);
  gl.activeTexture(gl.TEXTURE0 + textureUnitIndex);

  // Determine if we are handling a texture array.
  if (isWebGLDataTextureArray(webGLDataTexture)) {
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, webGLDataTexture.texture);
    if (dataToUpdate) {
      gl.texSubImage3D(
        gl.TEXTURE_2D_ARRAY,
        0, // mip level
        0, // x offset
        0, // y offset
        0, // z offset
        webGLDataTexture.RGBATextureShape[0], // width
        webGLDataTexture.RGBATextureShape[1], // height
        webGLDataTexture.RGBATextureShape[2], // depth aka number of textures
        gl.RGBA, // RGBA, not RGBA32F. RGBA32F is the internal format.
        gl.FLOAT,
        dataToUpdate // dataToUpdate should contain data for all layers
      );
    }
  } else {
    // Handle regular 2D texture - e.g. certain weights.
    const texture2D = webGLDataTexture as WebGLDataTexture;
    gl.bindTexture(gl.TEXTURE_2D, texture2D.texture);
    if (dataToUpdate) {
      gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        0,
        0,
        texture2D.RGBATextureShape[0], // width
        texture2D.RGBATextureShape[1], // height
        gl.RGBA,
        gl.FLOAT,
        dataToUpdate);
    }
  }

  gl.uniform1i(location, textureUnitIndex);

  // 2nd step - handle shape uniforms.
  /*
  const shapeUniformName = getShapeUniformName(uniformName);
  const squareShapeUniformName = getSquareShapeUniformName(uniformName);
 
  const shapeLocation = gl.getUniformLocation(programInfo.program, shapeUniformName);
  const squareShapeLocation = gl.getUniformLocation(programInfo.program, squareShapeUniformName);
 
  if (!shapeLocation) {
    throw new Error("Error getting shapeLocation for uniform: " + shapeUniformName);
  }
 
  if (!squareShapeLocation) {
    throw new Error("Error getting squareShapeLocation for uniform: " + squareShapeUniformName);
  }
 
  // Pad the shape with 0s if it's less than 4 dimensions.
  const paddedShape = webGLData.shape.concat(Array(4 - webGLData.shape.length).fill(0));
 
  gl.uniform4iv(shapeLocation, paddedShape);
  gl.uniform1i(squareShapeLocation, webGLData.squareSize);
  */
};

export function createOutputTextureArray(
  gl: WebGL2RenderingContext,
  originalShape: number[] | null | undefined,
  options: WedgeOptions,
  nodeName: string,
  customShapeUpdate?: CustomShapeUpdate): WebGLDataTextureArray | null {
  if (!originalShape || !originalShape.length) {
    console.error("createOutputTextureArray - originalShape is null or empty. Node name: " + nodeName);
    return null;
  }

  const hasCorrectDims = originalShape.length === 3 || (options.hasBatchDimension && originalShape.length === 4);
  if (!hasCorrectDims) {
    throw new Error("createOutputTextureArray - output shape must have 3 dimensions (4 if batch dim exists). Node name: " + nodeName);
  }

  const outputTextureOriginalElementCount = getElementCount(originalShape);
  const numRenderTargets = findRecommendedRenderTargets(outputTextureOriginalElementCount, options.renderTargetBreakpoints);

  let shapeWithPaddedChannels = originalShape.slice();

  if (options.transformations.padChannels) {
    shapeWithPaddedChannels = getChannelPaddedShape(originalShape);
  }

  // Get the final texture array dimensions.

  let [width, height, _, textureElementCount] = convertShapeToTexture2DShape(shapeWithPaddedChannels, nodeName, numRenderTargets);

  if (customShapeUpdate) {
    [width, height, _, textureElementCount] = customShapeUpdate(shapeWithPaddedChannels, width, height, numRenderTargets, nodeName);
  }

  // Create the texture array.
  const textureArray = createTextureArray(gl, width, height, numRenderTargets, nodeName, null);
  const splitNodeName = nodeName.split("/");
  const lastTwoNodeNameParts = splitNodeName.slice(-2);

  return {
    webGLType: "sampler2DArray",
    uniformName: "uOutputFrom_" + lastTwoNodeNameParts.join("_"),
    RGBATextureShape: [width, height, numRenderTargets, 4],
    nodeName,
    texture: textureArray,
    RGBATextureElementCount: width * height * numRenderTargets * 4,
    originalShape,
    originalElementCount: outputTextureOriginalElementCount,
  };
}
