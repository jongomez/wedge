import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { padChannels } from "../../transforms";
import { ModelType, NodeWebGLDataMap, OpName, WebGLData, WebGLDataNonTexture, WebGLOpNode, WebGLOpNodeMap, WebGLOpNodeWithProgram, WedgeOptions } from "./types";
import { convertShapeToTexture2DShape, createTextureArray, getElementCount, getFromWeightMap, handleTextureUniforms } from "./buffersAndTextures";
import { isWebGLDataNonTexture, isWebGLDataTexture, isWebGLDataTextureArray } from "./helpers";
import { initArithmeticWebGLData } from "./ops/arithmetic/init";
import { initConv2DWebGLData } from "./ops/conv2D/init";
import { initDepthwiseConv2DWebGLData } from "./ops/depthwiseConv2D/init";
import { initNotSupportedOpWebGLData } from "./ops/nonSupported/init";
import { initPadWebGLData } from "./ops/pad/init";
import { initReluWebGLData } from "./ops/relu/init";
import { initReshapeWebGLData } from "./ops/reshape/init";
import { initResizeBilinearWebGLData } from "./ops/ResizeBilinear/init";
import { initMaxPoolWebGLData } from "./ops/maxPool/init";
/*

AddV2 case:
- tensor input 1
- optional scalar input
- optional tensor input 2

Final result:
myMap["AddV2"] = input and output data (no weights necessary here bruh)

*/


let inputCount = 0;

// Non operation input nodes are, for example: constants, placeholders, and sometimes weights.
export function getNonOpWebGLData(
  gl: WebGL2RenderingContext,
  weightMap: NamedTensorsMap,
  node: Node,
  options: WedgeOptions,
): WebGLData {
  let weightMapTensor = getFromWeightMap(weightMap, node.name);
  let originalShape = weightMapTensor.shape;
  let data = weightMapTensor.dataSync();

  if (weightMapTensor.shape.length === 0) {
    // Scalar case. For scalars, the rank should be 0. If not, throw an error.
    if (weightMapTensor.rank !== 0) {
      throw new Error("Expected rank to be 0 when shape is empty. Node name: " + node.name);
    }

    return {
      webGLType: "float",
      nodeName: node.name,
      data: [data[0]],
      uniformName: "uInput_" + (++inputCount),
    } as WebGLDataNonTexture
  } else {
    if (node.name === "sklçafaçsd") {
      debugger;
    }

    // Texture case.
    if (options.transformations.padChannels) {
      // There should be no weights here. Only input placeholders and constants.
      // We use the weightMap here because it not only has weights, but placeholders and constants as well.
      weightMapTensor = padChannels(weightMapTensor, node.name);
      data = weightMapTensor.dataSync();
    }

    const [width, height, _] = convertShapeToTexture2DShape(weightMapTensor.shape, node.name);
    const numberOfTextures = 1;

    const textureArray = createTextureArray(gl, width, height, numberOfTextures, node.name, data as Float32Array);

    return {
      webGLType: "sampler2DArray",
      texture: textureArray,
      nodeName: node.name,
      RGBATextureShape: [width, height, numberOfTextures, 4],
      uniformName: "uInput_" + (++inputCount),
      RGBATextureElementCount: width * height * numberOfTextures * 4,
      originalElementCount: getElementCount(originalShape),
      originalShape,
    }
  }
}

export function getWebGLDataElseNull(node: Node, nodeWebGLDataMap: NodeWebGLDataMap, opNodeMap: WebGLOpNodeMap): WebGLData | null {
  // 1st try - get from nodeWebGLDataMap. This map has constants and placeholders.
  let webGLData = nodeWebGLDataMap.get(node.name) as WebGLData | null | undefined;

  if (!webGLData) {
    // 2nd try - get from opNodeMap. This map has operation nodes, so output data should be available here.
    webGLData = opNodeMap.get(node.name)?.output;

    if (!webGLData) {
      return null;
    }
  }

  return webGLData;
}

type InitWebGLDataReturn = {
  opNodeMap: WebGLOpNodeMap;
  nodeWebGLDataMap: NodeWebGLDataMap;
}

export function initWebGLData(
  gl: WebGL2RenderingContext,
  weightMap: NamedTensorsMap,
  orderedNodes: Node[],
  modelType: ModelType,
  options: WedgeOptions): InitWebGLDataReturn {
  const opNodeMap: WebGLOpNodeMap = new Map<string, WebGLOpNode>();
  const nodeWebGLDataMap: NodeWebGLDataMap = new Map<string, WebGLData>();

  for (const node of orderedNodes) {
    const isOperationNode = node.op !== "Placeholder" && node.op !== "Const";

    // Step 1 - get non operation input shapes.
    // Non operation input nodes are, for example: constants, placeholders, and sometimes weights.
    if (!isOperationNode) {
      const webGLData = getNonOpWebGLData(gl, weightMap, node, options);
      nodeWebGLDataMap.set(node.name, webGLData);
      continue;
    }

    // Step 2 - handle operation nodes. For example: adds, convolutions, etc.
    let opNode: WebGLOpNode | null = null;
    let opName: OpName = node.op as OpName;

    switch (opName) {
      case "Mul":
      case "AddV2":
        opNode = initArithmeticWebGLData(gl, node, nodeWebGLDataMap, opNodeMap, opName, options);
        break;
      case "Conv2D":
      case "_FusedConv2D":
        opNode = initConv2DWebGLData(
          gl,
          node,
          nodeWebGLDataMap,
          opNodeMap,
          weightMap,
          modelType,
          options);
        break;
      case "Relu":
        opNode = initReluWebGLData(gl, node, nodeWebGLDataMap, opNodeMap, options);
        break;
      case "Relu6":
        opNode = initReluWebGLData(gl, node, nodeWebGLDataMap, opNodeMap, options, "Relu6");
        break;
      case "Sigmoid":
        opNode = initReluWebGLData(gl, node, nodeWebGLDataMap, opNodeMap, options, "Sigmoid");
        break;
      case "DepthwiseConv2dNative":
      case "DepthwiseConv2D":
      case "FusedDepthwiseConv2dNative":
        opNode = initDepthwiseConv2DWebGLData(gl,
          node,
          nodeWebGLDataMap,
          opNodeMap,
          weightMap,
          modelType,
          options);
        break;
      case "ResizeBilinear":
        opNode = initResizeBilinearWebGLData(gl,
          node,
          nodeWebGLDataMap,
          opNodeMap,
          weightMap,
          modelType,
          options);
        break;
      case "Pad":
      case "PadV2":
      case "MirrorPad":
        opNode = initPadWebGLData(gl,
          node,
          nodeWebGLDataMap,
          opNodeMap,
          weightMap,
          modelType,
          options);
        break;
      case "Reshape":
        opNode = initReshapeWebGLData(gl,
          node,
          nodeWebGLDataMap,
          opNodeMap,
          weightMap,
          options);
        break;
      case "MaxPool":
        opNode = initMaxPoolWebGLData(gl,
          node,
          nodeWebGLDataMap,
          opNodeMap,
          modelType,
          options);
        break;
      default:
        opNode = initNotSupportedOpWebGLData(gl, node, nodeWebGLDataMap, opNodeMap, options);
        console.error("initWebGLData - operation not supported: " + node.op);
        break;
    }

    if (!opNode) {
      throw new Error("initWebGLData - opNode is null.");
    }

    // Check if the opNodeMap already has this node. If so, throw an error.
    if (opNodeMap.has(node.name)) {
      throw new Error("initWebGLData - opNodeMap already has node: " + node.name);
    }

    opNodeMap.set(node.name, opNode);
  }

  return { opNodeMap, nodeWebGLDataMap };
}


export function updateUniformsForProgram(
  gl: WebGL2RenderingContext,
  opNodeWithProgram: WebGLOpNodeWithProgram,
  inputTensorNames: Set<string>,
  inputRawData: ArrayBufferView[],
  options: WedgeOptions) {
  const programInfo = opNodeWithProgram.programInfo;
  let textureUnitIndex = 0; // Start from texture unit 0

  // Bind input textures.
  // console.log("opNode.node.name", opNode.node.name);

  let inputIndex = 0;
  opNodeWithProgram.opNode.inputs.forEach(opInput => {
    if (isWebGLDataTextureArray(opInput)) {
      // Handle initial input texture differently - as this texture's data may change every time we run the program.
      if (inputTensorNames.has(opInput.nodeName)) {
        handleTextureUniforms(gl, opInput, textureUnitIndex, programInfo, inputRawData[inputIndex]);
        inputIndex++;
      } else {
        // Handle other types of inputs, e.g. outputs of other operation nodes. If not, you may get:
        // GL_INVALID_OPERATION: Two textures of different types use the same sampler location.
        // NOTE: even when there is no data change, we still need to call bindTexture() and uniform1i().
        handleTextureUniforms(gl, opInput, textureUnitIndex, programInfo, null);
      }

      textureUnitIndex++;
    }
  });

  // Bind weight texture if it exists.
  opNodeWithProgram.opNode.weights.forEach(weights => {
    handleTextureUniforms(gl, weights, textureUnitIndex, programInfo, null);
    textureUnitIndex++;
  });

  // Now handle non-texture types if any.
  opNodeWithProgram.opNode.inputs.forEach(input => {
    const location = gl.getUniformLocation(programInfo.program, input!.uniformName);
    if (isWebGLDataNonTexture(input)) {
      switch (input.webGLType) {
        case 'float':
          gl.uniform1f(location, input.data[0]);
          break;
        case 'vec2':
          gl.uniform2fv(location, input.data);
          break;
        case 'vec3':
          gl.uniform3fv(location, input.data);
          break;
        case 'vec4':
          gl.uniform4fv(location, input.data);
          break;
      }
    }
  });
}

// For ops with multiple inputs, where the output original shape is equal to the largest input (for broadcasting).
export function getWebGLOpOutputOriginalShape(
  node: Node,
  nodeWebGLDataMap: NodeWebGLDataMap,
  opNodeMap: WebGLOpNodeMap,
): number[] {
  let originalShape: number[] = [];
  let maxElementCount = 0;

  for (const input of node.inputs) {
    const webGLData = getWebGLDataElseNull(input, nodeWebGLDataMap, opNodeMap);

    if (isWebGLDataTexture(webGLData) || isWebGLDataTextureArray(webGLData)) {
      const elementCount = getElementCount(webGLData.originalShape);
      // For broadcasting, use the shape with the most elements
      if (elementCount > maxElementCount) {
        maxElementCount = elementCount;
        originalShape = webGLData.originalShape;
      }
    }
  }

  return originalShape;
}

