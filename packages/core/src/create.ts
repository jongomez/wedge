import * as tf from '@tensorflow/tfjs';
import { createAndBindFramebuffer } from './backends/webgl/buffersAndTextures';
import { defaultOptions } from './constants';
import { WedgeOptions, WebGLDataTextureArray } from './backends/webgl/types';
import { initWebGL } from './backends/webgl/setupShadersAndWebGL';
import { initWebGLData, updateUniformsForProgram } from './backends/webgl/webGLData';
import { updateFramebufferTextureLayer } from './backends/webgl/buffersAndTextures';
import { createOpNodeMapPrograms, createWeightName, mapLayerClassesToOpName } from './modelHelpers';
import { Node } from '@tensorflow/tfjs-converter/dist/operations/types';
import { NamedTensorsMap, NamedTensorMap } from '@tensorflow/tfjs-converter/dist/data/types';
import { removePadChannels } from './transforms';

export interface WedgeInstance {
  predict(inputs: ArrayBufferView[]): Float32Array;
  finalOutputData: WebGLDataTextureArray | null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getLayersModelWeightNodes(weightTensors: any[], layerName: string): Node[] {
  return weightTensors.map((tensor, index) => ({
    name: createWeightName(layerName, index),
    op: 'Const',
    inputNames: [],
    inputs: [],
    attrParams: {},
    category: 'custom' as const,
    inputParams: {},
    children: []
  }));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getLayersModelInputNodes(layer: any, orderedNodes: Node[]): Node[] {
  const inputNodeNames = layer.inboundNodes.map((node: any) =>
    node.inboundLayers.map((layer: any) => layer.name)
  ).flat();
  const inputNodes = orderedNodes.filter(node => inputNodeNames.includes(node.name));
  const layerClassName = layer.getClassName();

  if (!inputNodes.length && layerClassName !== 'InputLayer') {
    throw new Error("Error: no input nodes found for layer: " + layer.name);
  }

  return inputNodes;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getAttrParams(layer: any): Record<string, unknown> {
  const config = layer.getConfig();
  const attrParams: Record<string, unknown> = {};

  if ('strides' in config) {
    attrParams['strides'] = { value: config.strides };
  }
  if ('padding' in config) {
    // Use 'pad' key to match what getConv2DParams expects
    attrParams['pad'] = { value: config.padding };
  }
  if ('dilationRate' in config) {
    attrParams['dilations'] = { value: config.dilationRate };
  }

  return attrParams;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function processLayersModel(
  layersModel: any,
  inputs: NamedTensorMap,
  hasBatchDimension: boolean
): { orderedNodes: Node[]; layersModelWeightMap: NamedTensorsMap } {
  const layersModelWeightMap: NamedTensorsMap = {};
  const orderedNodes: Node[] = [];

  for (let i = 0; i < layersModel.layers.length; i++) {
    const layer = layersModel.layers[i];
    const layerName = layer.name;
    const weightTensors = layer.getWeights();

    if (weightTensors.length > 2) {
      throw new Error("Error: only 2 weights max are supported. Layer " + layerName + " has " + weightTensors.length + " weights.");
    }

    const inputNodes = getLayersModelInputNodes(layer, orderedNodes);
    const weightNodes = getLayersModelWeightNodes(weightTensors, layerName);
    const opName = mapLayerClassesToOpName(layer.getClassName());
    const attrParams = getAttrParams(layer);

    const finalInputNodes: Node[] = inputNodes.concat(weightNodes);

    if (layer.getClassName() === 'InputLayer') {
      const inputLayerNode = {
        attrParams,
        name: layerName,
        op: opName,
        inputNames: [] as string[],
        inputs: [] as Node[],
        category: 'custom' as const,
        inputParams: {},
        children: [] as Node[]
      } as Node;

      Object.keys(inputs).forEach(name => {
        if (name !== layerName) return;

        let inputTensor = inputs[name];
        if (!hasBatchDimension) {
          // User's input won't have batch dimension, so squeeze the dummy tensor to match
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          inputTensor = (tf as any).squeeze(inputTensor, [0]);
        }
        layersModelWeightMap[layerName] = [inputTensor];
      });

      orderedNodes.unshift(inputLayerNode);
    } else {
      const opNode = {
        attrParams,
        name: layerName,
        op: opName,
        inputNames: finalInputNodes.map(node => node.name),
        inputs: finalInputNodes,
        category: 'custom' as const,
        inputParams: {},
        children: [] as Node[]
      } as Node;

      orderedNodes.push(opNode);
    }

    for (let j = 0; j < weightTensors.length; j++) {
      layersModelWeightMap[createWeightName(layerName, j)] = [weightTensors[j]];
    }
  }

  return { orderedNodes, layersModelWeightMap };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function createDummyInputsNamedTensorMap(model: any): NamedTensorMap {
  const inputs: NamedTensorMap = {};

  for (const input of model.inputs) {
    const shape = input.shape.map((dim: number | null) => dim === null ? 1 : dim);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    inputs[input.name.split('/')[0]] = (tf as any).zeros(shape);
  }

  return inputs;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function createWedge(
  model: any,
  options: WedgeOptions = defaultOptions
): Promise<WedgeInstance> {
  // Initialize WebGL context
  const { canvas, gl, maxColorAttachments } = initWebGL(
    options.canvasWidth,
    options.canvasHeight,
    options.viewportMaxSize
  );

  // Validate render target breakpoints
  for (const breakpoint of options.renderTargetBreakpoints) {
    if (breakpoint.numberOfRenderTargets > maxColorAttachments) {
      throw new Error("numberOfRenderTargets exceeds maxColorAttachments");
    }
  }

  // Create input tensor map
  const inputsNamedTensorMap = createDummyInputsNamedTensorMap(model);
  const inputTensorNames = new Set(Object.keys(inputsNamedTensorMap));

  // Process the layers model to get ordered nodes and weight map
  const { orderedNodes, layersModelWeightMap } = processLayersModel(
    model,
    inputsNamedTensorMap,
    options.hasBatchDimension
  );

  // Initialize WebGL data for all nodes
  const { opNodeMap, nodeWebGLDataMap } = initWebGLData(
    gl,
    layersModelWeightMap,
    orderedNodes,
    "LayersModel",
    options
  );

  // Create WebGL programs
  const opNodeWithProgramMap = createOpNodeMapPrograms(opNodeMap, gl, layersModelWeightMap);

  // Create and bind framebuffer
  const frameBuffer = createAndBindFramebuffer(gl);

  let finalOutputData: WebGLDataTextureArray | null = null;

  // Run function
  function run(inputRawData: ArrayBufferView[]): Float32Array {
    let opNodesRan = 0;

    opNodeWithProgramMap.forEach((opNodeWithProgram, opNodeName) => {
      gl.useProgram(opNodeWithProgram.programInfo.program);
      updateUniformsForProgram(gl, opNodeWithProgram, inputTensorNames, inputRawData, options);
      updateFramebufferTextureLayer(gl, opNodeWithProgram.opNode.output!);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      finalOutputData = opNodeWithProgram.opNode.output;
      opNodesRan++;
    });

    return readOutput();
  }

  // Read output function
  function readOutput(): Float32Array {
    if (!finalOutputData) {
      throw new Error("finalOutputData is not defined");
    }

    const [outputTexturesWidth, outputTexturesHeight, numberOfTextures] = finalOutputData.RGBATextureShape;

    if (!outputTexturesWidth || !outputTexturesHeight || !numberOfTextures) {
      throw new Error("Output texture dimensions not defined");
    }

    const sizePerLayer = outputTexturesWidth * outputTexturesHeight * 4;
    const output = new Float32Array(sizePerLayer * numberOfTextures);

    for (let layer = 0; layer < numberOfTextures; layer++) {
      gl.framebufferTextureLayer(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0 + layer,
        finalOutputData.texture,
        0,
        layer
      );
      gl.readBuffer(gl.COLOR_ATTACHMENT0 + layer);

      const layerData = new Float32Array(sizePerLayer);
      gl.readPixels(0, 0, outputTexturesWidth, outputTexturesHeight, gl.RGBA, gl.FLOAT, layerData);
      output.set(layerData, layer * sizePerLayer);
    }

    // Remove channel padding and return only original elements
    return removePadChannels(output, finalOutputData.originalElementCount, finalOutputData.originalShape);
  }

  return {
    predict: run,
    get finalOutputData() {
      return finalOutputData;
    }
  };
}
