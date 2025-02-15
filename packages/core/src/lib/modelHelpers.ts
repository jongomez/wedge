import { GraphModel, LayersModel, layers } from "@tensorflow/tfjs";
import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";
import { NamedTensorMap, Tensor, squeeze, tensor } from "@tensorflow/tfjs-core";
import { getElementCount } from "./buffersAndTextures";
import { getValidShape } from "./helpers";
import { parseNodeName } from "./model_analysis";
import { createWebGLProgram } from "./setupShadersAndWebGL";
import { defaultVsSource } from "./shaderHelpers";
import { GraphModelOpNames, LayersModelLayerClass, WebGLOpNode, WebGLOpNodeWithProgram, WebGLOpNodeWithProgramMap } from "./types";

export function graphModelExecuteSetup(
  executor: any,
  hasBatchDimension: boolean,
  inputs: NamedTensorMap,
  outputs?: string[],
): Node[] {
  if (!outputs) {
    throw new Error("graphModelExecuteSetup - error: outputs is not defined.");
  }

  let orderedNodes: Node[] = [];

  inputs = executor.mapInputs(inputs);
  executor.checkInputs(inputs);
  executor.checkInputShapeAndType(inputs);
  outputs = executor.mapOutputs(outputs);
  executor.checkOutputs(outputs);

  const outputNodeNames = outputs!.map(name => parseNodeName(name)[0]);
  // const outputNodeNameSet = new Set(outputNodeNames);
  let outputNodes: Node[] = outputNodeNames.map(name => executor.graph.nodes[name]);

  // If no outputs are specified, then use the default outputs of the model.
  if (outputNodes.length === 0) {
    throw new Error("Error: outputNodes.length is 0.");
  }

  // Add inputs to the weightMap. This way, weightMaps will have all the weights + input tensors.
  Object.keys(inputs).forEach(name => {
    let inputTensor = inputs[name];

    // Remove batch dimension - do this instead of calling removeBatchDimension
    if (hasBatchDimension) {
      inputTensor = squeeze(inputTensor, [0]);
    }

    const [nodeName, index] = parseNodeName(name);
    executor.weightMap[nodeName] = [inputTensor];
  });

  // Original compile function:
  // https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/src/executor/graph_executor.ts#L167
  const compilation = executor.compile(inputs, outputNodes);
  orderedNodes = compilation.orderedNodes;

  // FIXME: TODO: Removing identity nodes is not ideal - if there's an identity node somewhere
  // in the middle of the graph, it will be skipped. Assuming for now they're always unnecessary.
  orderedNodes = orderedNodes.filter(node => node.op !== "Identity");

  console.log("executor.weightMap", executor.weightMap)

  return orderedNodes;
}

export function createOpNodeMapPrograms(opNodeMap: Map<string, WebGLOpNode>, gl: WebGL2RenderingContext, weightMap: NamedTensorsMap): WebGLOpNodeWithProgramMap {
  const opNodeWithProgramMap: WebGLOpNodeWithProgramMap = new Map<string, WebGLOpNodeWithProgram>();

  opNodeMap.forEach((opNode: WebGLOpNode) => {
    if (!opNode.fsSource) {
      // console.error("createOpNodeMapPrograms - opNode.fsSource is not defined. OpNode: " + opNode.node.name);
      return;
    }

    const programInfo = createWebGLProgram(gl, defaultVsSource, opNode.fsSource);
    opNodeWithProgramMap.set(opNode.node.name, {
      opNode,
      programInfo
    });
  });

  return opNodeWithProgramMap;
}

export function mapLayerClassesToOpName(layerClassName: string): GraphModelOpNames {
  switch (layerClassName as LayersModelLayerClass) {
    case "InputLayer":
      return "Const";
    case "Conv2D":
      return "Conv2D";
    // case "Add":
    //   return "AddV2";
    case "DepthwiseConv2D":
      return "DepthwiseConv2D";
    case "ReLU":
      return "Relu";
    case "Add":
      return "AddV2";
    default:
      throw new Error("mapLayerClassesToOpName - layerClassName not supported: " + layerClassName);
  }
}

export function createWeightName(layerName: string, index: number): string {
  return layerName + '_weight_' + index;
}

// Helper function to create a weight node
export function getLayersModelWeightNodes(weightTensors: Tensor[], layerName: string): Node[] {
  return weightTensors.map((tensor, index) => ({
    name: createWeightName(layerName, index),
    op: 'Const',
    inputNames: [],
    inputs: [],
    attrParams: {},
    category: 'custom', // not sure what's the correct category for weights. But, it shouldn't matter much.
    inputParams: {},
    children: []
  }));
}

export function getLayersModelInputNodes(layer: layers.Layer, orderedNodes: Node[]) {
  // Assuming each layer takes input from the immediately preceding layer in a linear topology
  const inputNodeNames = layer.inboundNodes.map(node => node.inboundLayers.map(layer => layer.name)).flat();
  const inputNodes = orderedNodes.filter(node => inputNodeNames.includes(node.name));
  const layerClassName = layer.constructor.name;

  if (!inputNodes.length && layerClassName !== 'InputLayer') {
    // If no input nodes are found, and if the layer class name is not InputLayer, then throw an error.
    throw new Error("Error: no input nodes found for layer: " + layer.name);
  }

  return inputNodes;
}

export function getAttrParams(layer: any): Node['attrParams'] {

  switch (layer.constructor.name) {
    case 'Dense':
      return {
        units: { value: layer.units, type: 'number' },
        activation: { value: layer.activation, type: 'string' }
      };
    case 'Conv2D':
      return {
        filters: { value: layer.filters, type: 'number' },
        strides: { value: layer.strides, type: 'number[]' },
        pad: { value: layer.padding, type: 'string' },
        activation: { value: layer.activation.getClassName(), type: 'string' }
      };
    case 'DepthwiseConv2D':
      return {
        kernelSize: { value: layer.kernelSize, type: 'number[]' },
        strides: { value: layer.strides, type: 'number[]' },
        pad: { value: layer.padding, type: 'string' },
        activation: { value: layer.activation.getClassName(), type: 'string' }
      };
    // Add more cases as needed for other layer types
    default:
      return {}; // No attrParams for unknown layer types
  }
}

type ProcessLayersModelReturn = {
  orderedNodes: Node[],
  layersModelWeightMap: NamedTensorsMap
}


export function processLayersModel(
  layersModel: LayersModel,
  inputs: NamedTensorMap,
  hasBatchDimension: boolean): ProcessLayersModelReturn {
  const layersModelWeightMap: NamedTensorsMap = {};
  const orderedNodes: Node[] = [];

  for (let i = 0; i < layersModel.layers.length; i++) {
    const layer = layersModel.layers[i];
    const layerName = `${layer.name}`;
    const weightTensors = layer.getWeights(); // Get the weight tensors of the layer

    if (weightTensors.length > 2) {
      // Currently only support max 2 weights - one for the kernel and one for the bias. 
      throw new Error("Error: only 2 weights max. are supported right now. Layer " + layerName + " has " + weightTensors.length + " weights.");
    }

    const inputNodes = getLayersModelInputNodes(layer, orderedNodes);
    const weightNodes = getLayersModelWeightNodes(weightTensors, layerName);
    const opName = mapLayerClassesToOpName(layer.constructor.name);
    const attrParams = getAttrParams(layer);

    const finalInputNodes: Node[] = inputNodes.concat(weightNodes);

    if (layer.constructor.name === 'InputLayer') {
      // Handle InputLayer specially
      const inputLayerNode: Node = {
        attrParams,
        name: layerName,
        op: opName,
        inputNames: [], // InputLayer does not have incoming nodes
        inputs: [],
        category: 'custom',
        inputParams: {},
        children: []
      };

      // Add the input tensors to the weight map. This is also done for GraphModels.
      Object.keys(inputs).forEach(name => {
        // If the input tensor name does not match the layer name, then skip it.
        if (name !== layerName) {
          return;
        }

        let inputTensor = inputs[name];
        // Remove batch dimension - do this instead of calling removeBatchDimension
        if (hasBatchDimension) {
          inputTensor = squeeze(inputTensor, [0]);
        }

        layersModelWeightMap[layerName] = [inputTensor];
      });

      // Add the input layer node to the start of the ordered nodes list.
      orderedNodes.unshift(inputLayerNode);
    } else {
      const opNode: Node = {
        attrParams,
        name: layerName,
        op: opName,
        inputNames: finalInputNodes.map(node => node.name),
        inputs: finalInputNodes,
        category: 'custom',
        inputParams: {},
        children: []
      };

      orderedNodes.push(opNode)
    }

    if (weightTensors.length !== weightNodes.length) {
      throw new Error("Error: weightTensors and weightNodes length mismatch for layer: " + layerName);
    }

    // Regardless of layer type, process weights
    for (let i = 0; i < weightTensors.length; i++) {
      // XXX: I don't think we need the weights in the orderedNodes list?
      // orderedNodes.push(weightNodes[i]);
      layersModelWeightMap[createWeightName(layerName, i)] = [weightTensors[i]];
    }
  }

  return { orderedNodes, layersModelWeightMap };
}

// The dummy input tensors are create at the start of the model execution.
// They are used when the real input tensors are not available.
export function createDummyInputsNamedTensorMap(model: LayersModel | GraphModel): NamedTensorMap {
  let dummyInputsNamedTensorMap: NamedTensorMap = {};

  if (model instanceof GraphModel) {
    model.inputs.forEach(input => {
      const inputShape = getValidShape(input.shape as number[]);
      const elementCount = getElementCount(inputShape);
      let dummyInputTensor: Tensor;

      if (input.dtype === "int32") {
        dummyInputTensor = tensor(new Int32Array(elementCount), inputShape);
      } else if (input.dtype === "float32") {
        dummyInputTensor = tensor(new Float32Array(elementCount), inputShape);
      } else {
        throw new Error("createInputsNamedTensorMap - input.dtype not supported: " + input.dtype);
      }

      dummyInputsNamedTensorMap[input.name] = dummyInputTensor;
    });
  } else if (model instanceof LayersModel) {
    model.inputLayers.forEach(layer => {
      const inputTensorShape = layer.batchInputShape.slice(1);
      if (inputTensorShape.includes(null)) {
        throw new Error("createInputsNamedTensorMap - inputTensorShape has null values: " + inputTensorShape);
      }

      const elementCount = getElementCount(inputTensorShape as number[]);
      const dummyInputTensor = tensor(new Float32Array(elementCount), inputTensorShape as number[]);
      dummyInputsNamedTensorMap[layer.name] = dummyInputTensor;
    });
  } else {
    throw new Error("createInputsNamedTensorMap - modelType not supported.");
  }

  return dummyInputsNamedTensorMap;
}