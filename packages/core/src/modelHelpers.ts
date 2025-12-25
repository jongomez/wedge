import * as tf from '@tensorflow/tfjs';
import { NamedTensorsMap } from "@tensorflow/tfjs-converter/dist/data/types";
import { GraphModelOpNames, LayersModelLayerClass, WebGLOpNode, WebGLOpNodeWithProgram, WebGLOpNodeWithProgramMap } from "./backends/webgl/types";
import { createWebGLProgram } from "./backends/webgl/setupShadersAndWebGL";
import { defaultVsSource } from "./backends/webgl/shaderHelpers";


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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function mapLayerClassesToOpName(layerClassName: string, layerConfig?: any): GraphModelOpNames {
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
      // Check if maxValue is 6, which indicates Relu6
      if (layerConfig?.maxValue === 6) {
        return "Relu6";
      }
      return "Relu";
    case "Add":
      return "AddV2";
    case "ZeroPadding2D":
      return "Pad";
    case "MaxPooling2D":
      return "MaxPool";
    case "Reshape":
      return "Reshape";
    case "Activation":
      // Handle activation layers based on activation function
      if (layerConfig?.activation === 'sigmoid') {
        return "Sigmoid";
      } else if (layerConfig?.activation === 'relu') {
        return "Relu";
      } else if (layerConfig?.activation === 'relu6') {
        return "Relu6";
      }
      throw new Error("mapLayerClassesToOpName - activation not supported: " + layerConfig?.activation);
    default:
      throw new Error("mapLayerClassesToOpName - layerClassName not supported: " + layerClassName);
  }
}

export function createWeightName(layerName: string, index: number): string {
  return layerName + '_weight_' + index;
}



// Function to create the model
export function createAllConvModel(numConvLayers: number): tf.LayersModel {
  // Register the custom layer for serialization purposes
  // tf.serialization.registerClass(AddScalarLayer);

  const input: tf.SymbolicTensor = tf.input({ shape: [256, 256, 3] });
  /*
    // Add the first scalar (e.g., 1) using the custom layer
    let result = new AddScalarLayer(1, "firstAdd").apply(input) as tf.SymbolicTensor;
  
    // Add the second scalar (e.g., another scalar) using another custom layer
    result = new AddScalarLayer(2, "secondAdd").apply(result) as tf.SymbolicTensor;
  */

  const convLayerSettings = {
    filters: 1,
    kernelSize: 4,
    strides: 4,
    padding: "same" as const,
  };

  // Add 2d conv layer
  let result = tf.layers.conv2d({ ...convLayerSettings, filters: 40 }).apply(input) as tf.SymbolicTensor;

  for (let layerNum = 0; layerNum < numConvLayers - 1; layerNum++) {
    // Add another 2d conv layer
    result = tf.layers.conv2d(convLayerSettings).apply(result) as tf.SymbolicTensor;
  }

  // Create the model
  const model: tf.LayersModel = tf.model({ inputs: input, outputs: result });

  return model;
}
