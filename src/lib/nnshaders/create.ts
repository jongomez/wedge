import * as tf from '@tensorflow/tfjs';
import { NNShaders } from './NNShaders';
import { defaultOptions } from './constants';
import { ModelType, NNShadersOptions } from './types';

export async function createFromPath(
  modelPath: string,
  modelType: ModelType = "GraphModel",
  nns: NNShaders): Promise<void> {
  // Load the model.
  if (modelType === "GraphModel") {
    nns.graphModel = await tf.loadGraphModel(modelPath);
    // HACK: the executor is private. But, we can still access it by using square notation:
    nns.executor = nns.graphModel["executor"];
    // this.model.summary();
  } else if (modelType === "onnx") {
    // TODO:
    throw new Error("onnx model type not supported yet");
  } else {
    throw new Error("modelType not supported. Got: " + modelType);
  }
}

export async function createNNShaders(
  graphModelPathOrLayersModel: string | tf.LayersModel,
  options: NNShadersOptions = defaultOptions): Promise<NNShaders> {
  const nns = new NNShaders(options);

  if (typeof graphModelPathOrLayersModel === "string") {
    await createFromPath(graphModelPathOrLayersModel, "GraphModel", nns)
  } else {
    nns.layersModel = graphModelPathOrLayersModel;
  }

  return nns;
}
