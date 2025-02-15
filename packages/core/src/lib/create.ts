import * as tf from '@tensorflow/tfjs';
import { Wedge } from './Wedge';
import { defaultOptions } from './constants';
import { ModelType, WedgeOptions } from './types';

export async function createFromPath(
  modelPath: string,
  modelType: ModelType = "GraphModel",
  nns: Wedge): Promise<void> {
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

export async function createWedge(
  graphModelPathOrLayersModel: string | tf.LayersModel,
  options: WedgeOptions = defaultOptions): Promise<Wedge> {
  const nns = new Wedge(options);

  if (typeof graphModelPathOrLayersModel === "string") {
    await createFromPath(graphModelPathOrLayersModel, "GraphModel", nns)
  } else {
    nns.layersModel = graphModelPathOrLayersModel;
  }

  return nns;
}
