import { Model, ModelConfig } from '@/lib/types';
// Import @tensorflow/tfjs-core
import { FilesetResolver, ImageClassifier } from '@mediapipe/tasks-vision';
import type { GraphModel } from '@tensorflow/tfjs-converter';

//
//// tfjs imports
import * as tf from '@tensorflow/tfjs';

//
//// tflite imports
// import '@tensorflow/tfjs-backend-cpu';
// import * as tflite from '@tensorflow/tfjs-tflite';
// import * as tf from '@tensorflow/tfjs-core';
// import "@tensorflow/tfjs-core"

import { getModelType } from './helpers';
import { createWedge } from './wedge/create';
import { ModelType } from './wedge/types';

//
//// tfjs imports

//
//// tflite imports
// import '@tensorflow/tfjs-backend-cpu';

export async function loadModel(
  modelConfig: ModelConfig,
): Promise<Model> {
  const { backend, runtime, url } = modelConfig;

  if (runtime === "tfjs" || runtime === "tfjs-tflite") {
    return tfModelLoad(modelConfig);
  } else if (runtime === "mediapipe") {
    const vision = await FilesetResolver.forVisionTasks("/wasm/");
    // const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm");

    let delegate = null;

    if (backend === "webgl") {
      delegate = "GPU" as const;
    } else if (backend === "cpu") {
      delegate = "CPU" as const;
    } else {
      throw new Error(`Expected mediapipe backend to be 'webgl' or 'cpu', but got: ${backend}`);
    }

    // const imageClassifier = await ImageClassifier.createFromOptions(vision, {

    // Create offscreen canvas for mediapipe.
    const offscreenCanvas = document.createElement("canvas");
    // setupWebGL2ContextInterception(offscreenCanvas);

    const imageClassifier = await ImageClassifier.createFromOptions(vision, {
      baseOptions: {
        // modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite",
        modelAssetPath: "testModels/3_convs_1st_layer_with_40_filters/single_frame",
        delegate,
      },
      canvas: offscreenCanvas,
      runningMode: "VIDEO",
    });

    return imageClassifier;
  } else if (runtime === "wedge") {
    // const nns = new Wedge("3_convs_no_rescaling/model.json");
    // const nns = new Wedge("10_convs/model.json");
    // const nns = new Wedge("6_convs_with_relu_and_bias/model.json")
    // const nns = new Wedge("3_convs_1st_layer_with_40_filters/model.json")

    const modelType = getModelType(modelConfig);

    if (modelType === "tflite") {
      tfModelLoad(modelConfig, "tflite");
    }

    // const layersModel = createAllConvModel(3);
    // const nns = new Wedge(layersModel);

    // const nns = await createWedge(url);

    const nns = await createWedge("models/pose_four/model.json")
    nns.initializeGraphModel();

    return nns;
  }

  throw new Error(`Unsupported runtime: ${runtime}.`);
}

export async function tfModelLoad(
  modelConfig: ModelConfig,
  modelType: ModelType
): Promise<GraphModel> {
  const { backend, runtime } = modelConfig;
  let model: GraphModel | null = null;

  if (modelType === "tflite") {
    // tflite.setWasmPath('/wasm/');
    // setWasmPaths('/wasm/');
  }

  tf.setBackend(backend);

  if (modelType === "GraphModel") {
    //
    //// TFJS GraphModel
    model = await tf.loadGraphModel(modelConfig.url);
  } else if (modelType === "tflite") {
    //
    //// TFLite model
    // model = await tflite.loadTFLiteModel(modelConfig.url);
    throw new Error("TFLite model loading not implemented yet.");
  } else {
    throw new Error("Invalid runtime: " + runtime);
  }

  console.log("MODEL", model);
  console.log("tf.engine is:", tf.engine().backendName);

  // Return the pose estimator with the loaded model
  return model;
}