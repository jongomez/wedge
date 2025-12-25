import { ITFGraphDef } from '../../../external/tensorflow/compiled_api';
import { DEFAULT_MODEL_NAME, GraphModel, TFHUB_SEARCH_PARAM } from '../external/tensorflow/graph_model';
import { createDummyInputsNamedTensorMap } from '../external/tensorflow/helpers';
import { browserHTTPRequest } from '../../../external/tensorflow/io/http';
import { decodeWeights } from '../../../external/tensorflow/io/io_utils';
import { getLoadHandlers } from '../../../external/tensorflow/io/router_registry';
import { IOHandler, LoadOptions, ModelArtifacts } from '../../../external/tensorflow/io/types';
import { PseudoGraphExecutor } from '../../../external/tensorflow/pseudo_graph_executor';
import { tfGraphToWedgeGraph } from '../../../external/tensorflow/transform_graph';
import { NamedTensorMap, NamedTensorsMap, TFNode } from '../../../external/tensorflow/types';
import { WedgeBase } from '../../../types';
import { OperationMapper } from '../../../external/tensorflow/ops/operation_mappers';

function convertTensorMapToTensorsMap(map: NamedTensorMap): NamedTensorsMap {
  return Object.keys(map).reduce((newMap: NamedTensorsMap, key) => {
    newMap[key] = [map[key]];
    return newMap;
  }, {});
}

function getIOHandler(modelUrl: string | IOHandler, options: LoadOptions): IOHandler {
  const path = modelUrl;
  let handler: IOHandler;

  if ((path as IOHandler).load != null) {
    // Path is an IO Handler.
    handler = path as IOHandler;
  } else if (this.loadOptions.requestInit != null) {
    handler = browserHTTPRequest(path as string, this.loadOptions);
  } else {
    const handlers =
      getLoadHandlers(path as string, this.loadOptions);
    if (handlers.length === 0) {
      // For backward compatibility: if no load handler can be found,
      // assume it is a relative http path.
      handlers.push(browserHTTPRequest(path as string, this.loadOptions));
    } else if (handlers.length > 1) {
      throw new Error(
        `Found more than one (${handlers.length}) load handlers for ` +
        `URL '${[path]}'`);
    }
    handler = handlers[0];
  }

  return handler;
}


/**
 * Load a graph model given a URL to the model definition.
 *
 * Example of loading MobileNetV2 from a URL and making a prediction with a
 * zeros input:
 *
 * ```js
 * const modelUrl =
 *    'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
 * const model = await tf.loadGraphModel(modelUrl);
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * model.predict(zeros).print();
 * ```
 *
 * Example of loading MobileNetV2 from a TF Hub URL and making a prediction with
 * a zeros input:
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * model.predict(zeros).print();
 * ```
 * @param modelUrl The url or an `io.IOHandler` that loads the model.
 * @param options Options for the HTTP request, which allows to send credentials
 *    and custom headers.
 */
/** @doc {heading: 'Models', subheading: 'Loading'} */
export async function fetchAndCreateGraphModel(
  modelUrl: string | IOHandler,
  options: LoadOptions = {}): Promise<[ITFGraphDef, ModelArtifacts]> {
  if (modelUrl == null) {
    throw new Error(
      'modelUrl in loadGraphModel() cannot be null. Please provide a url ' +
      'or an IOHandler that loads the model');
  }
  if (options == null) {
    options = {};
  }

  if (options.fromTFHub) {
    if ((modelUrl as IOHandler).load == null) {
      if (!(modelUrl as string).endsWith('/')) {
        modelUrl = (modelUrl as string) + '/';
      }
      modelUrl = `${modelUrl}${DEFAULT_MODEL_NAME}${TFHUB_SEARCH_PARAM}`;
    }
  }

  const handler = getIOHandler(modelUrl, options);
  if (handler.load == null) {
    throw new Error(
      'Cannot proceed with model loading because the IOHandler provided ' +
      'does not have the `load` method implemented.');
  }
  const artifacts = await handler.load();
  const tfGraph = artifacts.modelTopology as ITFGraphDef;


  return [tfGraph, artifacts];
}


export async function loadGraphModel(
  modelPath: string,
  wedgeInstance: WedgeBase): Promise<{
    model: GraphModel;
    executor: any;
    orderedNodes: TFNode[];
    inputsNamedTensorMap: NamedTensorMap;
    inputTensorNames: Set<string>;
  }> {
  // Load the model
  const [tfGraph, modelArtifacts] = await fetchAndCreateGraphModel(modelPath);


  if (!modelArtifacts.weightData || !modelArtifacts.weightSpecs) {
    throw new Error('Weight data or weight specs are missing');
  }

  const version = `${tfGraph.versions?.producer}.${tfGraph.versions?.minConsumer}`;


  let weightMap: NamedTensorMap | NamedTensorsMap = decodeWeights(modelArtifacts.weightData, modelArtifacts.weightSpecs);

  // this.executor =
  //   new GraphExecutor(OperationMapper.Instance.transformGraph(graph));

  const wedgeGraph = tfGraphToWedgeGraph(tfGraph, wedgeInstance);
  weightMap = convertTensorMapToTensorsMap(weightMap);

  // Get the executor (it's private but accessible)
  const executor = model["executor"];

  // Initialize resource map if needed
  if (model["resourceIdToCapturedInput"] == null) {
    model["setResourceIdToCapturedInput"](model["executeInitializerGraph"]());
  }

  // Create input tensor map and get tensor names
  const inputsNamedTensorMap = createDummyInputsNamedTensorMap(model);
  const inputTensorNames = new Set(Object.keys(inputsNamedTensorMap));

  // Get output nodes
  const outputs: string[] = model["normalizeOutputs"](model.outputNodes);

  if (!outputs) {
    throw new Error("graphModelExecuteSetup - error: outputs is not defined.");
  }


  const wedge = new Wedge();
  wedge.


    return {
    model,
      executor,
      orderedNodes,
      inputsNamedTensorMap,
      inputTensorNames
  };
}