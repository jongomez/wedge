import * as tf from '@tensorflow/tfjs';
import { NamedTensorMap } from '@tensorflow/tfjs-converter/dist/data/types';
import { Node } from '@tensorflow/tfjs-converter/dist/operations/types';
import { createAndBindFramebuffer, updateFramebufferTextureLayer } from './buffersAndTextures';
import { defaultOptions } from './constants';
import { createDummyInputsNamedTensorMap, createOpNodeMapPrograms, graphModelExecuteSetup, processLayersModel } from './modelHelpers';
import { initWebGL } from './setupShadersAndWebGL';
import { removePadChannels } from './transforms';
import { NodeWebGLDataMap, WebGLDataTextureArray, WebGLOpNodeMap, WebGLOpNodeWithProgramMap, WedgeOptions } from './types';
import { initWebGLData, updateUniformsForProgram } from './webGLData';

/*

Info:
loadGraphModel source code:
https://github.com/tensorflow/tfjs/blob/f0f981fe306bf548e300536aca485c0ffdd6619e/tfjs-converter/src/executor/graph_model.ts#L624

*/


export class Wedge {
  public graphModel: tf.GraphModel | null = null;
  public canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
  public executor: any = null;
  public gl: WebGL2RenderingContext;
  public maxColorAttachments: number;

  public frameBuffer: WebGLFramebuffer | null = null;
  public orderedNodes: Node[] = [];
  public nodeWebGLDataMap: NodeWebGLDataMap = new Map();
  public opNodeMap: WebGLOpNodeMap = new Map();
  public opNodeWithProgramMap: WebGLOpNodeWithProgramMap = new Map();
  public inputsNamedTensorMap: NamedTensorMap = {};
  public inputTensorNames: Set<string> = new Set();

  public initialSetupComplete = false;

  public finalOutputData: WebGLDataTextureArray | null = null;

  public layersModel: tf.LayersModel | null = null;

  constructor(
    public options: WedgeOptions = defaultOptions,
  ) {
    // 1st step - create offscreen canvas.
    const { canvas, gl, maxColorAttachments } = initWebGL(
      this.options.canvasWidth,
      this.options.canvasHeight,
      this.options.viewportMaxSize);

    this.canvas = canvas;
    this.gl = gl;
    this.maxColorAttachments = maxColorAttachments;

    for (const renderTargetBreakpoint of this.options.renderTargetBreakpoints) {
      const numberOfRenderTargets = renderTargetBreakpoint.numberOfRenderTargets;
      if (numberOfRenderTargets > this.maxColorAttachments) {
        throw new Error("Error: numberOfRenderTargets is greater than maxColorAttachments. numberOfRenderTargets: " + numberOfRenderTargets + ", maxColorAttachments: " + this.maxColorAttachments);
      }
    }
  }

  readOutput(): Float32Array {
    // hmmm maybe this checkFramebufferStatus call is not necessary?
    // if (this.gl.checkFramebufferStatus(this.gl.FRAMEBUFFER) !== this.gl.FRAMEBUFFER_COMPLETE) {
    //   console.error('readOutput - Framebuffer not complete');
    //   return new Float32Array(0);
    // }

    let [outputTexturesWidth, outputTexturesHeight, numberOfTextures, _] = this.finalOutputData!.RGBATextureShape;

    if (!outputTexturesWidth || !outputTexturesHeight || !numberOfTextures) {
      throw new Error("Error: outputTexturesWidth, outputTexturesHeight, or numberOfTextures is not defined.");
    }

    let sizePerLayer = outputTexturesWidth * outputTexturesHeight * 4;
    let output = new Float32Array(sizePerLayer * numberOfTextures);

    for (let layer = 0; layer < numberOfTextures; layer++) {
      // Reattach the layer to the framebuffer for reading.
      this.gl.framebufferTextureLayer(
        this.gl.FRAMEBUFFER,
        this.gl.COLOR_ATTACHMENT0 + layer,
        this.finalOutputData!.texture,
        0,  // mipmap level
        layer
      );

      // Set the read buffer to the current attachment.
      this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0 + layer);

      // Read the pixels from the currently attached layer.
      let layerData = new Float32Array(sizePerLayer);
      this.gl.readPixels(0, 0, outputTexturesWidth, outputTexturesHeight, this.gl.RGBA, this.gl.FLOAT, layerData);

      // Copy the layer data into the output array at the correct offset.
      output.set(layerData, layer * sizePerLayer);
    }

    return output;
  }



  // The LayersModel's execute function is overridden. Original model's execute function:
  // https://github.com/tensorflow/tfjs/blob/master/tfjs-layers/src/engine/training.ts#L936
  layersModelExecute(inputRawData: ArrayBufferView[]): Float32Array {
    if (!this.layersModel) {
      throw new Error("layersModelExecute - error: layersModel is not defined.");
    }

    if (!this.initialSetupComplete) {
      this.inputsNamedTensorMap = createDummyInputsNamedTensorMap(this.layersModel);
      this.inputTensorNames = new Set(Object.keys(this.inputsNamedTensorMap));

      const { layersModelWeightMap, orderedNodes } = processLayersModel(this.layersModel, this.inputsNamedTensorMap, this.options.hasBatchDimension);
      this.orderedNodes = orderedNodes;

      const { opNodeMap, nodeWebGLDataMap } = initWebGLData(
        this.gl,
        layersModelWeightMap,
        this.orderedNodes,
        "LayersModel",
        this.options
      );

      // Create WebGL programs for all the operations.
      this.orderedNodes = orderedNodes;
      this.opNodeMap = opNodeMap;
      this.opNodeWithProgramMap = createOpNodeMapPrograms(opNodeMap, this.gl, layersModelWeightMap);
      this.nodeWebGLDataMap = nodeWebGLDataMap;
      this.frameBuffer = createAndBindFramebuffer(this.gl);
      this.initialSetupComplete = true;
    }

    const result = this.runOpNodes(inputRawData);

    // console.log("result float 32 array:")
    // console.log(result)

    return result;
  }


  initializeGraphModel(): void {
    if (!this.graphModel) {
      throw new Error("initializeGraphModel - error: graphModel is not defined.");
    }

    if (this.graphModel["resourceIdToCapturedInput"] == null) {
      this.graphModel["setResourceIdToCapturedInput"](this.graphModel["executeInitializerGraph"]());
    }

    this.inputsNamedTensorMap = createDummyInputsNamedTensorMap(this.graphModel);
    this.inputTensorNames = new Set(Object.keys(this.inputsNamedTensorMap));
    const outputs: string[] = this.graphModel["normalizeOutputs"](this.graphModel.outputNodes);

    this.orderedNodes = graphModelExecuteSetup(
      this.executor,
      this.options.hasBatchDimension,
      this.inputsNamedTensorMap,
      outputs)

    const { opNodeMap, nodeWebGLDataMap } = initWebGLData(
      this.gl,
      this.executor.weightMap,
      this.orderedNodes,
      "GraphModel",
      this.options);

    // Create WebGL programs for all the operations.
    this.opNodeWithProgramMap = createOpNodeMapPrograms(opNodeMap, this.gl, this.executor.weightMap);
    this.opNodeMap = opNodeMap;
    this.nodeWebGLDataMap = nodeWebGLDataMap;
    this.frameBuffer = createAndBindFramebuffer(this.gl);
    this.initialSetupComplete = true;
  }

  // This is a modified version of both the graph model and the executor's execute function.
  // https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/src/executor/graph_model.ts#L507
  // https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/src/executor/graph_executor.ts#L231
  graphModelExecute(inputRawData: ArrayBufferView[]): ArrayBufferView {
    if (!this.initialSetupComplete) {
      this.initializeGraphModel();
    }

    const result = this.runOpNodes(inputRawData);

    return result;
  }

  runOpNodes(inputRawData: ArrayBufferView[]): Float32Array {
    if (!this.gl) {
      throw new Error("runOpNodes - error: gl is not defined.");
    }

    //
    ////
    ////// Run all the WebGL programs.
    let opNodesRan = 0;
    this.opNodeWithProgramMap.forEach((opNodeWithProgam, opNodeName) => {
      this.gl.useProgram(opNodeWithProgam.programInfo.program);

      updateUniformsForProgram(this.gl, opNodeWithProgam, this.inputTensorNames, inputRawData, this.options);
      updateFramebufferTextureLayer(this.gl, opNodeWithProgam.opNode.output!);

      // There are 2 triangles that make up the square that covers the texture / canvas. 
      // 2 triangles means 6 vertices - hence the following 6:
      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
      this.finalOutputData = opNodeWithProgam.opNode.output;

      opNodesRan++;
    });

    // Read the output from the framebuffer
    const outputData = this.readOutput();

    // const dummyArray = new Float32Array(100);

    return outputData;
  }

  // tfjs predict function:
  // https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/src/executor/graph_model.ts#L357
  predict(
    inputRawData: ArrayBufferView[],
    returnFlatArray = false): Float32Array {
    let outputData: Float32Array;

    // Step 1 - Run the model.
    if (this.graphModel) {
      outputData = this.graphModelExecute(inputRawData) as unknown as Float32Array;
    } else if (this.layersModel) {
      outputData = this.layersModelExecute(inputRawData);
    } else {
      throw new Error("predict - error: no model defined.");
    }

    // Step 2 - Process the prediction.
    if (this.finalOutputData !== null) {
      if (returnFlatArray) {
        // if (!!true) {
        console.log("outputData", outputData);
        return outputData;
      } else {
        if (this.options.transformations.padChannels) {
          outputData = removePadChannels(
            outputData,
            this.finalOutputData.originalElementCount,
            this.finalOutputData.originalShape);
        }

        // Only get originalElementCount elements out of the outputData float32array.
        outputData = outputData.slice(0, this.finalOutputData.originalElementCount);

        return outputData;
      }
    } else {
      throw new Error("predict - error: finalOutputData is null.");
    }
  }
}
