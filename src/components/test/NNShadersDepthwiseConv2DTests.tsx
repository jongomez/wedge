import { convertShapeToTexture2DShape } from "@/lib/nnshaders/buffersAndTextures";
import { defaultOptions } from "@/lib/nnshaders/constants";
import { createNNShaders } from "@/lib/nnshaders/create";
import { padChannels } from "@/lib/nnshaders/transforms";
import { NNShadersOptions } from "@/lib/nnshaders/types";
import * as tf from '@tensorflow/tfjs';
import { expect } from "chai";
import { FC } from "react";
import { Test } from "./Test";
import { TestGroup } from "./TestGroup";
import { compareTensors, createSequentialTensor } from "./testHelpers";

const defaultOptionsWithoutBatchDim: NNShadersOptions = {
  ...defaultOptions,
  hasBatchDimension: false
}

type DepthwiseConvLayerArgs = {
  kernelSize: number,
  depthMultiplier: number,
  strides: number,
  padding: "valid" | "same",
  weights?: tf.Tensor[],
  kernelInitializer?: any,
  biasInitializer?: any,
}

async function predictAndCompare(
  input: tf.SymbolicTensor,
  depthwiseConvLayerArgs: DepthwiseConvLayerArgs,
  inputDimension: number,
  inputDepth: number,
  numConvLayers: number = 1,
  nnShadersOptions?: NNShadersOptions) {

  let result = tf.layers.depthwiseConv2d(depthwiseConvLayerArgs).apply(input) as tf.SymbolicTensor;

  for (let i = 0; i < numConvLayers - 1; i++) {
    result = tf.layers.depthwiseConv2d(depthwiseConvLayerArgs).apply(result) as tf.SymbolicTensor;
  }

  // result = tf.layers.conv2d(convLayerArgs).apply(result) as tf.SymbolicTensor;
  const model: tf.LayersModel = tf.model({ inputs: input, outputs: result });

  const nns = await createNNShaders(model, nnShadersOptions || defaultOptionsWithoutBatchDim);

  // const inputTensor = createSequentialTensor([1, inputDimension, inputDimension, inputDepth]);
  const inputTensor = tf.ones([1, inputDimension, inputDimension, inputDepth]);

  // console.log("inputTensor")
  // tf.print(inputTensor.squeeze([0]));

  // Get prediction from TensorFlow.js model
  const tfjsPrediction = model.predict(inputTensor) as tf.Tensor;

  const inputShape = [inputDimension, inputDimension, inputDepth]
  const channelPaddedInput = padChannels(inputTensor, "testInput");
  const [textWidth, textHeight, _] = convertShapeToTexture2DShape(channelPaddedInput.shape, "testInput");
  let channelPaddedAndTexturePadded = new Float32Array(textWidth * textHeight * 4);
  channelPaddedAndTexturePadded.set(channelPaddedInput.dataSync(), 0);

  const nnsPrediction = nns.predict([channelPaddedAndTexturePadded]) as tf.Tensor;

  return compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPrediction, 0.1);
}

type DepthwiseConvLayerTestArgs = {
  kernelSize: number,
  inputDepth: number,
  inputDimension?: number,
  useInitializers: boolean,
  numConvLayers?: number,
  strides?: number,
  nnShadersOptions?: NNShadersOptions
}

async function createDepthwiseConvLayerTest({
  kernelSize,
  inputDepth,
  inputDimension = 3,
  useInitializers,
  numConvLayers = 1,
  strides = 1,
  nnShadersOptions = defaultOptionsWithoutBatchDim
}: DepthwiseConvLayerTestArgs) {
  const input: tf.SymbolicTensor = tf.input({ shape: [inputDimension, inputDimension, inputDepth] });

  // XXX: Only 1 is supported for now. depthMultiplier defines how many filters to apply to each input channel.
  const depthMultiplier = 1;

  let depthwiseConvLayerArgs: DepthwiseConvLayerArgs = {
    kernelSize,
    strides,
    padding: "same",
    depthMultiplier,
  };

  let kernelWeights: tf.Tensor;

  if (useInitializers) {
    // Use constant initializers
    depthwiseConvLayerArgs.kernelInitializer = tf.initializers.constant({ value: 2.0 });
    depthwiseConvLayerArgs.biasInitializer = tf.initializers.constant({ value: 1.0 });
  } else {
    // Use sequential tensor weights
    kernelWeights = createSequentialTensor([kernelSize, kernelSize, inputDepth, depthMultiplier]);
    const biasWeights = createSequentialTensor([inputDepth * depthMultiplier]);
    depthwiseConvLayerArgs.weights = [kernelWeights, biasWeights];

    console.log("kernelWeights with format HWCN:");
    tf.print(kernelWeights);
  }

  const results = await predictAndCompare(
    input,
    depthwiseConvLayerArgs,
    inputDimension,
    inputDepth,
    numConvLayers,
    nnShadersOptions)

  expect(results).to.equal(true);
}

export const NNShadersDepthwiseConv2DTests: FC = () => {

  return <TestGroup title="NN Shaders Depthwise Conv 2D Tests">
    <Test title="kernel size 1 & input depth 1" fn={async () => {
      await createDepthwiseConvLayerTest({ kernelSize: 1, inputDepth: 1, useInitializers: true });
    }} />

    <Test title="kernel size 3 & input depth 1" fn={async () => {
      await createDepthwiseConvLayerTest({ kernelSize: 3, inputDepth: 1, useInitializers: false });
    }} />

    <Test title="kernel size 1 & input depth 5" fn={async () => {
      await createDepthwiseConvLayerTest({ kernelSize: 1, inputDepth: 5, useInitializers: false });
    }} />

    <Test title="kernel size 3 & input depth 5" fn={async () => {
      await createDepthwiseConvLayerTest({ kernelSize: 3, inputDepth: 5, useInitializers: false });
    }} />

    <Test title="kernel size 3 & input depth 10" fn={async () => {
      await createDepthwiseConvLayerTest({ kernelSize: 3, inputDepth: 10, useInitializers: false });
    }} />

    <Test title="RENDER TARGETS - 2 render targets with inputDimension 3" fn={async () => {
      const nnShadersOptions = {
        ...defaultOptionsWithoutBatchDim,
        renderTargetBreakpoints: [
          { outputTextureElementCount: 1, numberOfRenderTargets: 2 },
        ]
      };
      await createDepthwiseConvLayerTest({
        kernelSize: 3, inputDepth: 3, inputDimension: 3, useInitializers: false,
        nnShadersOptions
      });
    }} />

    <Test title="RENDER TARGETS - 64 by 64 input of 1s, and 8 render targets" fn={async () => {
      const nnShadersOptions = {
        ...defaultOptionsWithoutBatchDim,
        renderTargetBreakpoints: [
          { outputTextureElementCount: 1, numberOfRenderTargets: 8 },
        ]
      };
      await createDepthwiseConvLayerTest({
        kernelSize: 3, inputDepth: 3, inputDimension: 64, useInitializers: false,
        nnShadersOptions
      });
    }} />

    <Test title="stride 3 test" fn={async () => {
      await createDepthwiseConvLayerTest({ kernelSize: 3, inputDepth: 3, inputDimension: 5, useInitializers: false, strides: 3 });
    }} />

    <Test title="3 sequential convs kernel size 1" fn={async () => {
      await createDepthwiseConvLayerTest({ kernelSize: 1, inputDepth: 3, inputDimension: 10, useInitializers: false, numConvLayers: 3, strides: 3 });
    }} />

    <Test title="3 sequential convs kernel size 3" fn={async () => {
      await createDepthwiseConvLayerTest({
        kernelSize: 3, inputDepth: 3, inputDimension: 3, useInitializers: false,
        numConvLayers: 3, strides: 3
      });
    }} />
  </TestGroup>
}