"use client";

import { compareTensors, createSequentialTensor } from "@wedge/core/tests/testHelpers";
import { convertShapeToTexture2DShape } from "@wedge/core/backends/webgl/buffersAndTextures";
import { defaultOptions } from "@wedge/core/constants";
import { createWedge } from "@wedge/core/create";
import { padChannels } from "@wedge/core/transforms";
import { WedgeOptions } from "@wedge/core/backends/webgl/types";
import * as tfOriginal from '@tensorflow/tfjs';
import { expect, Test, TestContainer } from "react-browser-tests";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tf = tfOriginal as any;

const defaultOptionsWithoutBatchDim: WedgeOptions = {
  ...defaultOptions,
  hasBatchDimension: false
}

type ConvLayerArgs = {
  kernelSize: number,
  filters: number,
  padding: "valid" | "same",
  weights?: any[],
  kernelInitializer?: any,
  biasInitializer?: any,
  strides?: number,
  useBias?: boolean,
}

type InputValueType = 'ones' | 'zeros' | 'sequential' | 'negative' | 'alternating';
type WeightValueType = 'ones' | 'zeros' | 'sequential' | 'negative' | 'small';

function createInputTensor(shape: number[], valueType: InputValueType = 'ones'): any {
  switch (valueType) {
    case 'ones': return tf.ones(shape);
    case 'zeros': return tf.zeros(shape);
    case 'sequential': return createSequentialTensor(shape);
    case 'negative': return createSequentialTensor(shape, -10);
    case 'alternating':
      const size = shape.reduce((a, b) => a * b, 1);
      const data = new Float32Array(size);
      for (let i = 0; i < size; i++) data[i] = i % 2 === 0 ? 1 : -1;
      return tf.tensor(data, shape);
    default: return tf.ones(shape);
  }
}

function createWeightTensor(shape: number[], valueType: WeightValueType = 'sequential'): any {
  switch (valueType) {
    case 'ones': return tf.ones(shape);
    case 'zeros': return tf.zeros(shape);
    case 'sequential': return createSequentialTensor(shape);
    case 'negative': return createSequentialTensor(shape, -5);
    case 'small':
      const size = shape.reduce((a, b) => a * b, 1);
      const data = new Float32Array(size);
      for (let i = 0; i < size; i++) data[i] = 0.01 + (i % 10) * 0.01;
      return tf.tensor(data, shape);
    default: return createSequentialTensor(shape);
  }
}

type ConvLayerTestArgs = {
  kernelSize: number,
  inputDepth: number,
  filters: number,
  inputDimension?: number,
  useInitializers?: boolean,
  numConvLayers?: number,
  strides?: number,
  padding?: "valid" | "same",
  useBias?: boolean,
  nnShadersOptions?: WedgeOptions,
  inputValueType?: InputValueType,
  weightValueType?: WeightValueType,
}

async function createConvLayerTest({
  kernelSize,
  inputDepth,
  filters,
  inputDimension = 5,
  useInitializers = false,
  numConvLayers = 1,
  strides = 1,
  padding = "same",
  useBias = true,
  nnShadersOptions = defaultOptionsWithoutBatchDim,
  inputValueType = 'ones',
  weightValueType = 'sequential',
}: ConvLayerTestArgs) {
  const input = tf.input({ shape: [inputDimension, inputDimension, inputDepth] });

  const convLayerArgs: ConvLayerArgs = {
    filters,
    kernelSize,
    strides,
    padding,
    useBias,
  };

  if (useInitializers) {
    convLayerArgs.kernelInitializer = tf.initializers.constant({ value: 2.0 });
    if (useBias) convLayerArgs.biasInitializer = tf.initializers.constant({ value: 1.0 });
  } else {
    const kernelWeights = createWeightTensor([kernelSize, kernelSize, inputDepth, filters], weightValueType);
    if (useBias) {
      const biasWeights = createWeightTensor([filters], weightValueType);
      convLayerArgs.weights = [kernelWeights, biasWeights];
    } else {
      convLayerArgs.weights = [kernelWeights];
    }
  }

  let result = tf.layers.conv2d(convLayerArgs).apply(input);
  for (let i = 0; i < numConvLayers - 1; i++) {
    result = tf.layers.conv2d(convLayerArgs).apply(result);
  }

  const model = tf.model({ inputs: input, outputs: result });
  const nns = await createWedge(model, nnShadersOptions);

  const inputTensor = createInputTensor([1, inputDimension, inputDimension, inputDepth], inputValueType);
  const tfjsPrediction = model.predict(inputTensor);

  const inputForWedge = nnShadersOptions.hasBatchDimension ? inputTensor : tf.squeeze(inputTensor, [0]);
  const channelPaddedInput = padChannels(inputForWedge, "testInput");
  const [textWidth, textHeight] = convertShapeToTexture2DShape(channelPaddedInput.shape, "testInput");
  const channelPaddedAndTexturePadded = new Float32Array(textWidth * textHeight * 4);
  channelPaddedAndTexturePadded.set(channelPaddedInput.dataSync(), 0);

  const nnsPrediction = nns.predict([channelPaddedAndTexturePadded]);
  const nnsPredictionTensor = tf.tensor(nnsPrediction, nns.finalOutputData!.originalShape);

  const results = compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPredictionTensor, 0.1);
  expect(results).to.equal(true);
}

export function Conv2DTests() {
  return <TestContainer>
    {/* KERNEL SIZE TESTS */}
    <Test title="1x1 kernel" fn={async () => {
      await createConvLayerTest({ kernelSize: 1, inputDepth: 4, filters: 4 });
    }} />
    <Test title="2x2 kernel" fn={async () => {
      await createConvLayerTest({ kernelSize: 2, inputDepth: 4, filters: 4 });
    }} />
    <Test title="3x3 kernel" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4 });
    }} />
    <Test title="5x5 kernel" fn={async () => {
      await createConvLayerTest({ kernelSize: 5, inputDepth: 4, filters: 8, inputDimension: 7 });
    }} />
    <Test title="7x7 kernel" fn={async () => {
      await createConvLayerTest({ kernelSize: 7, inputDepth: 4, filters: 8, inputDimension: 9 });
    }} />

    {/* CHANNEL DEPTH TESTS */}
    <Test title="depth 1 → 4" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 1, filters: 4 });
    }} />
    <Test title="depth 4 → 8" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 8 });
    }} />
    <Test title="depth 8 → 16" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 8, filters: 16 });
    }} />
    <Test title="depth 16 → 32" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 16, filters: 32 });
    }} />
    <Test title="depth 32 → 64" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 32, filters: 64 });
    }} />
    <Test title="depth 64 → 32 (reduction)" fn={async () => {
      await createConvLayerTest({ kernelSize: 1, inputDepth: 64, filters: 32, inputDimension: 4 });
    }} />

    {/* STRIDE TESTS */}
    <Test title="stride 2" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 8, inputDimension: 8, strides: 2 });
    }} />
    <Test title="stride 3" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, inputDimension: 9, strides: 3 });
    }} />
    <Test title="stride 4" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 8, inputDimension: 16, strides: 4 });
    }} />

    {/* PADDING TESTS */}
    <Test title="valid padding" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, padding: "valid" });
    }} />
    <Test title="valid padding + stride 2" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 8, inputDimension: 10, padding: "valid", strides: 2 });
    }} />

    {/* BIAS TESTS */}
    <Test title="without bias" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, useBias: false });
    }} />
    <Test title="without bias, deep" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 16, filters: 32, useBias: false });
    }} />

    {/* DIMENSION TESTS */}
    <Test title="7x7 input (prime)" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, inputDimension: 7 });
    }} />
    <Test title="11x11 input (prime)" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 8, inputDimension: 11 });
    }} />
    <Test title="32x32 input" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 8, inputDimension: 32 });
    }} />
    <Test title="64x64 input" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 8, inputDimension: 64 });
    }} />

    {/* EDGE CASES */}
    <Test title="1x1 input with 1x1 kernel" fn={async () => {
      await createConvLayerTest({ kernelSize: 1, inputDepth: 4, filters: 4, inputDimension: 1 });
    }} />
    <Test title="3x3 input with 3x3 kernel" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, inputDimension: 3 });
    }} />
    <Test title="single channel in/out" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 1, filters: 1 });
    }} />
    <Test title="1 filter from 32 channels" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 32, filters: 1 });
    }} />

    {/* VALUE PATTERN TESTS */}
    <Test title="zero input" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, inputValueType: 'zeros' });
    }} />
    <Test title="sequential input" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, inputValueType: 'sequential' });
    }} />
    <Test title="negative input" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, inputValueType: 'negative' });
    }} />
    <Test title="negative weights" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, weightValueType: 'negative' });
    }} />
    <Test title="small weights" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, weightValueType: 'small' });
    }} />

    {/* SEQUENTIAL LAYER TESTS */}
    <Test title="2 sequential convs" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, inputDimension: 7, numConvLayers: 2 });
    }} />
    <Test title="3 sequential convs" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, inputDimension: 9, numConvLayers: 3 });
    }} />
    <Test title="3 convs + stride 2" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 4, inputDimension: 32, numConvLayers: 3, strides: 2 });
    }} />

    {/* RENDER TARGET TESTS */}
    <Test title="2 render targets" fn={async () => {
      await createConvLayerTest({
        kernelSize: 3, inputDepth: 4, filters: 4,
        nnShadersOptions: {
          ...defaultOptionsWithoutBatchDim,
          renderTargetBreakpoints: [{ outputTextureElementCount: 1, numberOfRenderTargets: 2 }]
        }
      });
    }} />
    <Test title="8 render targets on 64x64" fn={async () => {
      await createConvLayerTest({
        kernelSize: 3, inputDepth: 4, filters: 4, inputDimension: 64,
        nnShadersOptions: {
          ...defaultOptionsWithoutBatchDim,
          renderTargetBreakpoints: [{ outputTextureElementCount: 1, numberOfRenderTargets: 8 }]
        }
      });
    }} />

    {/* STRESS TESTS */}
    <Test title="64 filters" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 4, filters: 64 });
    }} />
    <Test title="64 input channels" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 64, filters: 4 });
    }} />
    <Test title="32 → 32 channels" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 32, filters: 32 });
    }} />
    <Test title="large + deep + stride" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 16, filters: 32, inputDimension: 32, strides: 2 });
    }} />
  </TestContainer>
}
