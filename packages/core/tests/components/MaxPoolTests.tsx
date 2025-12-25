"use client"

import { compareTensors, createSequentialTensor } from "@wedge/core/tests/testHelpers";
import { convertShapeToTexture2DShape } from "@wedge/core/backends/webgl/buffersAndTextures";
import { createWedge } from "@wedge/core/create";
import { padChannels } from "@wedge/core/transforms";
import { defaultOptions } from "@wedge/core/constants";
import { WedgeOptions } from "@wedge/core/backends/webgl/types";
import * as tfOriginal from '@tensorflow/tfjs';
import { expect, Test, TestContainer } from "react-browser-tests";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tf = tfOriginal as any;

const defaultOptionsWithoutBatchDim: WedgeOptions = {
  ...defaultOptions,
  hasBatchDimension: false
}

type MaxPoolTestArgs = {
  inputHeight: number;
  inputWidth: number;
  inputDepth: number;
  poolSize: number | [number, number];
  strides?: number | [number, number];
  padding?: "valid" | "same";
  sequenceStart?: number;
  nnShadersOptions?: WedgeOptions;
}

async function createMaxPoolTest({
  inputHeight,
  inputWidth,
  inputDepth,
  poolSize,
  strides,
  padding = "valid",
  sequenceStart = 1,
  nnShadersOptions = defaultOptionsWithoutBatchDim
}: MaxPoolTestArgs) {
  const shape = [inputHeight, inputWidth, inputDepth];
  const input = tf.input({ shape });

  // Apply max pooling layer
  const poolingLayer = tf.layers.maxPooling2d({
    poolSize,
    strides: strides ?? poolSize,
    padding
  });
  const result = poolingLayer.apply(input);
  const model = tf.model({ inputs: input, outputs: result });

  const nns = await createWedge(model, nnShadersOptions);

  // Create input tensor with batch dimension
  const inputTensor = createSequentialTensor([1, inputHeight, inputWidth, inputDepth], sequenceStart);

  // Get prediction from TensorFlow.js model
  const tfjsPrediction = model.predict(inputTensor);

  // Prepare input for wedge (pad channels and texture size)
  const channelPaddedInput = padChannels(inputTensor, "testInput");
  const [textWidth, textHeight] = convertShapeToTexture2DShape(channelPaddedInput.shape, "testInput");
  const channelPaddedAndTexturePadded = new Float32Array(textWidth * textHeight * 4);
  channelPaddedAndTexturePadded.set(channelPaddedInput.dataSync(), 0);

  const nnsPrediction = nns.predict([channelPaddedAndTexturePadded]);
  const nnsPredictionTensor = tf.tensor(nnsPrediction, nns.finalOutputData!.originalShape);

  return compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPredictionTensor, 0.01);
}

export function MaxPoolTests() {
  return <TestContainer>
    <Test title="MaxPool 2x2, stride 2, valid padding - basic test" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 4,
        inputWidth: 4,
        inputDepth: 4,
        poolSize: 2,
        strides: 2,
        padding: "valid"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool 2x2, stride 2, same padding" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 4,
        inputWidth: 4,
        inputDepth: 4,
        poolSize: 2,
        strides: 2,
        padding: "same"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool 2x2, stride 1, valid padding" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 4,
        inputWidth: 4,
        inputDepth: 4,
        poolSize: 2,
        strides: 1,
        padding: "valid"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool 2x2, stride 1, same padding" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 4,
        inputWidth: 4,
        inputDepth: 4,
        poolSize: 2,
        strides: 1,
        padding: "same"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool 3x3, stride 2, valid padding" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 6,
        inputWidth: 6,
        inputDepth: 4,
        poolSize: 3,
        strides: 2,
        padding: "valid"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool 3x3, stride 3, same padding" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 6,
        inputWidth: 6,
        inputDepth: 4,
        poolSize: 3,
        strides: 3,
        padding: "same"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool with asymmetric pool size [2,3]" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 6,
        inputWidth: 6,
        inputDepth: 4,
        poolSize: [2, 3],
        strides: [2, 3],
        padding: "valid"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool larger input 8x8x8" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 8,
        inputWidth: 8,
        inputDepth: 8,
        poolSize: 2,
        strides: 2,
        padding: "valid"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool larger input 16x16x16" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 16,
        inputWidth: 16,
        inputDepth: 16,
        poolSize: 2,
        strides: 2,
        padding: "valid"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool with negative values in input" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 4,
        inputWidth: 4,
        inputDepth: 4,
        poolSize: 2,
        strides: 2,
        padding: "valid",
        sequenceStart: -10
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool non-square input 4x8x4" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 4,
        inputWidth: 8,
        inputDepth: 4,
        poolSize: 2,
        strides: 2,
        padding: "valid"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool non-square input 8x4x4" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 8,
        inputWidth: 4,
        inputDepth: 4,
        poolSize: 2,
        strides: 2,
        padding: "valid"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool 2x2, same padding, odd input size 5x5" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 5,
        inputWidth: 5,
        inputDepth: 4,
        poolSize: 2,
        strides: 2,
        padding: "same"
      });
      expect(result).to.equal(true);
    }} />

    <Test title="MaxPool 2x2, same padding, odd input size 7x7" fn={async () => {
      const result = await createMaxPoolTest({
        inputHeight: 7,
        inputWidth: 7,
        inputDepth: 4,
        poolSize: 2,
        strides: 2,
        padding: "same"
      });
      expect(result).to.equal(true);
    }} />
  </TestContainer>
}
