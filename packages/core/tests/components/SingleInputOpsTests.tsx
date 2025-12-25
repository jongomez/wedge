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

type SingleInputTestArgs = {
  inputHeight: number;
  inputWidth: number;
  inputDepth: number;
  sequenceStart: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  layer: any;
  nnShadersOptions?: WedgeOptions;
}

async function createSingleInputTest({
  inputHeight,
  inputWidth,
  inputDepth,
  sequenceStart,
  layer,
  nnShadersOptions = defaultOptionsWithoutBatchDim
}: SingleInputTestArgs) {
  const shape = [inputHeight, inputWidth, inputDepth];
  const input = tf.input({ shape });

  // Apply the layer
  const result = layer.apply(input);
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

export function SingleInputOpsTests() {

  return <TestContainer>
    <Test title="Relu test" skip fn={async () => {
      const result = await createSingleInputTest({
        inputHeight: 3,
        inputWidth: 3,
        inputDepth: 4,
        sequenceStart: -2,
        layer: tf.layers.reLU()
      });
      expect(result).to.equal(true);
    }} />

    <Test title="Relu6 test - values below 0 clamped to 0" fn={async () => {
      const result = await createSingleInputTest({
        inputHeight: 3,
        inputWidth: 3,
        inputDepth: 4,
        sequenceStart: -5,
        layer: tf.layers.reLU({ maxValue: 6 })
      });
      expect(result).to.equal(true);
    }} />

    <Test title="Relu6 test - values above 6 clamped to 6" fn={async () => {
      const result = await createSingleInputTest({
        inputHeight: 3,
        inputWidth: 3,
        inputDepth: 4,
        sequenceStart: 3,
        layer: tf.layers.reLU({ maxValue: 6 })
      });
      expect(result).to.equal(true);
    }} />

    <Test title="Relu6 test - mixed values (negative, in range, above 6)" fn={async () => {
      const result = await createSingleInputTest({
        inputHeight: 4,
        inputWidth: 4,
        inputDepth: 4,
        sequenceStart: -10,
        layer: tf.layers.reLU({ maxValue: 6 })
      });
      expect(result).to.equal(true);
    }} />

    <Test title="Sigmoid test - positive values" fn={async () => {
      const result = await createSingleInputTest({
        inputHeight: 3,
        inputWidth: 3,
        inputDepth: 4,
        sequenceStart: 0,
        layer: tf.layers.activation({ activation: 'sigmoid' })
      });
      expect(result).to.equal(true);
    }} />

    <Test title="Sigmoid test - negative values" fn={async () => {
      const result = await createSingleInputTest({
        inputHeight: 3,
        inputWidth: 3,
        inputDepth: 4,
        sequenceStart: -5,
        layer: tf.layers.activation({ activation: 'sigmoid' })
      });
      expect(result).to.equal(true);
    }} />

    <Test title="Sigmoid test - values around zero (transition region)" fn={async () => {
      const result = await createSingleInputTest({
        inputHeight: 4,
        inputWidth: 4,
        inputDepth: 4,
        sequenceStart: -8,
        layer: tf.layers.activation({ activation: 'sigmoid' })
      });
      expect(result).to.equal(true);
    }} />

    <Test title="Sigmoid test - larger input dimensions" fn={async () => {
      const result = await createSingleInputTest({
        inputHeight: 8,
        inputWidth: 8,
        inputDepth: 8,
        sequenceStart: -32,
        layer: tf.layers.activation({ activation: 'sigmoid' })
      });
      expect(result).to.equal(true);
    }} />
  </TestContainer>
}