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

type PadTestArgs = {
  inputDimension: number;
  inputDepth: number;
  padding: number | [[number, number], [number, number]];
  nnShadersOptions?: WedgeOptions;
}

async function createPadTest({
  inputDimension,
  inputDepth,
  padding,
  nnShadersOptions = defaultOptionsWithoutBatchDim
}: PadTestArgs) {
  const input = tf.input({ shape: [inputDimension, inputDimension, inputDepth] });

  // Apply ZeroPadding2D layer
  const result = tf.layers.zeroPadding2d({ padding }).apply(input);
  const model = tf.model({ inputs: input, outputs: result });

  const nns = await createWedge(model, nnShadersOptions);

  // Create input tensor
  const inputTensor = createSequentialTensor([1, inputDimension, inputDimension, inputDepth]);

  console.log("inputTensor:");
  tf.print(inputTensor.squeeze([0]));

  // Get prediction from TensorFlow.js model
  const tfjsPrediction = model.predict(inputTensor);

  console.log("tfjsPrediction:");
  tf.print(tfjsPrediction.squeeze([0]));

  const inputShape = [inputDimension, inputDimension, inputDepth];
  const channelPaddedInput = padChannels(inputTensor, "testInput");
  const [textWidth, textHeight, _] = convertShapeToTexture2DShape(channelPaddedInput.shape, "testInput");
  let channelPaddedAndTexturePadded = new Float32Array(textWidth * textHeight * 4);
  channelPaddedAndTexturePadded.set(channelPaddedInput.dataSync(), 0);

  const nnsPrediction = nns.predict([channelPaddedAndTexturePadded]);
  const nnsPredictionTensor = tf.tensor(nnsPrediction, nns.finalOutputData!.originalShape);

  console.log("nnsPrediction:");
  tf.print(nnsPredictionTensor);

  return compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPredictionTensor, 0.01);
}

export function PadTests() {
  return <TestContainer>
    <Test title="symmetric padding 1 with 3x3 input depth 4" fn={async () => {
      const results = await createPadTest({
        inputDimension: 3,
        inputDepth: 4,
        padding: 1
      });
      expect(results).to.equal(true);
    }} />

    <Test title="symmetric padding 2 with 4x4 input depth 8" fn={async () => {
      const results = await createPadTest({
        inputDimension: 4,
        inputDepth: 8,
        padding: 2
      });
      expect(results).to.equal(true);
    }} />

    <Test title="asymmetric padding [[1,2], [2,1]] with 3x3 input depth 4" fn={async () => {
      const results = await createPadTest({
        inputDimension: 3,
        inputDepth: 4,
        padding: [[1, 2], [2, 1]]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="padding 1 with 5x5 input depth 1" fn={async () => {
      const results = await createPadTest({
        inputDimension: 5,
        inputDepth: 1,
        padding: 1
      });
      expect(results).to.equal(true);
    }} />

    <Test title="padding 1 with 8x8 input depth 3" fn={async () => {
      const results = await createPadTest({
        inputDimension: 8,
        inputDepth: 3,
        padding: 1
      });
      expect(results).to.equal(true);
    }} />

    <Test title="no padding (0) with 4x4 input depth 4" fn={async () => {
      const results = await createPadTest({
        inputDimension: 4,
        inputDepth: 4,
        padding: 0
      });
      expect(results).to.equal(true);
    }} />

    <Test title="large padding 3 with 2x2 input depth 4" fn={async () => {
      const results = await createPadTest({
        inputDimension: 2,
        inputDepth: 4,
        padding: 3
      });
      expect(results).to.equal(true);
    }} />

    <Test title="RENDER TARGETS - padding 1 with 2 render targets" fn={async () => {
      const nnShadersOptions = {
        ...defaultOptionsWithoutBatchDim,
        renderTargetBreakpoints: [
          { outputTextureElementCount: 1, numberOfRenderTargets: 2 },
        ]
      };
      const results = await createPadTest({
        inputDimension: 4,
        inputDepth: 4,
        padding: 1,
        nnShadersOptions
      });
      expect(results).to.equal(true);
    }} />

    <Test title="RENDER TARGETS - padding 2 with 4 render targets" fn={async () => {
      const nnShadersOptions = {
        ...defaultOptionsWithoutBatchDim,
        renderTargetBreakpoints: [
          { outputTextureElementCount: 1, numberOfRenderTargets: 4 },
        ]
      };
      const results = await createPadTest({
        inputDimension: 8,
        inputDepth: 8,
        padding: 2,
        nnShadersOptions
      });
      expect(results).to.equal(true);
    }} />
  </TestContainer>;
}
