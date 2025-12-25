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

type ReshapeTestArgs = {
  inputShape: number[];  // [H, W, C] without batch
  targetShape: number[]; // [H', W', C'] without batch
  nnShadersOptions?: WedgeOptions;
}

async function createReshapeTest({
  inputShape,
  targetShape,
  nnShadersOptions = defaultOptionsWithoutBatchDim
}: ReshapeTestArgs) {
  // Create LayersModel with Reshape layer
  const input = tf.input({ shape: inputShape });
  const result = tf.layers.reshape({ targetShape }).apply(input);
  const model = tf.model({ inputs: input, outputs: result });

  const nns = await createWedge(model, nnShadersOptions);

  // Create input tensor with batch dimension [1, H, W, C]
  const inputTensor = createSequentialTensor([1, ...inputShape]);

  console.log("inputTensor shape:", inputTensor.shape);
  console.log("inputTensor:");
  tf.print(inputTensor.squeeze([0]));

  // Get prediction from TensorFlow.js model
  const tfjsPrediction = model.predict(inputTensor);

  console.log("tfjsPrediction shape:", tfjsPrediction.shape);
  console.log("tfjsPrediction:");
  tf.print(tfjsPrediction.squeeze([0]));

  // Prepare input for Wedge (channel padding + texture padding)
  const channelPaddedInput = padChannels(inputTensor, "testInput");
  const [textWidth, textHeight, _] = convertShapeToTexture2DShape(channelPaddedInput.shape, "testInput");
  let channelPaddedAndTexturePadded = new Float32Array(textWidth * textHeight * 4);
  channelPaddedAndTexturePadded.set(channelPaddedInput.dataSync(), 0);

  const nnsPrediction = nns.predict([channelPaddedAndTexturePadded]);
  const nnsPredictionTensor = tf.tensor(nnsPrediction, nns.finalOutputData!.originalShape);

  console.log("nnsPrediction shape:", nns.finalOutputData!.originalShape);
  console.log("nnsPrediction:");
  tf.print(nnsPredictionTensor);

  return compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPredictionTensor, 0.01);
}

export function ReshapeTests() {
  return <TestContainer>
    {/* Basic reshape tests - same shape (identity) */}
    <Test title="identity reshape [4,4,4] -> [4,4,4]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 4],
        targetShape: [4, 4, 4]
      });
      expect(results).to.equal(true);
    }} />

    {/* Spatial dimension changes */}
    <Test title="spatial reshape [4,4,4] -> [8,2,4]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 4],
        targetShape: [8, 2, 4]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="spatial reshape [4,4,4] -> [2,8,4]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 4],
        targetShape: [2, 8, 4]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="spatial reshape [8,8,4] -> [16,4,4]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [8, 8, 4],
        targetShape: [16, 4, 4]
      });
      expect(results).to.equal(true);
    }} />

    {/* Channel dimension changes */}
    <Test title="channel reshape [4,4,4] -> [4,8,2]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 4],
        targetShape: [4, 8, 2]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="channel reshape [4,4,4] -> [8,4,2]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 4],
        targetShape: [8, 4, 2]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="channel reshape [4,4,8] -> [4,4,8] (identity with 8 channels)" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 8],
        targetShape: [4, 4, 8]
      });
      expect(results).to.equal(true);
    }} />

    {/* Non-power-of-2 dimensions */}
    <Test title="non-power-of-2 reshape [3,5,2] -> [5,3,2]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [3, 5, 2],
        targetShape: [5, 3, 2]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="non-power-of-2 reshape [6,5,4] -> [10,3,4]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [6, 5, 4],
        targetShape: [10, 3, 4]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="prime dimensions [7,3,2] -> [3,7,2]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [7, 3, 2],
        targetShape: [3, 7, 2]
      });
      expect(results).to.equal(true);
    }} />

    {/* Small tensors */}
    <Test title="small tensor [2,2,1] -> [4,1,1]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [2, 2, 1],
        targetShape: [4, 1, 1]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="small tensor [2,2,2] -> [2,4,1]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [2, 2, 2],
        targetShape: [2, 4, 1]
      });
      expect(results).to.equal(true);
    }} />

    {/* Larger tensors */}
    <Test title="larger tensor [16,16,3] -> [32,8,3]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [16, 16, 3],
        targetShape: [32, 8, 3]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="larger tensor [8,8,16] -> [16,4,16]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [8, 8, 16],
        targetShape: [16, 4, 16]
      });
      expect(results).to.equal(true);
    }} />

    {/* Edge case: single channel */}
    <Test title="single channel [4,4,1] -> [8,2,1]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 1],
        targetShape: [8, 2, 1]
      });
      expect(results).to.equal(true);
    }} />

    {/* Edge case: 3 channels (RGB-like) */}
    <Test title="RGB-like [8,8,3] -> [4,16,3]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [8, 8, 3],
        targetShape: [4, 16, 3]
      });
      expect(results).to.equal(true);
    }} />

    {/* Flatten-like (but keeping 3D shape) */}
    <Test title="flatten-like [4,4,4] -> [64,1,1]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 4],
        targetShape: [64, 1, 1]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="flatten-like [4,4,4] -> [1,64,1]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 4],
        targetShape: [1, 64, 1]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="flatten-like [4,4,4] -> [1,1,64]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [4, 4, 4],
        targetShape: [1, 1, 64]
      });
      expect(results).to.equal(true);
    }} />

    {/* Render target tests */}
    <Test title="RENDER TARGETS - reshape with 2 render targets" fn={async () => {
      const nnShadersOptions = {
        ...defaultOptionsWithoutBatchDim,
        renderTargetBreakpoints: [
          { outputTextureElementCount: 1, numberOfRenderTargets: 2 },
        ]
      };
      const results = await createReshapeTest({
        inputShape: [4, 4, 8],
        targetShape: [8, 4, 4],
        nnShadersOptions
      });
      expect(results).to.equal(true);
    }} />

    <Test title="RENDER TARGETS - reshape with 4 render targets" fn={async () => {
      const nnShadersOptions = {
        ...defaultOptionsWithoutBatchDim,
        renderTargetBreakpoints: [
          { outputTextureElementCount: 1, numberOfRenderTargets: 4 },
        ]
      };
      const results = await createReshapeTest({
        inputShape: [8, 8, 16],
        targetShape: [16, 8, 8],
        nnShadersOptions
      });
      expect(results).to.equal(true);
    }} />

    {/* Complex reshapes */}
    <Test title="complex reshape [6,6,8] -> [9,4,8]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [6, 6, 8],
        targetShape: [9, 4, 8]
      });
      expect(results).to.equal(true);
    }} />

    <Test title="complex reshape [5,5,4] -> [10,5,2]" fn={async () => {
      const results = await createReshapeTest({
        inputShape: [5, 5, 4],
        targetShape: [10, 5, 2]
      });
      expect(results).to.equal(true);
    }} />
  </TestContainer>;
}
