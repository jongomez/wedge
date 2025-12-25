"use client";

import { compareTensors } from "@wedge/core/tests/testHelpers";
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

export function ArithmeticTests() {

  return <TestContainer>
    <Test title="Same sized texture array inputs adds 1" fn={async () => {
      const inputHeight = 3;
      const inputWidth = 3;
      const inputDepth = 1;
      const shape = [inputHeight, inputWidth, inputDepth];

      const input1 = tf.input({ shape });
      const input2 = tf.input({ shape });

      const onesData = tf.ones([1, inputHeight, inputWidth, inputDepth]);
      const onesDataPadded = padChannels(onesData, "testInput");
      const onesDataPaddedArray = onesDataPadded.dataSync();

      // Add the sequential input and zero input
      const result = tf.layers.add().apply([input1, input2]);

      const model = tf.model({ inputs: [input1, input2], outputs: result });
      // Get prediction from TensorFlow.js model
      const tfjsPrediction = model.predict([onesData, onesData]);

      const nns = await createWedge(model, defaultOptionsWithoutBatchDim);
      const nnsPrediction = nns.predict([onesDataPaddedArray, onesDataPaddedArray]);
      const nnsPredictionTensor = tf.tensor(nnsPrediction, nns.finalOutputData!.originalShape);

      const comparisonRes = compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPredictionTensor, 0.1);

      expect(comparisonRes).to.equal(true);
    }} />

    <Test title="Scalar constant and array texture adds (broadcasting)" fn={async () => {
      const scalarConstant = 5;

      const inputHeight = 3;
      const inputWidth = 3;
      const inputDepth = 1;
      const shape = [inputHeight, inputWidth, inputDepth];

      // Define an input tensor with a shape of 3x3x1
      const inputTensor = tf.input({ shape });

      // Define a second input for the constant value, note it is shaped as (1,1,1) for broadcasting
      const constantInput = tf.input({ shape: [1] });

      // Use tf.layers.add() to combine these two inputs
      const addLayer = tf.layers.add();
      const result = addLayer.apply([inputTensor, constantInput]);

      // Create the TensorFlow.js model
      const model = tf.model({ inputs: [inputTensor, constantInput], outputs: result });

      // Create tensors for prediction
      const onesData = tf.ones([1, inputHeight, inputWidth, inputDepth]);
      const onesDataPadded = padChannels(onesData, "testInput");
      const onesDataPaddedArray = onesDataPadded.dataSync();

      const constantValue = tf.fill([1], scalarConstant);
      const constantValuePadded = padChannels(constantValue, "constantInput");
      const constantValuePaddedArray = constantValuePadded.dataSync();

      // Get prediction from TensorFlow.js model
      const tfjsPrediction = model.predict([onesData, constantValue]);

      const nns = await createWedge(model, defaultOptionsWithoutBatchDim);

      const nnsPrediction = nns.predict([onesDataPaddedArray, constantValuePaddedArray]);
      const nnsPredictionTensor = tf.tensor(nnsPrediction, nns.finalOutputData!.originalShape);

      const comparisonRes = compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPredictionTensor, 0.1);

      expect(comparisonRes).to.equal(true);
    }} />
  </TestContainer>
}




