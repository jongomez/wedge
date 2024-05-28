import { defaultOptions } from "@/lib/nnshaders/constants";
import { createNNShaders } from "@/lib/nnshaders/create";
import { padChannels } from "@/lib/nnshaders/transforms";
import { NNShadersOptions } from "@/lib/nnshaders/types";
import * as tf from '@tensorflow/tfjs';
import { expect } from "chai";
import { FC } from "react";
import { Test } from "./Test";
import { TestGroup } from "./TestGroup";
import { compareTensors } from "./testHelpers";


const defaultOptionsWithoutBatchDim: NNShadersOptions = {
  ...defaultOptions,
  hasBatchDimension: false
}


export const NNShadersArithmeticTests: FC = () => {

  return <TestGroup title="NN Shaders Arithmetic Tests">

    <Test title="Same sized texture array inputs adds 1" fn={async () => {
      const inputHeight = 3;
      const inputWidth = 3;
      const inputDepth = 1;
      const shape = [inputHeight, inputWidth, inputDepth];

      const input1: tf.SymbolicTensor = tf.input({ shape });
      const input2: tf.SymbolicTensor = tf.input({ shape });

      const onesData = tf.ones([1, inputHeight, inputWidth, inputDepth]);
      const onesDataPadded = padChannels(onesData, "testInput");
      const onesDataPaddedArray = onesDataPadded.dataSync();

      // Add the sequential input and zero input
      let result = tf.layers.add().apply([input1, input2]) as tf.SymbolicTensor;

      const model: tf.LayersModel = tf.model({ inputs: [input1, input2], outputs: result });
      // Get prediction from TensorFlow.js model
      const tfjsPrediction = model.predict([onesData, onesData]) as tf.Tensor;

      const nns = await createNNShaders(model, defaultOptionsWithoutBatchDim);
      const nnsPrediction = nns.predict([onesDataPaddedArray, onesDataPaddedArray]);
      const nnsPredictionTensor = tf.tensor(nnsPrediction, nns.finalOutputData!.originalShape);

      const comparisonRes = compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPredictionTensor, 0.1);

      expect(comparisonRes).to.equal(true);
    }} />

    <Test title="TODO Scalar constant and array texture adds" skip fn={async () => {
      const scalarConstant = 5;

      const inputHeight = 3;
      const inputWidth = 3;
      const inputDepth = 1;
      const shape = [inputHeight, inputWidth, inputDepth];

      // Define an input tensor with a shape of 3x3x1
      const inputTensor: tf.SymbolicTensor = tf.input({ shape });

      // Define a second input for the constant value, note it is shaped as (1,1,1) for broadcasting
      const constantInput: tf.SymbolicTensor = tf.input({ shape: [1] });

      // Use tf.layers.add() to combine these two inputs
      const addLayer = tf.layers.add();
      const result = addLayer.apply([inputTensor, constantInput]) as tf.SymbolicTensor;

      // Create the TensorFlow.js model
      const model: tf.LayersModel = tf.model({ inputs: [inputTensor, constantInput], outputs: result });

      // Create tensors for prediction
      const onesData = tf.ones([1, inputHeight, inputWidth, inputDepth]);
      const onesDataPadded = padChannels(onesData, "testInput");
      const onesDataPaddedArray = onesDataPadded.dataSync();

      const constantValue = tf.fill([1], scalarConstant);

      // Get prediction from TensorFlow.js model
      const tfjsPrediction = model.predict([onesData, constantValue]) as tf.Tensor;

      // console.log("tfjsPrediction", tfjsPrediction.dataSync());

      const nns = await createNNShaders(model, defaultOptionsWithoutBatchDim);

      const nnsPrediction = nns.predict([onesDataPaddedArray, new Float32Array(scalarConstant)]);
      const nnsPredictionTensor = tf.tensor(nnsPrediction, nns.finalOutputData!.originalShape);

      const comparisonRes = compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPredictionTensor, 0.1);

      expect(comparisonRes).to.equal(true);
    }} />
  </TestGroup>
}