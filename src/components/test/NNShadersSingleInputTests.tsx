import { createNNShaders } from "@/lib/nnshaders/create";
import * as tf from '@tensorflow/tfjs';
import { expect } from "chai";
import { FC } from "react";
import { Test } from "./Test";
import { TestGroup } from "./TestGroup";
import { compareTensors, createSequentialTensor } from "./testHelpers";

export const NNShadersSingleInputTests: FC = () => {

  return <TestGroup title="NN Shaders Single Input Basic Ops Tests">
    <Test title="Relu test" fn={async () => {
      const inputHeight = 3;
      const inputWidth = 3;
      const inputDepth = 1;
      const shape = [inputHeight, inputWidth, inputDepth];

      // Start the sequence at -2 so the sequentialData has negative values,
      // making the ReLU function actually cap some values at 0.
      const sequenceStart = -2;
      const sequentialData = createSequentialTensor(shape, sequenceStart);
      const sequentialDataArray = sequentialData.dataSync();
      const sequentialInput = tf.input({ shape });

      // Add the sequential input and zero input
      let result = tf.layers.reLU().apply(sequentialInput) as tf.SymbolicTensor;

      const model: tf.LayersModel = tf.model({ inputs: sequentialInput, outputs: result });

      const nns = await createNNShaders(model);

      const unsqueezedSequentialData = tf.expandDims(sequentialData, 0);
      const tfjsPrediction = model.predict(unsqueezedSequentialData) as tf.Tensor;
      const nnsPrediction = nns.predict([sequentialDataArray]) as tf.Tensor;

      const compareResult = compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPrediction, 0.1);

      expect(compareResult).to.equal(true);
    }} />
  </TestGroup>
}