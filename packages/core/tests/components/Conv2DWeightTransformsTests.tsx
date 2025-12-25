"use client";

import { createSequentialTensor } from "@wedge/core/tests/testHelpers";
import { conv2dWeightsTransform } from "@wedge/core/transforms";
import * as tfOriginal from '@tensorflow/tfjs';
import { expect, Test, TestContainer, TestGroup } from "react-browser-tests";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tf = tfOriginal as any;

export function Conv2DWeightTransformsTests() {
  const env = process.env.NODE_ENV

  if (env !== "development") {
    return <div>What are you doing here?</div>
  }

  // TODO: Raise active state and url param handling to this level.

  return <TestContainer>
    <TestGroup title="Conv 2D Weight Transforms">
      <Test title="layer with 2 simple filters" fn={() => {
        // The filters have dimensions 3x3x3
        const firstFilterWeights = createSequentialTensor([3, 3, 3])
        const secondFilterWeights = createSequentialTensor([3, 3, 3])
        // The final weights should have dimensions 3x3x3x2
        const finalWeights = tf.stack([firstFilterWeights, secondFilterWeights], 3)

        console.log("finalWeights shape", finalWeights.shape)

        const transformedTensor = conv2dWeightsTransform(finalWeights)
        const transfomedData = transformedTensor.dataSync()

        const firsFilterx0y0z0r = transfomedData[0]
        const firsFilterx0y0z0g = transfomedData[1]
        const firsFilterx0y0z0b = transfomedData[2]
        const firsFilterx0y0z0a = transfomedData[3]

        const secondFilterx0y0z0r = transfomedData[4]
        const secondFilterx0y0z0g = transfomedData[5]
        const secondFilterx0y0z0b = transfomedData[6]
        const secondFilterx0y0z0a = transfomedData[7]

        const firstFilterx0y0z0 = [firsFilterx0y0z0r, firsFilterx0y0z0g, firsFilterx0y0z0b, firsFilterx0y0z0a]
        const secondFilterx0y0z0 = [secondFilterx0y0z0r, secondFilterx0y0z0g, secondFilterx0y0z0b, secondFilterx0y0z0a]

        // The data format should be HWC - so the shape should be:
        // 3 for the y axis (Height)
        // 6 for the x axis (Width) - this is because we're stacking 2 3x3 filters
        // 4 for the z axis (Depth)
        expect(transformedTensor.shape).to.deep.equal([3, 6, 4])
        expect(firstFilterx0y0z0).to.deep.equal([1, 2, 3, 0])
        expect(secondFilterx0y0z0).to.deep.equal([1, 2, 3, 0])
      }} />

      <Test title="layer with 2 filters with depth > 4" fn={() => {
        // The filters have dimensions 3x3x3
        const filterDepth = 5;
        const filterSpatialDim = 3;
        const numFilters = 2;
        const firstFilterWeights = createSequentialTensor([filterSpatialDim, filterSpatialDim, filterDepth])
        const secondFilterWeights = createSequentialTensor([filterSpatialDim, filterSpatialDim, filterDepth])
        // The final weights should have dimensions 3x3x3x2
        const finalWeights = tf.stack([firstFilterWeights, secondFilterWeights], 3)

        console.log("finalWeights shape", finalWeights.shape)

        const transformedTensor = conv2dWeightsTransform(finalWeights)
        const transfomedData = transformedTensor.dataSync()

        const firsFilterx0y0z0r = transfomedData[0]
        const firsFilterx0y0z0g = transfomedData[1]
        const firsFilterx0y0z0b = transfomedData[2]
        const firsFilterx0y0z0a = transfomedData[3]

        const secondFilterx0y0z0r = transfomedData[4]
        const secondFilterx0y0z0g = transfomedData[5]
        const secondFilterx0y0z0b = transfomedData[6]
        const secondFilterx0y0z0a = transfomedData[7]

        const firstFilterx0y0z0 = [firsFilterx0y0z0r, firsFilterx0y0z0g, firsFilterx0y0z0b, firsFilterx0y0z0a]
        const secondFilterx0y0z0 = [secondFilterx0y0z0r, secondFilterx0y0z0g, secondFilterx0y0z0b, secondFilterx0y0z0a]

        // The textures array should have 4 depth values for every index along the z axis.
        // Fetching x = 0, y = 0, z = 1 in texture coords should give us the 5th depth value.
        const singleTextureNumElements = 4 * filterSpatialDim * filterSpatialDim * numFilters;
        const firstFilterx0y0z1r = transfomedData[singleTextureNumElements]
        const firstFilterx0y0z1g = transfomedData[singleTextureNumElements + 1]
        const firstFilterx0y0z1b = transfomedData[singleTextureNumElements + 2]
        const firstFilterx0y0z1a = transfomedData[singleTextureNumElements + 3]

        const secondFilterx0y0z1r = transfomedData[singleTextureNumElements + 4]
        const secondFilterx0y0z1g = transfomedData[singleTextureNumElements + 5]
        const secondFilterx0y0z1b = transfomedData[singleTextureNumElements + 6]
        const secondFilterx0y0z1a = transfomedData[singleTextureNumElements + 7]

        const firstFilterx0y0z1 = [firstFilterx0y0z1r, firstFilterx0y0z1g, firstFilterx0y0z1b, firstFilterx0y0z1a]
        const secondFilterx0y0z1 = [secondFilterx0y0z1r, secondFilterx0y0z1g, secondFilterx0y0z1b, secondFilterx0y0z1a]

        // The data format should be HWC - so the shape should be:
        // 3 for the y axis (Height)
        // 6 for the x axis (Width) - this is because we're stacking 2 3x3 filters
        // 8 for the z axis (Depth) - this is because the depth is zero padded to be divisible by 4.
        expect(transformedTensor.shape).to.deep.equal([3, 6, 8])
        expect(firstFilterx0y0z0).to.deep.equal([1, 2, 3, 4])
        expect(secondFilterx0y0z0).to.deep.equal([1, 2, 3, 4])

      }} />
    </TestGroup>
  </TestContainer>
}
