import { createOutputTextureArray } from '@/lib/nnshaders/buffersAndTextures';
import { updateConv2DOutputDimensions } from '@/lib/nnshaders/ops/conv2D/output';
import { NNShadersOptions } from '@/lib/nnshaders/types';
import { expect } from 'chai';
import { FC } from "react";
import { defaultOptions } from '../../lib/nnshaders/constants';
import { Test } from "./Test";
import { TestGroup } from "./TestGroup";
import { createWebGLContext } from './testHelpers';

export const NNShadersOutputTests: FC = () => {

  return <TestGroup title="NN Shaders Output Tests">

    <Test title="Simple check if channels are padded" fn={() => {
      const gl = createWebGLContext();
      const originalOutputShape = [3, 3, 1];

      const webGLDataTextureArray = createOutputTextureArray(gl,
        originalOutputShape,
        defaultOptions,
        "dummyNodeName");

      expect(webGLDataTextureArray.height).to.equal(3);
      expect(webGLDataTextureArray.width).to.equal(3);
      expect(webGLDataTextureArray.numberOfTextures).to.equal(1);
      expect(webGLDataTextureArray.shape).to.deep.equal([3, 3, 4]);
      expect(webGLDataTextureArray.elementCount).to.equal(36);
      expect(webGLDataTextureArray.originalShape).to.deep.equal([3, 3, 1]);
      expect(webGLDataTextureArray.originalElementCount).to.equal(9);
    }} />

    <Test title="Simple test breakpoint test - 2 render targets" fn={() => {
      const gl = createWebGLContext();
      const originalOutputShape = [100, 100, 3];
      const renderTargetBreakpoints: NNShadersOptions["renderTargetBreakpoints"] = [
        { outputTextureElementCount: 1000, numberOfRenderTargets: 1 },
        { outputTextureElementCount: 10000, numberOfRenderTargets: 2 },
      ]

      // Because the total size is 50000, we should have 2 render targets.
      const webGLDataTextureArray = createOutputTextureArray(gl,
        originalOutputShape,
        { ...defaultOptions, renderTargetBreakpoints },
        "dummyNodeName");

      // Here's why it should be 71:
      // - 3 channels are padded to 4, and so 100 * 100 * 4 = 40000 is the total size of the texture array.
      // - Because of the breakpoint, the textures should be split into 2 - so 20000 each.
      // - Each texture is rgba, so the total size of the texture is 20000 / 4 = 5000.
      // - The square root of 5000 is 71. Some calculations are done to find width x height, and we get 71 x 71.
      expect(webGLDataTextureArray.originalElementCount).to.equal(30000);
      expect(webGLDataTextureArray.height).to.equal(71);
      expect(webGLDataTextureArray.width).to.equal(71);
      expect(webGLDataTextureArray.numberOfTextures).to.equal(2);
      expect(webGLDataTextureArray.shape).to.deep.equal([100, 100, 4]);
      expect(webGLDataTextureArray.elementCount).to.equal(40328);
      expect(webGLDataTextureArray.originalShape).to.deep.equal([100, 100, 3]);
    }} />

    <Test title="Basic Conv2D test" fn={() => {
      const gl = createWebGLContext();
      const numFilters = 5;
      const originalOutputShape = [6, 6, numFilters];
      const renderTargetBreakpoints: NNShadersOptions["renderTargetBreakpoints"] = [
        { outputTextureElementCount: 10, numberOfRenderTargets: 1 },
        { outputTextureElementCount: 100, numberOfRenderTargets: 2 }, // reduce the size value for this breakpoint.
      ]

      const webGLDataTextureArray = createOutputTextureArray(gl,
        originalOutputShape,
        { ...defaultOptions, renderTargetBreakpoints },
        "dummyNodeName",
        updateConv2DOutputDimensions); // XXX: pass in the updateConv2DOutputDimensions callback function.

      // The conv2D output texture has the additional constraint (besides the usual channels padding to 4):
      // - Every single texture height*width*4 in the output texture array must be a multiple of the number of filters.
      expect(webGLDataTextureArray.originalShape).to.deep.equal([6, 6, 5]);
      expect(webGLDataTextureArray.originalElementCount).to.equal(180); // 6 * 6 * 5 = 180
      expect(webGLDataTextureArray.height).to.equal(6);
      expect(webGLDataTextureArray.width).to.equal(6);
      expect(webGLDataTextureArray.numberOfTextures).to.equal(2);
      // The final data shape should be equal to the original shape, but with the last dimension padded to 4.
      expect(webGLDataTextureArray.shape).to.deep.equal([6, 6, 8]);
      // elementCount is actually the final texture's total size: width * height * numberOfTextures * 4 = 6*6*2*4 = 288
      expect(webGLDataTextureArray.elementCount).to.equal(288);
    }} />

    <Test title="Basic Conv2D test - with 3 render targets" fn={() => {
      const gl = createWebGLContext();
      const numFilters = 5;
      const originalOutputShape = [100, 100, numFilters];
      const renderTargetBreakpoints: NNShadersOptions["renderTargetBreakpoints"] = [
        { outputTextureElementCount: 10, numberOfRenderTargets: 1 },
        { outputTextureElementCount: 10000, numberOfRenderTargets: 3 }, // reduce the size value for this breakpoint.
      ]

      const webGLDataTextureArray = createOutputTextureArray(gl,
        originalOutputShape,
        { ...defaultOptions, renderTargetBreakpoints },
        "dummyNodeName",
        updateConv2DOutputDimensions); // XXX: pass in the updateConv2DOutputDimensions callback function.

      // The conv2D output texture has the additional constraint (besides the usual channels padding to 4):
      // - Every single texture height*width*4 in the output texture array must be a multiple of the number of filters.
      expect(webGLDataTextureArray.originalShape).to.deep.equal([100, 100, 5]);
      expect(webGLDataTextureArray.originalElementCount).to.equal(50000); // 100 * 100 * 5 = 50000
      expect(webGLDataTextureArray.height).to.equal(82);
      expect(webGLDataTextureArray.width).to.equal(82);
      expect(webGLDataTextureArray.numberOfTextures).to.equal(3);
      // The final data shape should be equal to the original shape, but with the last dimension padded to 4.
      expect(webGLDataTextureArray.shape).to.deep.equal([100, 100, 8]);
      // elementCount is actually the final texture's total size: width * height * numberOfTextures * 4 = 82*82*3*4 = 80688
      expect(webGLDataTextureArray.elementCount).to.equal(80688);
    }} />


  </TestGroup>
}