"use client";

import { createWebGLContext } from "@wedge/core/tests/testHelpers";
import { createOutputTextureArray } from "@wedge/core/backends/webgl/buffersAndTextures";
import { defaultOptions } from "@wedge/core/constants";
import { updateConv2DOutputDimensions } from "@wedge/core/backends/webgl/ops/conv2D/output";
import { WedgeOptions } from "@wedge/core/backends/webgl/types";
import { Test, TestContainer, expect } from "react-browser-tests";

export default function TextureSizesTests() {
  return <TestContainer>
    <Test title="Simple check if channels are padded" fn={() => {
      const gl = createWebGLContext();
      const originalOutputShape = [3, 3, 1];

      const webGLDataTextureArray = createOutputTextureArray(gl,
        originalOutputShape,
        defaultOptions,
        "dummyNodeName");

      if (!webGLDataTextureArray) {
        throw new Error("webGLDataTextureArray is null");
      }

      expect(webGLDataTextureArray.RGBATextureShape).to.deep.equal([3, 3, 1, 4]);
      expect(webGLDataTextureArray.RGBATextureElementCount).to.equal(36);
      expect(webGLDataTextureArray.originalShape).to.deep.equal([3, 3, 1]);
      expect(webGLDataTextureArray.originalElementCount).to.equal(9);
    }} />

    <Test title="Simple test breakpoint test - 2 render targets" fn={() => {
      const gl = createWebGLContext();
      const originalOutputShape = [100, 100, 3];
      const renderTargetBreakpoints: WedgeOptions["renderTargetBreakpoints"] = [
        { outputTextureElementCount: 1000, numberOfRenderTargets: 1 },
        { outputTextureElementCount: 10000, numberOfRenderTargets: 2 },
      ]

      // Because the total size is 50000, we should have 2 render targets.
      const webGLDataTextureArray = createOutputTextureArray(gl,
        originalOutputShape,
        { ...defaultOptions, renderTargetBreakpoints },
        "dummyNodeName");

      if (!webGLDataTextureArray) {
        throw new Error("webGLDataTextureArray is null");
      }

      // Here's why it should be 71:
      // - 3 channels are padded to 4, and so 100 * 100 * 4 = 40000 is the total size of the texture array.
      // - Because of the breakpoint, the textures should be split into 2 - so 20000 each.
      // - Each texture is rgba, so the total size of the texture is 20000 / 4 = 5000.
      // - The square root of 5000 is 71. Some calculations are done to find width x height, and we get 71 x 71.
      expect(webGLDataTextureArray.originalElementCount).to.equal(30000);
      expect(webGLDataTextureArray.RGBATextureShape).to.deep.equal([71, 71, 2, 4]);
      expect(webGLDataTextureArray.RGBATextureElementCount).to.equal(40328);
      expect(webGLDataTextureArray.originalShape).to.deep.equal([100, 100, 3]);
    }} />

    <Test title="Basic Conv2D test" fn={() => {
      const gl = createWebGLContext();
      const numFilters = 5;
      const originalOutputShape = [6, 6, numFilters];
      const renderTargetBreakpoints: WedgeOptions["renderTargetBreakpoints"] = [
        { outputTextureElementCount: 10, numberOfRenderTargets: 1 },
        { outputTextureElementCount: 100, numberOfRenderTargets: 2 }, // reduce the size value for this breakpoint.
      ]

      const webGLDataTextureArray = createOutputTextureArray(gl,
        originalOutputShape,
        { ...defaultOptions, renderTargetBreakpoints },
        "dummyNodeName",
        updateConv2DOutputDimensions); // XXX: pass in the updateConv2DOutputDimensions callback function.

      if (!webGLDataTextureArray) {
        throw new Error("webGLDataTextureArray is null");
      }

      // The conv2D output texture has the additional constraint (besides the usual channels padding to 4):
      // - Every single texture height*width*4 in the output texture array must be a multiple of the number of filters.
      expect(webGLDataTextureArray.originalShape).to.deep.equal([6, 6, 5]);
      expect(webGLDataTextureArray.originalElementCount).to.equal(180); // 6 * 6 * 5 = 180
      expect(webGLDataTextureArray.RGBATextureShape).to.deep.equal([6, 6, 2, 4]);
      // elementCount is actually the final texture's total size: width * height * numberOfTextures * 4 = 6*6*2*4 = 288
      expect(webGLDataTextureArray.RGBATextureElementCount).to.equal(288);
    }} />

    <Test title="Basic Conv2D test - with 3 render targets" fn={() => {
      const gl = createWebGLContext();
      const numFilters = 5;
      const originalOutputShape = [100, 100, numFilters];
      const renderTargetBreakpoints: WedgeOptions["renderTargetBreakpoints"] = [
        { outputTextureElementCount: 10, numberOfRenderTargets: 1 },
        { outputTextureElementCount: 10000, numberOfRenderTargets: 3 }, // reduce the size value for this breakpoint.
      ]

      const webGLDataTextureArray = createOutputTextureArray(gl,
        originalOutputShape,
        { ...defaultOptions, renderTargetBreakpoints },
        "dummyNodeName",
        updateConv2DOutputDimensions); // XXX: pass in the updateConv2DOutputDimensions callback function.

      if (!webGLDataTextureArray) {
        throw new Error("webGLDataTextureArray is null");
      }

      // The conv2D output texture has the additional constraint (besides the usual channels padding to 4):
      // - Every single texture height*width*4 in the output texture array must be a multiple of the number of filters.
      expect(webGLDataTextureArray.originalShape).to.deep.equal([100, 100, 5]);
      expect(webGLDataTextureArray.originalElementCount).to.equal(50000); // 100 * 100 * 5 = 50000
      expect(webGLDataTextureArray.RGBATextureShape).to.deep.equal([82, 82, 3, 4]);
      // elementCount is actually the final texture's total size: width * height * numberOfTextures * 4 = 82*82*3*4 = 80688
      expect(webGLDataTextureArray.RGBATextureElementCount).to.equal(80688);
    }} />
  </TestContainer>
}