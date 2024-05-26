import { NNShadersOptions } from "./types";

export const shapeUniformName = "uShape";
export const squareShhapeUniformName = "uSquareShape";

export const defaultOptions: NNShadersOptions = {
  // XXX: NOTE: these width and height values may not be necessary? As we're drawing to a texture, not the canvas.
  canvasWidth: 256,
  canvasHeight: 256,
  viewportMaxSize: 2048,
  hasBatchDimension: true,
  transformations: {
    padChannels: true
  },
  renderTargetBreakpoints: [
    // If it's over outputTextureElementCount, we'll use numberOfRenderTargets.
    { outputTextureElementCount: 10000, numberOfRenderTargets: 2 },
    { outputTextureElementCount: 100000, numberOfRenderTargets: 4 },
  ]
}