import puppeteer from 'puppeteer';
import { NNShaders } from "../NNShaders";
import { checkFloatTextureSupport } from "../helpers";

describe('WebGL Tests', () => {
  it('checks WebGL float texture support', async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    const result = await page.evaluate(() => {
      const nns = new NNShaders();
      const supportsFloatTextures = checkFloatTextureSupport(nns.gl as WebGL2RenderingContext);

      if (!supportsFloatTextures) {
        return 'WebGL float textures are not supported';
      }

      return 'WebGL float textures are supported';
    });

    expect(result).toBe('WebGL float textures are supported');

    await browser.close();
  });
});