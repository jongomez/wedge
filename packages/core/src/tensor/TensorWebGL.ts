import { maxTextureDim } from '../constants';
import { TensorBase } from './TensorBase';
import { DataType, TensorData } from './types';

// Helper function to flatten nested arrays
function flattenArray(arr: any[]): number[] {
  return arr.reduce((flat, item) => {
    return flat.concat(Array.isArray(item) ? flattenArray(item) : item);
  }, []);
}

export interface TensorWebGLConfig {
  gl: WebGL2RenderingContext;
  shape?: number[];
  dtype?: DataType;
  nodeName?: string;
}

export class TensorWebGL extends TensorBase {
  public readonly texture: WebGLTexture;
  public readonly RGBATextureShape: [number, number, number, number];
  public readonly RGBATextureElementCount: number;
  public readonly webGLType: "float" | "vec2" | "vec3" | "vec4" | "sampler2D" | "sampler2DArray" = "sampler2DArray";

  private gl: WebGL2RenderingContext;

  constructor(data: TensorData, gl: WebGL2RenderingContext, options?: { shape?: number[], dtype?: DataType, nodeName?: string }) {
    super(data, options);
    this.gl = gl;

    // Initialize WebGL texture
    const [texture, textureShape] = this.initWebGLTensor(data);
    this.texture = texture;
    this.RGBATextureShape = textureShape;
    this.RGBATextureElementCount = textureShape[0] * textureShape[1] * textureShape[2] * textureShape[3];
  }

  protected initializeData(data: TensorData): void {
    // Data initialization is handled in initWebGLTensor
    // This method is required by TensorBase but the actual initialization
    // happens when creating the WebGL texture
  }

  protected inferShape(data: TensorData): number[] {
    if (!data || (Array.isArray(data) && data.length === 0)) {
      throw new Error('Cannot infer shape from empty data');
    }

    if (Array.isArray(data)) {
      // For array data, we assume it's a flat array and return a 1D shape
      return [data.length];
    } else {
      // For TypedArrays, we also assume 1D shape
      return [data.byteLength / (data as Float32Array).BYTES_PER_ELEMENT];
    }
  }

  toTypedArray(): Float32Array {
    // Read from the texture
    const pixels = new Float32Array(this.RGBATextureElementCount);
    const framebuffer = this.gl.createFramebuffer();
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.framebufferTexture2D(
      this.gl.FRAMEBUFFER,
      this.gl.COLOR_ATTACHMENT0,
      this.gl.TEXTURE_2D,
      this.texture,
      0
    );

    this.gl.readPixels(
      0, 0,
      this.RGBATextureShape[0],
      this.RGBATextureShape[1],
      this.gl.RGBA,
      this.gl.FLOAT,
      pixels
    );

    this.gl.deleteFramebuffer(framebuffer);
    return pixels.slice(0, this.size);
  }

  private initWebGLTensor(
    data: TensorData
  ): [WebGLTexture, [number, number, number, number]] {
    // Convert shape to texture dimensions
    const [width, height, originalElementCount, textureElementCount] = this.convertShapeToTexture2DShape(this.shape);
    const numberOfTextures = 1; // For now using single texture

    // Create padded data array if needed
    const paddedData = new Float32Array(width * height * numberOfTextures * 4);
    if (data) {
      if (Array.isArray(data)) {
        // Convert nested arrays to flat Float32Array
        const flatData = flattenArray(data);
        paddedData.set(new Float32Array(flatData), 0);
      } else {
        paddedData.set(new Float32Array(data.buffer), 0);
      }
    }

    let texture = this.gl.createTexture();
    if (!texture) {
      throw new Error("Error creating WebGL texture array");
    }

    this.gl.bindTexture(this.gl.TEXTURE_2D_ARRAY, texture);

    this.gl.texParameteri(this.gl.TEXTURE_2D_ARRAY, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D_ARRAY, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D_ARRAY, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D_ARRAY, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);

    // Initialize texture with data for all layers
    this.gl.texImage3D(
      this.gl.TEXTURE_2D_ARRAY,
      0, // mip level
      this.gl.RGBA32F, // internal format
      width,
      height,
      numberOfTextures,
      0, // border
      this.gl.RGBA,
      this.gl.FLOAT,
      paddedData
    );

    return [texture, [width, height, numberOfTextures, 4]];
  }

  private convertShapeToTexture2DShape(shape: number[], divideBy = 1): [number, number, number, number] {
    if (shape.length === 0) {
      throw new Error('Shape must have at least one dimension');
    }

    // Calculate total size of the shape
    const originalElementCount = this.size;

    // Adjust the total size for a texture to account for 4 elements per position (RGBA)
    if (originalElementCount % 4 !== 0) {
      throw new Error('originalElementCount must be divisible by 4');
    }

    // Adjust the total size for a texture to account for 4 elements per position (RGBA)
    const totalTexturePositions = originalElementCount / (4 * divideBy);

    // Initialize height
    let height = Math.ceil(Math.sqrt(totalTexturePositions));

    // Calculate the width based on the height
    let width = Math.floor(totalTexturePositions / height);

    // Check if the calculated area is enough, if not, adjust the width
    if (width * height < totalTexturePositions) {
      width++;
    }

    // Check if the resulting dimensions are excessively large
    if (width >= maxTextureDim || height >= maxTextureDim) {
      throw new Error(`Shape size is too large. Width: ${width}, Height: ${height}`);
    }

    if (width * height < totalTexturePositions) {
      throw new Error(`Error: width * height < totalTexturePositions. width: ${width}, height: ${height}, totalTexturePositions: ${totalTexturePositions}`);
    }

    // Calculate the textureElementCount
    const textureElementCount = width * height * divideBy * 4; // Now considering the 4 elements per position

    return [width, height, originalElementCount, textureElementCount];
  }

  toString(): string {
    return `TensorWebGL(shape=[${this.shape.join(', ')}], dtype=${this.dtype})`;
  }

  dispose(): void {
    if (this.texture) {
      this.gl.deleteTexture(this.texture);
    }
  }
} 