import { DataArray } from "../tensor/types";
import { TensorType } from "../types";

export type Conv2DOpParams = {
  strides: number[];
  pad: "same" | "valid";
  kernelX: number;
  kernelY: number;
  kernelDepth: number;
  numFilters: number;
  activation: 'relu' | 'linear' | null;
  hasBias: boolean;
}

export type ResizeBilinearOpParams = {
  alignCorners: boolean;
  halfPixelCenters: boolean;
  dtype: "float32" | "int32";
  size: number;
}

export type OpParams = Conv2DOpParams | ResizeBilinearOpParams;

export type DepthwiseConv2DOpParams = Conv2DOpParams;

export type ArithmeticOpParams = {
  operation: "add" | "multiply";
}

export type OpInput = TensorType | DataArray;


export type Ops<TInput, TOutput> = {
  conv2d: (input: TInput, filters: TInput, params: Conv2DOpParams) => TOutput;
  depthwiseConv2d: (input: TInput, filters: TInput, params: DepthwiseConv2DOpParams) => TOutput;
  relu: (input: TInput) => TOutput;
  resizeBilinear: (input: TInput, params: ResizeBilinearOpParams) => TOutput;

  // Arithmetic operations
  add: (input1: TInput, input2: TInput) => TOutput;
  multiply: (input1: TInput, input2: TInput) => TOutput;
}