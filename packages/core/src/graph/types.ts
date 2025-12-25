import { WebGLOpNode } from "../backends/webgl/types";
import { Ops } from "../ops/types";
import { TensorBase } from "../tensor/TensorBase";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TFTensor = any;

/*
export interface GraphType {
  nodes: { [key: string]: GraphNode };
  placeholders: DataNode[];
  inputs: Tensor[];
  outputs: Tensor[];
  weights: Tensor[];
  // signature?: ISignatureDef;
  // functions?: { [key: string]: GraphType };
  // initNodes?: DataNode[];
}
*/

export type ParamType = 'number' | 'string' | 'string[]' | 'number[]' | 'bool' | 'bool[]' | 'shape' | 'shape[]' | 'tensor' | 'tensors' | 'dtype' | 'dtype[]' | 'func';

export type ValueType = string | string[] | number | number[] | number[][] | boolean | boolean[] | TFTensor | TFTensor[];

export declare interface ParamValue {
  value?: ValueType;
  type: ParamType;
}

export declare interface InputParamValue extends ParamValue {
  inputIndexStart?: number;
  inputIndexEnd?: number;
}

interface GraphNodeParam {
  source: 'tensor' | 'static';
  type: ParamType;
  value: GraphNode | ValueType;  // For static values
  tensorIndex?: number;  // For tensor inputs when source is 'tensor'
}

export type GraphNode = {
  name: string;
  op: {
    name: keyof Ops<TensorBase, TensorBase>;
    category?: string;  // From TF mapper
  };

  // Input handling
  inputs: GraphNode[];  // Direct references to input nodes
  inputNames: string[];  // TF-style input references (for serialization)
  params: { [key: string]: GraphNodeParam };  // Unified params (both input and attr)

  // Graph structure
  children: GraphNode[];

  // Optional TF-specific fields
  signatureKey?: string;  // For graph inputs/outputs
  defaultOutput?: number;  // For multi-output ops
  outputs?: string[];     // Possible named outputs
}

export interface Graph {
  nodes: { [key: string]: GraphNode };
  placeholders: GraphNode[];
  inputs: GraphNode[];
  outputs: GraphNode[];
  weights: GraphNode[];
  optimizations: {}
}

export type OpNode = WebGLOpNode;
