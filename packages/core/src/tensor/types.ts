
export type RegularArray<T> =
  T[] | T[][] | T[][][] | T[][][][] | T[][][][][] | T[][][][][][];

export type DataArray = TypedArray | RegularArray<number>;

export interface DataTypeMap {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
  complex64: Float32Array;
  string: string[];
}

export type DataType = keyof DataTypeMap;
export type NumericDataType = 'float32' | 'int32' | 'bool' | 'complex64';

export type TypedArray = Float32Array | Int32Array | Uint8Array;
/** Tensor data used in tensor creation and user-facing API. */
export type DataValues = DataTypeMap[DataType];

/** Data type for tensor initialization */
export type TensorData = DataArray | TypedArray;