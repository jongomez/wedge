import type { CSSProperties } from "react";

// Operation categories
export type OperationCategory =
  | "input"
  | "convolution"
  | "arithmetic"
  | "activation"
  | "padding"
  | "transform"
  | "output";

// Handle types for connections
export type HandleType = "tensor" | "weight";

// Operation configuration
export interface OperationConfig {
  name: string;
  displayName: string;
  category: OperationCategory;
  inputs: { id: string; type: HandleType; label: string }[];
  outputs: { id: string; type: HandleType; label: string }[];
  defaultParams: Record<string, unknown>;
  icon: string;
}

// All supported operations
export const OPERATIONS: Record<string, OperationConfig> = {
  Placeholder: {
    name: "Placeholder",
    displayName: "Input",
    category: "input",
    inputs: [],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: { shape: [1, 224, 224, 3] },
    icon: "IN",
  },
  Conv2D: {
    name: "Conv2D",
    displayName: "Conv2D",
    category: "convolution",
    inputs: [
      { id: "input", type: "tensor", label: "In" },
      { id: "kernel", type: "weight", label: "K" },
    ],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: { strides: [1, 1], padding: "same", filters: 32 },
    icon: "C2",
  },
  DepthwiseConv2D: {
    name: "DepthwiseConv2D",
    displayName: "DWConv2D",
    category: "convolution",
    inputs: [
      { id: "input", type: "tensor", label: "In" },
      { id: "kernel", type: "weight", label: "K" },
    ],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: { strides: [1, 1], padding: "same" },
    icon: "DW",
  },
  AddV2: {
    name: "AddV2",
    displayName: "Add",
    category: "arithmetic",
    inputs: [
      { id: "a", type: "tensor", label: "A" },
      { id: "b", type: "tensor", label: "B" },
    ],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: {},
    icon: "+",
  },
  Mul: {
    name: "Mul",
    displayName: "Multiply",
    category: "arithmetic",
    inputs: [
      { id: "a", type: "tensor", label: "A" },
      { id: "b", type: "tensor", label: "B" },
    ],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: {},
    icon: "x",
  },
  Relu: {
    name: "Relu",
    displayName: "ReLU",
    category: "activation",
    inputs: [{ id: "input", type: "tensor", label: "In" }],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: {},
    icon: "R",
  },
  Relu6: {
    name: "Relu6",
    displayName: "ReLU6",
    category: "activation",
    inputs: [{ id: "input", type: "tensor", label: "In" }],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: {},
    icon: "R6",
  },
  Pad: {
    name: "Pad",
    displayName: "Pad",
    category: "padding",
    inputs: [{ id: "input", type: "tensor", label: "In" }],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: { paddings: [[0, 0], [1, 1], [1, 1], [0, 0]] },
    icon: "P",
  },
  PadV2: {
    name: "PadV2",
    displayName: "PadV2",
    category: "padding",
    inputs: [{ id: "input", type: "tensor", label: "In" }],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: { paddings: [[0, 0], [1, 1], [1, 1], [0, 0]], constantValue: 0 },
    icon: "P2",
  },
  MirrorPad: {
    name: "MirrorPad",
    displayName: "MirrorPad",
    category: "padding",
    inputs: [{ id: "input", type: "tensor", label: "In" }],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: { paddings: [[0, 0], [1, 1], [1, 1], [0, 0]], mode: "reflect" },
    icon: "MP",
  },
  ResizeBilinear: {
    name: "ResizeBilinear",
    displayName: "Resize",
    category: "transform",
    inputs: [{ id: "input", type: "tensor", label: "In" }],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: { size: [224, 224], alignCorners: false },
    icon: "RS",
  },
  Reshape: {
    name: "Reshape",
    displayName: "Reshape",
    category: "transform",
    inputs: [{ id: "input", type: "tensor", label: "In" }],
    outputs: [{ id: "output", type: "tensor", label: "Out" }],
    defaultParams: { newShape: [-1] },
    icon: "SH",
  },
  Output: {
    name: "Output",
    displayName: "Output",
    category: "output",
    inputs: [{ id: "input", type: "tensor", label: "In" }],
    outputs: [],
    defaultParams: {},
    icon: "OUT",
  },
};

// Operations grouped by category
export const OPERATION_CATEGORIES: { name: string; category: OperationCategory; ops: string[] }[] = [
  { name: "Input", category: "input", ops: ["Placeholder"] },
  { name: "Convolution", category: "convolution", ops: ["Conv2D", "DepthwiseConv2D"] },
  { name: "Arithmetic", category: "arithmetic", ops: ["AddV2", "Mul"] },
  { name: "Activation", category: "activation", ops: ["Relu", "Relu6"] },
  { name: "Padding", category: "padding", ops: ["Pad", "PadV2", "MirrorPad"] },
  { name: "Transform", category: "transform", ops: ["ResizeBilinear", "Reshape"] },
  { name: "Output", category: "output", ops: ["Output"] },
];

// Color palette (Bauhaus-inspired)
export const COLORS = {
  bg: "#0a0a0a",
  bgSecondary: "#111",
  bgTertiary: "rgba(255,255,255,0.03)",
  accent: "#6366f1",
  accentLight: "#818cf8",
  text: "#fff",
  textSecondary: "#9ca3af",
  textMuted: "#6b7280",
  border: "rgba(255,255,255,0.1)",
  borderHover: "rgba(99,102,241,0.5)",
  borderSelected: "#6366f1",
  valid: "#22c55e",
  invalid: "#ef4444",
  // Category colors
  catInput: "#22c55e",
  catConvolution: "#6366f1",
  catArithmetic: "#8b5cf6",
  catActivation: "#ec4899",
  catPadding: "#14b8a6",
  catTransform: "#f97316",
  catOutput: "#f59e0b",
};

export const CATEGORY_COLORS: Record<OperationCategory, string> = {
  input: COLORS.catInput,
  convolution: COLORS.catConvolution,
  arithmetic: COLORS.catArithmetic,
  activation: COLORS.catActivation,
  padding: COLORS.catPadding,
  transform: COLORS.catTransform,
  output: COLORS.catOutput,
};

// Dimensions
export const DIMS = {
  nodeWidth: 160,
  handleSize: 10,
  gridSize: 20,
  borderRadius: 2,
};

// Shared styles
export const styles: Record<string, CSSProperties> = {
  handle: {
    width: DIMS.handleSize,
    height: DIMS.handleSize,
    background: COLORS.accent,
    border: `2px solid ${COLORS.bg}`,
    borderRadius: DIMS.borderRadius,
  },
  handleLeft: {
    left: -DIMS.handleSize / 2 - 1,
  },
  handleRight: {
    right: -DIMS.handleSize / 2 - 1,
  },
};
