# Wedge

Wedge is a web based machine learning framework.

## What This Project Currently Does

The main focus at the moment is compiling and executing TensorFlow.js models
using custom GLSL shaders. Currently, this project:

1. **Loads** TensorFlow.js GraphModel or LayersModel formats
2. **Compiles** them into an internal graph representation
3. **Executes** operations via custom WebGL2 fragment shaders
4. **Uses textures** for efficient GPU tensor storage and computation

## Goals

The main goals are:

1. Be able to load other types of models - onnx, tflite, gguf and possibly more.
2. Have a custom top level API for building / editing / debugging / machine
   learning graphs.
3. Support for different backends. While the current focus is WebGL2, an
   Electron app that supports other backends (e.g. a C or C++ backend) is also
   planned.

## Project Structure

```
wedge/
├── packages/
│   ├── core/                 # @wedge/core - Main inference engine
│   │   └── src/
│   │       ├── backends/
│   │       │   └── webgl/
│   │       │       ├── WedgeWebGL.ts       # Main runtime class
│   │       │       ├── ops/                # WebGL shader implementations
│   │       │       │   ├── conv2D/         # 2D convolution
│   │       │       │   ├── depthwiseConv2D/
│   │       │       │   ├── relu/           # Activation
│   │       │       │   ├── arithmetic/     # Add, multiply
│   │       │       │   ├── ResizeBilinear/
│   │       │       │   └── singleInputBasic/
│   │       │       ├── buffersAndTextures.ts
│   │       │       ├── setupShadersAndWebGL.ts
│   │       │       └── types.ts
│   │       ├── graph/
│   │       │   ├── types.ts      # GraphNode, Graph interfaces
│   │       │   ├── compile.ts    # Graph compilation
│   │       │   └── transform.ts  # Graph optimizations
│   │       ├── tensor/
│   │       │   ├── TensorBase.ts
│   │       │   └── TensorWebGL.ts
│   │       ├── ops/
│   │       │   └── types.ts      # Op parameter types
│   │       ├── external/
│   │       │   └── tensorflow/   # Vendored TF.js code for model loading
│   │       └── types.ts          # Core type definitions
│   └── ui-react/             # @wedge/ui-react - React components
├── apps/
│   ├── site/                 # Next.js demo/test application
│   └── electron/             # Desktop app (WIP)
└── benchmarks/               # Performance comparison app
```

## Key Concepts

### WedgeWebGL Class (`packages/core/src/backends/webgl/WedgeWebGL.ts`)

The main runtime. Key properties:

- `gl`: WebGL2RenderingContext
- `originalGraph` / `compiledGraph`: Graph representations
- `orderedNodes`: Execution order of operations
- `opNodeWithProgramMap`: Maps operations to compiled WebGL programs

Key methods:

- `loadGraphModel(path)`: Load a TensorFlow.js model
- `run(inputRawData)`: Execute inference, returns Float32Array
- `readOutput()`: Read results from GPU textures

### Graph System (`packages/core/src/graph/`)

```typescript
// GraphNode represents a single operation in the computation graph
type GraphNode = {
  name: string;
  op: { name: keyof Ops }; // Operation type (conv2d, relu, etc.)
  inputs: GraphNode[]; // Input nodes
  params: { [key: string]: GraphNodeParam };
  children: GraphNode[];
};

// Graph is the full computation graph
interface Graph {
  nodes: { [key: string]: GraphNode };
  placeholders: GraphNode[]; // Model inputs
  inputs: GraphNode[];
  outputs: GraphNode[];
  weights: GraphNode[];
}
```

### Operations (`packages/core/src/ops/types.ts`)

Supported operations:

- `conv2d` - 2D convolution with optional ReLU fusion
- `depthwiseConv2d` - Depthwise separable convolution
- `relu` - ReLU activation
- `resizeBilinear` - Image resizing
- `add`, `multiply` - Element-wise arithmetic

Each operation has:

- `init.ts` - Setup and initialization
- `output.ts` - Output texture configuration
- `webGLShader.ts` - GLSL fragment shader code

### Tensor System (`packages/core/src/tensor/`)

- `TensorBase`: Abstract base class
- `TensorWebGL`: WebGL-backed tensor stored as GPU textures

Data formats:

- `NHWC`: Batch, Height, Width, Channels
- `HWC`: Height, Width, Channels
- `VEC`: Flat vector

## How to Run

This is an npm workspaces monorepo.

### Install Dependencies

```bash
npm install
```

### Run Development Servers

```bash
# Main site app
npm run -w apps/site dev

# Core package dev server (includes tests)
npm run -w packages/core dev

# Benchmarks
npm run -w benchmarks dev
```

### Build

```bash
npm run -w packages/core build
npm run -w apps/site build
```

### Lint

```bash
npm run -w packages/core lint
```

## Architecture Flow

1. **Model Loading**: TensorFlow.js model loaded via `loadGraphModel()`
2. **Graph Construction**: Model converted to internal `Graph` representation
3. **Compilation**: Graph nodes ordered for execution, WebGL programs created
4. **Execution**: For each operation:
   - Bind input textures
   - Set shader uniforms
   - Render to output texture (framebuffer)
5. **Output**: Final texture read back to CPU as Float32Array

## Key Files for Understanding the Codebase

| File                                                         | Purpose                            |
| ------------------------------------------------------------ | ---------------------------------- |
| `packages/core/src/backends/webgl/WedgeWebGL.ts`             | Main runtime, execution loop       |
| `packages/core/src/graph/types.ts`                           | Graph and node type definitions    |
| `packages/core/src/ops/types.ts`                             | Operation interfaces and params    |
| `packages/core/src/backends/webgl/ops/conv2D/webGLShader.ts` | Example shader implementation      |
| `packages/core/src/backends/webgl/setupShadersAndWebGL.ts`   | WebGL context initialization       |
| `packages/core/src/backends/webgl/buffersAndTextures.ts`     | Framebuffer and texture management |

## Technology Stack

- **Runtime**: TypeScript, WebGL2
- **Framework**: React 19, Next.js 15
- **ML**: TensorFlow.js 4.16 (for model loading/conversion)
- **Testing**: react-browser-tests, Puppeteer

## Status

Active development. WebGL backend is functional. WebGPU backend is planned but
not yet implemented.
