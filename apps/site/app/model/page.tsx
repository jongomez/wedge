"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

// Supported operations in Wedge WebGL backend
const SUPPORTED_OPS = new Set([
  // Convolutions
  "Conv2D",
  "_FusedConv2D",
  "FusedConv2D",
  "DepthwiseConv2D",
  "DepthwiseConv2dNative",
  "FusedDepthwiseConv2dNative",
  // Arithmetic
  "AddV2",
  "Add",
  "Mul",
  // Activations
  "Relu",
  "ReLU",
  "Relu6",
  "Sigmoid",
  // Padding
  "Pad",
  "PadV2",
  "MirrorPad",
  // Pooling
  "MaxPool",
  // Other
  "ResizeBilinear",
  "Reshape",
  "Identity",
  // Data nodes (always supported)
  "Placeholder",
  "Const",
]);

// Shader code snippets for supported operations
const SHADER_SNIPPETS: Record<string, string> = {
  Conv2D: `// Conv2D Fragment Shader
for (int ky = 0; ky < kernelY; ky++) {
  for (int kx = 0; kx < kernelX; kx++) {
    vec4 inputVal = texelFetch(input, inputXYZ, 0);
    vec4 weightVal = texelFetch(weights, weightXYZ, 0);
    result += inputVal * weightVal;
  }
}`,
  DepthwiseConv2D: `// DepthwiseConv2D Fragment Shader
for (int ky = 0; ky < kernelY; ky++) {
  for (int kx = 0; kx < kernelX; kx++) {
    vec4 inputVal = texelFetch(input, inputXYZ, 0);
    vec4 weightVal = texelFetch(weights, weightXYZ, 0);
    result += inputVal * weightVal; // Per-channel
  }
}`,
  Relu: `// ReLU Fragment Shader
vec4 inputVal = texelFetch(input, texelXYZ, 0);
result = max(inputVal, vec4(0.0));`,
  Relu6: `// ReLU6 Fragment Shader
vec4 inputVal = texelFetch(input, texelXYZ, 0);
result = clamp(inputVal, vec4(0.0), vec4(6.0));`,
  Sigmoid: `// Sigmoid Fragment Shader
vec4 inputVal = texelFetch(input, texelXYZ, 0);
result = vec4(1.0) / (vec4(1.0) + exp(-inputVal));`,
  AddV2: `// Add Fragment Shader
vec4 a = texelFetch(inputA, texelXYZ, 0);
vec4 b = texelFetch(inputB, texelXYZ, 0);
result = a + b;`,
  Mul: `// Multiply Fragment Shader
vec4 a = texelFetch(inputA, texelXYZ, 0);
vec4 b = texelFetch(inputB, texelXYZ, 0);
result = a * b;`,
  ResizeBilinear: `// ResizeBilinear Fragment Shader
vec2 srcCoord = (vec2(outXY) + 0.5) * scale - 0.5;
vec4 tl = texelFetch(input, ivec3(floor(srcCoord), z), 0);
vec4 tr = texelFetch(input, ivec3(ceil(srcCoord.x), floor(srcCoord.y), z), 0);
vec4 bl = texelFetch(input, ivec3(floor(srcCoord.x), ceil(srcCoord.y), z), 0);
vec4 br = texelFetch(input, ivec3(ceil(srcCoord), z), 0);
result = mix(mix(tl, tr, fract(srcCoord.x)), mix(bl, br, fract(srcCoord.x)), fract(srcCoord.y));`,
  Pad: `// Pad Fragment Shader
ivec3 inputPos = outputPos - padBefore;
bool inBounds = all(greaterThanEqual(inputPos, ivec3(0))) &&
                all(lessThan(inputPos, inputDims));
if (inBounds) {
  result = texelFetch(input, inputTextureXYZ, 0);
} else {
  result = vec4(constantValue);
}`,
  MaxPool: `// MaxPool Fragment Shader
float maxVal = -1.0e38;
for (int ph = 0; ph < poolSize.x; ph++) {
  for (int pw = 0; pw < poolSize.y; pw++) {
    int inputH = outH * strides.x + ph;
    int inputW = outW * strides.y + pw;
    if (inputH >= 0 && inputH < inputDims.x &&
        inputW >= 0 && inputW < inputDims.y) {
      float val = texelFetch(input, inputXYZ, 0).r;
      maxVal = max(maxVal, val);
    }
  }
}
result = vec4(maxVal);`,
};

type ModelNode = {
  name: string;
  op: string;
  inputs: string[];
  shape?: number[];
  attrs?: Record<string, unknown>;
};

type LoadedModel = {
  nodes: ModelNode[];
  inputs: string[];
  outputs: string[];
};

const AVAILABLE_MODELS = [
  { name: "BlazeFace Lite", path: "/models/blaze_lite/model.json" },
];

export default function ModelVisualizationPage() {
  const [model, setModel] = useState<LoadedModel | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<ModelNode | null>(null);
  const [modelUrl, setModelUrl] = useState(AVAILABLE_MODELS[0].path);

  const loadModel = async (url: string) => {
    setLoading(true);
    setError(null);
    setSelectedNode(null);

    try {
      const graphModel = await tf.loadGraphModel(url);

      // Extract nodes from the model
      const modelArtifacts = (graphModel as any).artifacts;
      const modelTopology = modelArtifacts?.modelTopology;

      if (!modelTopology?.node) {
        throw new Error("Could not read model topology");
      }

      const nodes: ModelNode[] = modelTopology.node.map((node: any) => ({
        name: node.name || "unnamed",
        op: node.op || "Unknown",
        inputs: node.input || [],
        attrs: node.attr,
      }));

      // Find inputs and outputs
      const inputNodes = nodes.filter(n => n.op === "Placeholder").map(n => n.name);
      const allInputRefs = new Set(nodes.flatMap(n => n.inputs.map(i => i.split(":")[0])));
      const outputNodes = nodes
        .filter(n => !allInputRefs.has(n.name) && n.op !== "Const")
        .map(n => n.name);

      setModel({
        nodes,
        inputs: inputNodes,
        outputs: outputNodes.length > 0 ? outputNodes : [nodes[nodes.length - 1]?.name],
      });
    } catch (err) {
      console.error("Model load error:", err);
      setError(err instanceof Error ? err.message : "Failed to load model");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModel(modelUrl);
  }, []);

  const isSupported = (op: string) => SUPPORTED_OPS.has(op);
  const isDataNode = (op: string) => op === "Placeholder" || op === "Const";

  const getNodeStatus = (op: string) => {
    if (isDataNode(op)) return "data";
    if (isSupported(op)) return "supported";
    return "unsupported";
  };

  const stats = model
    ? {
        total: model.nodes.length,
        ops: model.nodes.filter(n => !isDataNode(n.op)).length,
        supported: model.nodes.filter(n => isSupported(n.op) && !isDataNode(n.op)).length,
        unsupported: model.nodes.filter(n => !isSupported(n.op)).length,
        data: model.nodes.filter(n => isDataNode(n.op)).length,
      }
    : null;

  return (
    <main style={styles.main}>
      <div style={styles.header}>
        <Link href="/" style={styles.backLink}>← Back</Link>
        <h1 style={styles.title}>Model Visualization</h1>
        <p style={styles.subtitle}>
          Load a TensorFlow.js model to see its layers and WebGL support status
        </p>
      </div>

      <div style={styles.controls}>
        <select
          style={styles.select}
          value={modelUrl}
          onChange={(e) => setModelUrl(e.target.value)}
        >
          {AVAILABLE_MODELS.map((m) => (
            <option key={m.path} value={m.path}>{m.name}</option>
          ))}
        </select>
        <button
          style={styles.loadButton}
          onClick={() => loadModel(modelUrl)}
          disabled={loading}
        >
          {loading ? "Loading..." : "Load Model"}
        </button>
      </div>

      {error && (
        <div style={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {stats && (
        <div style={styles.stats}>
          <div style={styles.statCard}>
            <span style={styles.statValue}>{stats.ops}</span>
            <span style={styles.statLabel}>Operations</span>
          </div>
          <div style={{ ...styles.statCard, borderColor: "#22c55e" }}>
            <span style={{ ...styles.statValue, color: "#22c55e" }}>{stats.supported}</span>
            <span style={styles.statLabel}>Supported</span>
          </div>
          <div style={{ ...styles.statCard, borderColor: "#ef4444" }}>
            <span style={{ ...styles.statValue, color: "#ef4444" }}>{stats.unsupported}</span>
            <span style={styles.statLabel}>Unsupported</span>
          </div>
          <div style={{ ...styles.statCard, borderColor: "#6b7280" }}>
            <span style={styles.statValue}>{stats.data}</span>
            <span style={styles.statLabel}>Data Nodes</span>
          </div>
        </div>
      )}

      <div style={styles.legend}>
        <span style={styles.legendItem}>
          <span style={{ ...styles.legendDot, background: "#22c55e" }} /> Supported
        </span>
        <span style={styles.legendItem}>
          <span style={{ ...styles.legendDot, background: "#ef4444" }} /> Unsupported
        </span>
        <span style={styles.legendItem}>
          <span style={{ ...styles.legendDot, background: "#6b7280" }} /> Data
        </span>
      </div>

      <div style={styles.container}>
        <div style={styles.nodeList}>
          {loading && <div style={styles.loading}>Loading model...</div>}

          {model?.nodes.map((node, idx) => {
            const status = getNodeStatus(node.op);
            const isSelected = selectedNode?.name === node.name;

            return (
              <div
                key={node.name + idx}
                style={{
                  ...styles.nodeCard,
                  ...(status === "supported" ? styles.nodeSupported : {}),
                  ...(status === "unsupported" ? styles.nodeUnsupported : {}),
                  ...(status === "data" ? styles.nodeData : {}),
                  ...(isSelected ? styles.nodeSelected : {}),
                }}
                onClick={() => setSelectedNode(isSelected ? null : node)}
              >
                <div style={styles.nodeHeader}>
                  <span style={styles.nodeOp}>{node.op}</span>
                  <span style={{
                    ...styles.badge,
                    background: status === "supported" ? "#22c55e" : status === "unsupported" ? "#ef4444" : "#6b7280"
                  }}>
                    {status}
                  </span>
                </div>
                <div style={styles.nodeName}>{node.name}</div>
                {node.inputs.length > 0 && (
                  <div style={styles.nodeInputs}>
                    ← {node.inputs.length} input{node.inputs.length > 1 ? "s" : ""}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {selectedNode && (
          <div style={styles.sidebar}>
            <div style={styles.sidebarHeader}>
              <h2 style={styles.sidebarTitle}>{selectedNode.op}</h2>
              <button style={styles.closeButton} onClick={() => setSelectedNode(null)}>×</button>
            </div>

            <div style={styles.sidebarSection}>
              <h3 style={styles.sectionTitle}>Node Name</h3>
              <code style={styles.code}>{selectedNode.name}</code>
            </div>

            <div style={styles.sidebarSection}>
              <h3 style={styles.sectionTitle}>Status</h3>
              <span style={{
                ...styles.statusBadge,
                background: isSupported(selectedNode.op) ? "#22c55e20" : "#ef444420",
                color: isSupported(selectedNode.op) ? "#22c55e" : "#ef4444",
              }}>
                {isSupported(selectedNode.op) ? "✓ Supported in Wedge" : "✗ Not Yet Supported"}
              </span>
            </div>

            {selectedNode.inputs.length > 0 && (
              <div style={styles.sidebarSection}>
                <h3 style={styles.sectionTitle}>Inputs ({selectedNode.inputs.length})</h3>
                {selectedNode.inputs.map((input, i) => (
                  <code key={i} style={styles.inputCode}>{input}</code>
                ))}
              </div>
            )}

            {SHADER_SNIPPETS[selectedNode.op] && (
              <div style={styles.sidebarSection}>
                <h3 style={styles.sectionTitle}>WebGL Shader</h3>
                <pre style={styles.shaderCode}>{SHADER_SNIPPETS[selectedNode.op]}</pre>
              </div>
            )}

            {!SHADER_SNIPPETS[selectedNode.op] && isSupported(selectedNode.op) && !isDataNode(selectedNode.op) && (
              <div style={styles.sidebarSection}>
                <h3 style={styles.sectionTitle}>Implementation</h3>
                <code style={styles.code}>packages/core/src/backends/webgl/ops/{selectedNode.op.toLowerCase()}/</code>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  main: {
    minHeight: "100vh",
    background: "linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%)",
    color: "#fff",
    fontFamily: "system-ui, -apple-system, sans-serif",
    padding: "1.5rem",
  },
  header: { marginBottom: "1.5rem" },
  backLink: { color: "#818cf8", textDecoration: "none", fontSize: "0.9rem", display: "inline-block", marginBottom: "1rem" },
  title: { fontSize: "2rem", fontWeight: 700, margin: "0 0 0.5rem 0" },
  subtitle: { color: "#9ca3af", margin: 0 },
  controls: { display: "flex", gap: "1rem", marginBottom: "1.5rem", flexWrap: "wrap" },
  select: { padding: "0.75rem 1rem", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.1)", background: "#1a1a2e", color: "#fff", fontSize: "1rem", minWidth: "200px" },
  loadButton: { padding: "0.75rem 1.5rem", borderRadius: "8px", border: "none", background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)", color: "#fff", fontWeight: 600, cursor: "pointer" },
  error: { padding: "1rem", background: "#ef444420", border: "1px solid #ef4444", borderRadius: "8px", marginBottom: "1rem", color: "#fca5a5" },
  stats: { display: "flex", gap: "1rem", marginBottom: "1.5rem", flexWrap: "wrap" },
  statCard: { padding: "1rem 1.5rem", background: "rgba(255,255,255,0.03)", borderRadius: "8px", borderWidth: "1px", borderStyle: "solid", borderColor: "rgba(255,255,255,0.1)", textAlign: "center" as const, minWidth: "100px" },
  statValue: { display: "block", fontSize: "1.75rem", fontWeight: 700, color: "#fff" },
  statLabel: { fontSize: "0.75rem", color: "#9ca3af", textTransform: "uppercase" as const, letterSpacing: "1px" },
  legend: { display: "flex", gap: "1.5rem", marginBottom: "1rem", fontSize: "0.85rem", color: "#9ca3af" },
  legendItem: { display: "flex", alignItems: "center", gap: "0.5rem" },
  legendDot: { width: "10px", height: "10px", borderRadius: "50%" },
  container: { display: "grid", gridTemplateColumns: "1fr 400px", gap: "1.5rem" },
  nodeList: { display: "flex", flexDirection: "column" as const, gap: "0.5rem", maxHeight: "70vh", overflowY: "auto" as const, paddingRight: "0.5rem" },
  loading: { padding: "2rem", textAlign: "center" as const, color: "#9ca3af" },
  nodeCard: { padding: "1rem", background: "rgba(255,255,255,0.03)", borderRadius: "8px", borderWidth: "1px", borderStyle: "solid", borderColor: "rgba(255,255,255,0.1)", cursor: "pointer", transition: "all 0.2s" },
  nodeSupported: { borderColor: "rgba(34, 197, 94, 0.3)", background: "rgba(34, 197, 94, 0.05)" },
  nodeUnsupported: { borderColor: "rgba(239, 68, 68, 0.3)", background: "rgba(239, 68, 68, 0.05)" },
  nodeData: { borderColor: "rgba(107, 114, 128, 0.3)", opacity: 0.7 },
  nodeSelected: { borderColor: "#818cf8", boxShadow: "0 0 0 2px rgba(129, 140, 248, 0.3)" },
  nodeHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.25rem" },
  nodeOp: { fontWeight: 600, fontSize: "1rem" },
  badge: { padding: "0.2rem 0.5rem", borderRadius: "4px", fontSize: "0.7rem", textTransform: "uppercase" as const, color: "#fff" },
  nodeName: { fontSize: "0.8rem", color: "#9ca3af", fontFamily: "'Fira Code', monospace", wordBreak: "break-all" as const },
  nodeInputs: { fontSize: "0.75rem", color: "#6b7280", marginTop: "0.5rem" },
  sidebar: { position: "sticky" as const, top: "1.5rem", background: "rgba(255,255,255,0.03)", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.1)", padding: "1.5rem", maxHeight: "80vh", overflowY: "auto" as const },
  sidebarHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" },
  sidebarTitle: { fontSize: "1.25rem", fontWeight: 600, margin: 0 },
  closeButton: { background: "none", border: "none", color: "#9ca3af", fontSize: "1.5rem", cursor: "pointer", padding: "0.25rem 0.5rem" },
  sidebarSection: { marginBottom: "1.25rem" },
  sectionTitle: { fontSize: "0.75rem", textTransform: "uppercase" as const, letterSpacing: "1px", color: "#6b7280", margin: "0 0 0.5rem 0" },
  code: { display: "block", padding: "0.5rem 0.75rem", background: "rgba(0,0,0,0.3)", borderRadius: "6px", fontFamily: "'Fira Code', monospace", fontSize: "0.85rem", wordBreak: "break-all" as const },
  inputCode: { display: "block", padding: "0.4rem 0.6rem", background: "rgba(0,0,0,0.2)", borderRadius: "4px", fontFamily: "'Fira Code', monospace", fontSize: "0.8rem", marginBottom: "0.25rem", color: "#9ca3af" },
  statusBadge: { display: "inline-block", padding: "0.5rem 0.75rem", borderRadius: "6px", fontWeight: 500 },
  shaderCode: { margin: 0, padding: "1rem", background: "rgba(0,0,0,0.4)", borderRadius: "8px", fontSize: "0.8rem", lineHeight: 1.5, overflowX: "auto" as const, fontFamily: "'Fira Code', monospace", color: "#a5b4fc" },
};
