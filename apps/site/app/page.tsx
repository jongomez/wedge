"use client";

import Link from "next/link";
import { useState, useEffect } from "react";

const features = [
  {
    title: "WebGL2 Acceleration",
    description: "Run neural networks directly on the GPU using custom GLSL shaders",
    icon: "GPU",
  },
  {
    title: "TensorFlow.js Compatible",
    description: "Load and execute LayersModel and GraphModel formats",
    icon: "TF",
  },
  {
    title: "Custom Shader Pipeline",
    description: "Optimized texture-based computation for maximum performance",
    icon: "SHD",
  },
];

const operations = [
  "Conv2D",
  "DepthwiseConv2D",
  "ReLU",
  "Add",
  "Multiply",
  "ResizeBilinear",
];

export default function Home() {
  const [activeOp, setActiveOp] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveOp((prev) => (prev + 1) % operations.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <main style={styles.main}>
      <div style={styles.hero}>
        <div style={styles.badge}>WebGL Neural Network Engine</div>
        <h1 style={styles.title}>
          <span style={styles.titleAccent}>Wedge</span>
        </h1>
        <p style={styles.subtitle}>
          GPU-accelerated neural network inference for the browser.
          <br />
          Custom WebGL2 shaders. Zero dependencies on TensorFlow runtime.
        </p>

        <div style={styles.buttonGroup}>
          <Link href="/model" style={styles.primaryButton}>
            Visualize Model
          </Link>
          <Link href="/editor" style={styles.secondaryButton}>
            Graph Editor
          </Link>
          <Link href="/tests" style={styles.secondaryButton}>
            Run Tests
          </Link>
          <Link href="/classification" style={styles.secondaryButton}>
            Try Classification
          </Link>
        </div>
      </div>

      <div style={styles.opsShowcase}>
        <div style={styles.opsLabel}>Supported Operations</div>
        <div style={styles.opsList}>
          {operations.map((op, i) => (
            <span
              key={op}
              style={{
                ...styles.opTag,
                ...(i === activeOp ? styles.opTagActive : {}),
              }}
            >
              {op}
            </span>
          ))}
        </div>
      </div>

      <div style={styles.features}>
        {features.map((feature) => (
          <div key={feature.title} style={styles.featureCard}>
            <div style={styles.featureIcon}>{feature.icon}</div>
            <h3 style={styles.featureTitle}>{feature.title}</h3>
            <p style={styles.featureDesc}>{feature.description}</p>
          </div>
        ))}
      </div>

      <div style={styles.codeBlock}>
        <div style={styles.codeHeader}>
          <span style={styles.codeDot} />
          <span style={{ ...styles.codeDot, background: "#febc2e" }} />
          <span style={{ ...styles.codeDot, background: "#28c840" }} />
          <span style={styles.codeTitle}>Quick Start</span>
        </div>
        <pre style={styles.code}>
{`import { WedgeWebGL } from '@wedge/core';

const wedge = new WedgeWebGL();
await wedge.loadGraphModel('/models/model.json');

const output = wedge.run(inputData);`}
        </pre>
      </div>

      <footer style={styles.footer}>
        <p>Built for performance. Designed for the web.</p>
      </footer>
    </main>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  main: {
    minHeight: "100vh",
    background: "linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%)",
    color: "#fff",
    fontFamily: "system-ui, -apple-system, sans-serif",
    padding: "2rem",
  },
  hero: {
    textAlign: "center",
    paddingTop: "4rem",
    paddingBottom: "3rem",
  },
  badge: {
    display: "inline-block",
    padding: "0.5rem 1rem",
    background: "rgba(99, 102, 241, 0.15)",
    border: "1px solid rgba(99, 102, 241, 0.3)",
    borderRadius: "100px",
    fontSize: "0.85rem",
    color: "#818cf8",
    marginBottom: "1.5rem",
    letterSpacing: "0.5px",
  },
  title: {
    fontSize: "4.5rem",
    fontWeight: 700,
    margin: 0,
    letterSpacing: "-2px",
  },
  titleAccent: {
    background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    backgroundClip: "text",
  },
  subtitle: {
    fontSize: "1.2rem",
    color: "#9ca3af",
    marginTop: "1.5rem",
    lineHeight: 1.7,
  },
  buttonGroup: {
    display: "flex",
    gap: "1rem",
    justifyContent: "center",
    marginTop: "2.5rem",
  },
  primaryButton: {
    padding: "0.875rem 2rem",
    background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    color: "#fff",
    borderRadius: "8px",
    fontWeight: 600,
    fontSize: "1rem",
    textDecoration: "none",
    transition: "transform 0.2s, box-shadow 0.2s",
  },
  secondaryButton: {
    padding: "0.875rem 2rem",
    background: "transparent",
    color: "#fff",
    borderRadius: "8px",
    fontWeight: 600,
    fontSize: "1rem",
    textDecoration: "none",
    border: "1px solid rgba(255, 255, 255, 0.2)",
  },
  opsShowcase: {
    textAlign: "center",
    marginTop: "3rem",
    marginBottom: "4rem",
  },
  opsLabel: {
    fontSize: "0.75rem",
    textTransform: "uppercase",
    letterSpacing: "2px",
    color: "#6b7280",
    marginBottom: "1rem",
  },
  opsList: {
    display: "flex",
    flexWrap: "wrap",
    gap: "0.75rem",
    justifyContent: "center",
  },
  opTag: {
    padding: "0.5rem 1rem",
    background: "rgba(255, 255, 255, 0.05)",
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: "rgba(255, 255, 255, 0.1)",
    borderRadius: "6px",
    fontSize: "0.9rem",
    color: "#9ca3af",
    fontFamily: "'Fira Code', monospace",
    transition: "all 0.3s ease",
  },
  opTagActive: {
    background: "rgba(99, 102, 241, 0.2)",
    borderColor: "#6366f1",
    color: "#a5b4fc",
    transform: "scale(1.05)",
  },
  features: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: "1.5rem",
    maxWidth: "1000px",
    margin: "0 auto",
    padding: "2rem 0",
  },
  featureCard: {
    padding: "2rem",
    background: "rgba(255, 255, 255, 0.03)",
    border: "1px solid rgba(255, 255, 255, 0.08)",
    borderRadius: "12px",
  },
  featureIcon: {
    width: "48px",
    height: "48px",
    background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    borderRadius: "10px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 700,
    fontSize: "0.85rem",
    marginBottom: "1rem",
  },
  featureTitle: {
    fontSize: "1.1rem",
    fontWeight: 600,
    margin: "0 0 0.5rem 0",
  },
  featureDesc: {
    fontSize: "0.95rem",
    color: "#9ca3af",
    margin: 0,
    lineHeight: 1.6,
  },
  codeBlock: {
    maxWidth: "600px",
    margin: "3rem auto",
    borderRadius: "12px",
    overflow: "hidden",
    background: "#0d0d0d",
    border: "1px solid rgba(255, 255, 255, 0.1)",
  },
  codeHeader: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "12px 16px",
    background: "rgba(255, 255, 255, 0.05)",
    borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
  },
  codeDot: {
    width: "12px",
    height: "12px",
    borderRadius: "50%",
    background: "#ff5f56",
  },
  codeTitle: {
    marginLeft: "auto",
    fontSize: "0.8rem",
    color: "#6b7280",
  },
  code: {
    margin: 0,
    padding: "1.5rem",
    fontSize: "0.9rem",
    lineHeight: 1.7,
    color: "#e5e7eb",
    fontFamily: "'Fira Code', 'Consolas', monospace",
    overflow: "auto",
  },
  footer: {
    textAlign: "center",
    padding: "4rem 0 2rem",
    color: "#4b5563",
    fontSize: "0.9rem",
  },
};
