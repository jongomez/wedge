"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { COLORS, DIMS, OPERATIONS, CATEGORY_COLORS, type OperationCategory } from "../../constants";
import type { CSSProperties, ReactNode } from "react";

export interface EditorNodeData extends Record<string, unknown> {
  opName: string;
  label: string;
  params: Record<string, unknown>;
}

interface BaseNodeProps {
  data: EditorNodeData;
  selected?: boolean;
  children?: ReactNode;
}

export function BaseNode({ data, selected, children }: BaseNodeProps) {
  const config = OPERATIONS[data.opName];
  if (!config) return null;

  const categoryColor = CATEGORY_COLORS[config.category as OperationCategory];

  const containerStyle: CSSProperties = {
    width: DIMS.nodeWidth,
    background: COLORS.bgTertiary,
    border: `2px solid ${selected ? COLORS.borderSelected : COLORS.border}`,
    borderRadius: DIMS.borderRadius,
    fontFamily: "system-ui, -apple-system, sans-serif",
    overflow: "hidden",
    boxShadow: selected ? `0 0 0 2px rgba(99,102,241,0.2)` : "none",
  };

  const headerStyle: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 10px",
    borderBottom: `1px solid ${COLORS.border}`,
  };

  const iconStyle: CSSProperties = {
    width: 24,
    height: 24,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: categoryColor,
    borderRadius: DIMS.borderRadius,
    fontSize: 10,
    fontWeight: 700,
    color: "#fff",
    flexShrink: 0,
  };

  const labelStyle: CSSProperties = {
    fontSize: 12,
    fontWeight: 600,
    color: COLORS.text,
    margin: 0,
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  };

  const bodyStyle: CSSProperties = {
    padding: "6px 10px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 8,
    minHeight: 28,
  };

  const handleContainerStyle: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    gap: 4,
  };

  const handleLabelStyle: CSSProperties = {
    fontSize: 9,
    color: COLORS.textMuted,
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  };

  const handleStyle: CSSProperties = {
    width: DIMS.handleSize,
    height: DIMS.handleSize,
    background: COLORS.accent,
    border: `2px solid ${COLORS.bg}`,
    borderRadius: DIMS.borderRadius,
    position: "relative",
  };

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>
        <div style={iconStyle}>{config.icon}</div>
        <p style={labelStyle}>{config.displayName}</p>
      </div>
      <div style={bodyStyle}>
        {/* Input handles */}
        <div style={handleContainerStyle}>
          {config.inputs.map((input, i) => (
            <div key={input.id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <Handle
                type="target"
                position={Position.Left}
                id={input.id}
                style={{
                  ...handleStyle,
                  top: "auto",
                  transform: "none",
                  position: "relative",
                  left: 0,
                  background: input.type === "weight" ? COLORS.textMuted : COLORS.accent,
                }}
              />
              <span style={handleLabelStyle}>{input.label}</span>
            </div>
          ))}
        </div>
        {/* Output handles */}
        <div style={{ ...handleContainerStyle, alignItems: "flex-end" }}>
          {config.outputs.map((output, i) => (
            <div key={output.id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={handleLabelStyle}>{output.label}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={output.id}
                style={{
                  ...handleStyle,
                  top: "auto",
                  transform: "none",
                  position: "relative",
                  right: 0,
                }}
              />
            </div>
          ))}
        </div>
      </div>
      {children}
    </div>
  );
}
