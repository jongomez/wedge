"use client";

import type { DragEvent, CSSProperties } from "react";
import { OPERATIONS, OPERATION_CATEGORIES, COLORS, DIMS, CATEGORY_COLORS, type OperationCategory } from "../constants";

interface ToolbarProps {
  onClear: () => void;
}

export function Toolbar({ onClear }: ToolbarProps) {
  const onDragStart = (event: DragEvent, opName: string) => {
    event.dataTransfer.setData("application/reactflow", opName);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>Operations</span>
      </div>
      <div style={styles.content}>
        {OPERATION_CATEGORIES.map((category) => (
          <div key={category.category} style={styles.category}>
            <div style={styles.categoryHeader}>
              <div
                style={{
                  ...styles.categoryDot,
                  background: CATEGORY_COLORS[category.category as OperationCategory],
                }}
              />
              <span style={styles.categoryName}>{category.name}</span>
            </div>
            <div style={styles.opsList}>
              {category.ops.map((opName) => {
                const config = OPERATIONS[opName];
                if (!config) return null;
                return (
                  <div
                    key={opName}
                    style={styles.opItem}
                    draggable
                    onDragStart={(e) => onDragStart(e, opName)}
                  >
                    <div
                      style={{
                        ...styles.opIcon,
                        background: CATEGORY_COLORS[config.category as OperationCategory],
                      }}
                    >
                      {config.icon}
                    </div>
                    <span style={styles.opName}>{config.displayName}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div style={styles.footer}>
        <button style={styles.clearButton} onClick={onClear}>
          Clear All
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    width: 200,
    height: "100%",
    background: COLORS.bgSecondary,
    borderRight: `1px solid ${COLORS.border}`,
    display: "flex",
    flexDirection: "column",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  header: {
    padding: "16px",
    borderBottom: `1px solid ${COLORS.border}`,
  },
  title: {
    fontSize: 11,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "1px",
    color: COLORS.textMuted,
  },
  content: {
    flex: 1,
    overflowY: "auto",
    padding: "8px 0",
  },
  category: {
    marginBottom: 8,
  },
  categoryHeader: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 16px",
  },
  categoryDot: {
    width: 8,
    height: 8,
    borderRadius: 1,
  },
  categoryName: {
    fontSize: 10,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    color: COLORS.textSecondary,
  },
  opsList: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
  },
  opItem: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "8px 16px",
    cursor: "grab",
    transition: "background 0.15s",
    userSelect: "none",
  },
  opIcon: {
    width: 24,
    height: 24,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: DIMS.borderRadius,
    fontSize: 9,
    fontWeight: 700,
    color: "#fff",
    flexShrink: 0,
  },
  opName: {
    fontSize: 12,
    color: COLORS.text,
  },
  footer: {
    padding: "12px 16px",
    borderTop: `1px solid ${COLORS.border}`,
  },
  clearButton: {
    width: "100%",
    padding: "8px 12px",
    background: "transparent",
    border: `1px solid ${COLORS.border}`,
    borderRadius: DIMS.borderRadius,
    color: COLORS.textSecondary,
    fontSize: 11,
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.15s",
  },
};
