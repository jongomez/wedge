"use client";

import type { CSSProperties } from "react";
import { COLORS, DIMS, OPERATIONS } from "../constants";
import type { EditorNode } from "../hooks/useEditorState";

interface PropertiesPanelProps {
  node: EditorNode | null;
  onUpdateParams: (nodeId: string, params: Record<string, unknown>) => void;
  onDelete: (nodeId: string) => void;
}

export function PropertiesPanel({ node, onUpdateParams, onDelete }: PropertiesPanelProps) {
  if (!node) {
    return (
      <div style={styles.container}>
        <div style={styles.empty}>
          <span style={styles.emptyText}>Select a node to edit</span>
        </div>
      </div>
    );
  }

  const config = OPERATIONS[node.data.opName];
  if (!config) return null;

  const handleParamChange = (key: string, value: string) => {
    let parsed: unknown = value;

    // Try to parse as JSON for arrays/objects
    if (value.startsWith("[") || value.startsWith("{")) {
      try {
        parsed = JSON.parse(value);
      } catch {
        parsed = value;
      }
    } else if (value === "true" || value === "false") {
      parsed = value === "true";
    } else if (!isNaN(Number(value)) && value !== "") {
      parsed = Number(value);
    }

    onUpdateParams(node.id, { [key]: parsed });
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>Properties</span>
      </div>
      <div style={styles.content}>
        <div style={styles.section}>
          <div style={styles.field}>
            <label style={styles.label}>Operation</label>
            <div style={styles.value}>{config.displayName}</div>
          </div>
          <div style={styles.field}>
            <label style={styles.label}>ID</label>
            <div style={{ ...styles.value, fontSize: 10, color: COLORS.textMuted }}>
              {node.id}
            </div>
          </div>
        </div>

        {Object.keys(node.data.params).length > 0 && (
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Parameters</div>
            {Object.entries(node.data.params).map(([key, value]) => (
              <div key={key} style={styles.field}>
                <label style={styles.label}>{key}</label>
                <input
                  type="text"
                  style={styles.input}
                  value={typeof value === "object" ? JSON.stringify(value) : String(value)}
                  onChange={(e) => handleParamChange(key, e.target.value)}
                />
              </div>
            ))}
          </div>
        )}
      </div>
      <div style={styles.footer}>
        <button style={styles.deleteButton} onClick={() => onDelete(node.id)}>
          Delete Node
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    width: 240,
    height: "100%",
    background: COLORS.bgSecondary,
    borderLeft: `1px solid ${COLORS.border}`,
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
    padding: "12px 16px",
  },
  section: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    color: COLORS.textMuted,
    marginBottom: 8,
  },
  field: {
    marginBottom: 12,
  },
  label: {
    display: "block",
    fontSize: 10,
    fontWeight: 500,
    color: COLORS.textSecondary,
    marginBottom: 4,
    textTransform: "capitalize",
  },
  value: {
    fontSize: 12,
    color: COLORS.text,
  },
  input: {
    width: "100%",
    padding: "6px 8px",
    background: COLORS.bgTertiary,
    border: `1px solid ${COLORS.border}`,
    borderRadius: DIMS.borderRadius,
    color: COLORS.text,
    fontSize: 11,
    fontFamily: "'Fira Code', monospace",
    outline: "none",
  },
  empty: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  emptyText: {
    fontSize: 11,
    color: COLORS.textMuted,
  },
  footer: {
    padding: "12px 16px",
    borderTop: `1px solid ${COLORS.border}`,
  },
  deleteButton: {
    width: "100%",
    padding: "8px 12px",
    background: "transparent",
    border: `1px solid rgba(239,68,68,0.3)`,
    borderRadius: DIMS.borderRadius,
    color: "#ef4444",
    fontSize: 11,
    fontWeight: 500,
    cursor: "pointer",
  },
};
