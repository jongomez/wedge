"use client";

import Link from "next/link";
import { useState, type CSSProperties } from "react";
import { COLORS, DIMS } from "../constants";

interface ModelOption {
  name: string;
  path: string;
}

interface EditorHeaderProps {
  onSave: () => void;
  onLoad: () => void;
  onExport: () => void;
  onLoadModel: (url: string, name: string) => void;
  availableModels: ModelOption[];
  loadingModel: boolean;
  loadedModelName: string | null;
}

export function EditorHeader({
  onSave,
  onLoad,
  onExport,
  onLoadModel,
  availableModels,
  loadingModel,
  loadedModelName,
}: EditorHeaderProps) {
  const [showModelDropdown, setShowModelDropdown] = useState(false);

  return (
    <header style={styles.header}>
      <div style={styles.left}>
        <Link href="/" style={styles.backLink}>
          <span style={styles.arrow}>&larr;</span>
          <span>Back</span>
        </Link>
        <div style={styles.divider} />
        <h1 style={styles.title}>Graph Editor</h1>
        {loadedModelName && (
          <span style={styles.modelBadge}>{loadedModelName}</span>
        )}
      </div>
      <div style={styles.right}>
        <div style={styles.dropdown}>
          <button
            style={styles.modelButton}
            onClick={() => setShowModelDropdown(!showModelDropdown)}
            disabled={loadingModel}
          >
            {loadingModel ? "Loading..." : "Load Model"}
            <span style={styles.chevron}>&#9662;</span>
          </button>
          {showModelDropdown && (
            <div style={styles.dropdownMenu}>
              {availableModels.map((model) => (
                <button
                  key={model.path}
                  style={styles.dropdownItem}
                  onClick={() => {
                    onLoadModel(model.path, model.name);
                    setShowModelDropdown(false);
                  }}
                >
                  {model.name}
                </button>
              ))}
            </div>
          )}
        </div>
        <div style={styles.divider} />
        <button style={styles.button} onClick={onLoad}>
          Load File
        </button>
        <button style={styles.button} onClick={onSave}>
          Save
        </button>
        <button style={styles.primaryButton} onClick={onExport}>
          Export to Wedge
        </button>
      </div>
    </header>
  );
}

const styles: Record<string, CSSProperties> = {
  header: {
    height: 48,
    background: COLORS.bgSecondary,
    borderBottom: `1px solid ${COLORS.border}`,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 16px",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  left: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  backLink: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    color: COLORS.textSecondary,
    textDecoration: "none",
    fontSize: 12,
    fontWeight: 500,
    transition: "color 0.15s",
  },
  arrow: {
    fontSize: 14,
  },
  divider: {
    width: 1,
    height: 20,
    background: COLORS.border,
  },
  title: {
    fontSize: 13,
    fontWeight: 600,
    color: COLORS.text,
    margin: 0,
  },
  right: {
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  button: {
    padding: "6px 12px",
    background: "transparent",
    border: `1px solid ${COLORS.border}`,
    borderRadius: DIMS.borderRadius,
    color: COLORS.textSecondary,
    fontSize: 11,
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.15s",
  },
  primaryButton: {
    padding: "6px 12px",
    background: COLORS.accent,
    border: `1px solid ${COLORS.accent}`,
    borderRadius: DIMS.borderRadius,
    color: "#fff",
    fontSize: 11,
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.15s",
  },
  modelBadge: {
    padding: "4px 8px",
    background: "rgba(99,102,241,0.15)",
    border: `1px solid rgba(99,102,241,0.3)`,
    borderRadius: DIMS.borderRadius,
    color: COLORS.accentLight,
    fontSize: 10,
    fontWeight: 500,
  },
  dropdown: {
    position: "relative" as const,
  },
  modelButton: {
    padding: "6px 12px",
    background: "rgba(99,102,241,0.1)",
    border: `1px solid rgba(99,102,241,0.3)`,
    borderRadius: DIMS.borderRadius,
    color: COLORS.accentLight,
    fontSize: 11,
    fontWeight: 500,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: 6,
  },
  chevron: {
    fontSize: 8,
  },
  dropdownMenu: {
    position: "absolute" as const,
    top: "100%",
    left: 0,
    marginTop: 4,
    background: COLORS.bgSecondary,
    border: `1px solid ${COLORS.border}`,
    borderRadius: DIMS.borderRadius,
    minWidth: 160,
    zIndex: 100,
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
  },
  dropdownItem: {
    display: "block",
    width: "100%",
    padding: "8px 12px",
    background: "transparent",
    border: "none",
    color: COLORS.text,
    fontSize: 11,
    textAlign: "left" as const,
    cursor: "pointer",
  },
};
