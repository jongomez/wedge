"use client";

import Link from "next/link";
import { useState, useRef, useEffect } from "react";

type Status = "idle" | "waiting" | "trying" | "retrying" | "ready" | "error";

export default function Classification() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hasCamera, setHasCamera] = useState(true);

  useEffect(() => {
    let isMounted = true;

    async function tryCamera(): Promise<boolean> {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: 640, height: 480 },
        });
        if (isMounted && videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setStatus("ready");
          return true;
        }
        return false;
      } catch (err) {
        console.error("Camera error:", err);
        return false;
      }
    }

    async function initCamera() {
      // Wait 2 seconds before first attempt
      setStatus("waiting");
      await new Promise((resolve) => setTimeout(resolve, 2000));

      if (!isMounted) return;

      // First attempt
      setStatus("trying");
      const firstAttempt = await tryCamera();

      if (firstAttempt || !isMounted) return;

      // Wait 3 seconds before retry
      setStatus("retrying");
      await new Promise((resolve) => setTimeout(resolve, 3000));

      if (!isMounted) return;

      // Second attempt
      const secondAttempt = await tryCamera();

      if (!secondAttempt && isMounted) {
        setHasCamera(false);
        setError("Camera access denied or not available");
        setStatus("error");
      }
    }

    initCamera();

    return () => {
      isMounted = false;
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <main style={styles.main}>
      <div style={styles.header}>
        <Link href="/" style={styles.backLink}>
          ← Back
        </Link>
        <h1 style={styles.title}>Image Classification</h1>
        <p style={styles.subtitle}>
          Real-time classification using Wedge WebGL engine
        </p>
      </div>

      <div style={styles.container}>
        <div style={styles.videoContainer}>
          {(status === "waiting" || status === "trying" || status === "retrying") && (
            <div style={styles.overlay}>
              <div style={styles.spinner} />
              <p>
                {status === "waiting" && "Preparing camera..."}
                {status === "trying" && "Trying to access camera..."}
                {status === "retrying" && "Retrying camera access..."}
              </p>
            </div>
          )}

          {status === "error" && (
            <div style={styles.overlay}>
              <div style={styles.errorIcon}>⚠</div>
              <p>{error}</p>
              <p style={styles.hint}>
                Please allow camera access or use a device with a camera.
              </p>
            </div>
          )}

          <video
            ref={videoRef}
            style={{
              ...styles.video,
              display: hasCamera ? "block" : "none",
            }}
            playsInline
            muted
          />
          <canvas
            ref={canvasRef}
            style={styles.canvas}
            width={640}
            height={480}
          />
        </div>

        <div style={styles.infoPanel}>
          <div style={styles.statusCard}>
            <h3 style={styles.cardTitle}>Status</h3>
            <div style={styles.statusRow}>
              <span
                style={{
                  ...styles.statusDot,
                  background:
                    status === "ready"
                      ? "#22c55e"
                      : status === "error"
                      ? "#ef4444"
                      : "#eab308",
                }}
              />
              <span style={styles.statusText}>
                {status === "idle" && "Idle"}
                {status === "waiting" && "Preparing..."}
                {status === "trying" && "Accessing camera..."}
                {status === "retrying" && "Retrying..."}
                {status === "ready" && "Camera Ready"}
                {status === "error" && "Error"}
              </span>
            </div>
          </div>

          <div style={styles.statusCard}>
            <h3 style={styles.cardTitle}>Model</h3>
            <p style={styles.modelName}>BlazeFace Lite</p>
            <p style={styles.modelInfo}>Face detection model</p>
            <p style={styles.modelPath}>/models/blaze_lite/model.json</p>
          </div>

          <div style={styles.statusCard}>
            <h3 style={styles.cardTitle}>Coming Soon</h3>
            <p style={styles.comingSoon}>
              Full classification pipeline with Wedge WebGL inference is under
              development. The model loading and inference code needs to be
              connected.
            </p>
          </div>
        </div>
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
  header: {
    marginBottom: "2rem",
  },
  backLink: {
    color: "#818cf8",
    textDecoration: "none",
    fontSize: "0.9rem",
    display: "inline-block",
    marginBottom: "1rem",
  },
  title: {
    fontSize: "2rem",
    fontWeight: 700,
    margin: "0 0 0.5rem 0",
  },
  subtitle: {
    color: "#9ca3af",
    margin: 0,
  },
  container: {
    display: "grid",
    gridTemplateColumns: "1fr 320px",
    gap: "1.5rem",
    maxWidth: "1200px",
    margin: "0 auto",
  },
  videoContainer: {
    position: "relative",
    background: "#000",
    borderRadius: "12px",
    overflow: "hidden",
    aspectRatio: "4/3",
    border: "1px solid rgba(255, 255, 255, 0.1)",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  canvas: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    pointerEvents: "none",
  },
  overlay: {
    position: "absolute",
    inset: 0,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    background: "rgba(0, 0, 0, 0.8)",
    color: "#9ca3af",
    gap: "1rem",
  },
  spinner: {
    width: "40px",
    height: "40px",
    border: "3px solid rgba(255, 255, 255, 0.1)",
    borderTopColor: "#818cf8",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
  },
  errorIcon: {
    fontSize: "3rem",
  },
  hint: {
    fontSize: "0.85rem",
    color: "#6b7280",
  },
  infoPanel: {
    display: "flex",
    flexDirection: "column",
    gap: "1rem",
  },
  statusCard: {
    background: "rgba(255, 255, 255, 0.03)",
    border: "1px solid rgba(255, 255, 255, 0.08)",
    borderRadius: "12px",
    padding: "1.25rem",
  },
  cardTitle: {
    fontSize: "0.75rem",
    textTransform: "uppercase",
    letterSpacing: "1px",
    color: "#6b7280",
    margin: "0 0 0.75rem 0",
  },
  statusRow: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
  },
  statusDot: {
    width: "10px",
    height: "10px",
    borderRadius: "50%",
  },
  statusText: {
    fontWeight: 500,
  },
  modelName: {
    fontSize: "1.1rem",
    fontWeight: 600,
    margin: "0 0 0.25rem 0",
  },
  modelInfo: {
    color: "#9ca3af",
    fontSize: "0.9rem",
    margin: "0 0 0.5rem 0",
  },
  modelPath: {
    fontFamily: "'Fira Code', monospace",
    fontSize: "0.8rem",
    color: "#6b7280",
    margin: 0,
    wordBreak: "break-all",
  },
  comingSoon: {
    color: "#9ca3af",
    fontSize: "0.9rem",
    lineHeight: 1.6,
    margin: 0,
  },
};
