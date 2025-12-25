"use client";

import Link from "next/link";
import { useState, useRef, useEffect, useCallback } from "react";
import { createWedgeFromGraphModel, prepareInputForWedge, WedgeInstance } from "@wedge/core/create";
import { defaultOptions } from "@wedge/core/constants";
import { WedgeOptions } from "@wedge/core/backends/webgl/types";

type Status = "idle" | "loading-model" | "waiting" | "trying" | "retrying" | "ready" | "running" | "error";

const MODEL_URL = "/models/blaze_lite/model.json";
const INPUT_SIZE = 256; // BlazePose lite input size

// BlazePose outputs 39 keypoints
const KEYPOINT_NAMES = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer",
  "right_eye_inner", "right_eye", "right_eye_outer",
  "left_ear", "right_ear", "mouth_left", "mouth_right",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky",
  "left_index", "right_index", "left_thumb", "right_thumb",
  "left_hip", "right_hip", "left_knee", "right_knee",
  "left_ankle", "right_ankle", "left_heel", "right_heel",
  "left_foot_index", "right_foot_index",
  // Additional BlazePose keypoints
  "body_center", "forehead", "left_thumb_tip", "left_index_tip",
  "right_thumb_tip", "right_index_tip"
];

export default function Classification() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hasCamera, setHasCamera] = useState(true);
  const wedgeRef = useRef<WedgeInstance | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);

  // Load the model
  useEffect(() => {
    let isMounted = true;

    async function loadModel() {
      setStatus("loading-model");
      try {
        const options: WedgeOptions = {
          ...defaultOptions,
          hasBatchDimension: false,
        };
        const wedge = await createWedgeFromGraphModel(MODEL_URL, options);
        if (isMounted) {
          wedgeRef.current = wedge;
          console.log("Model loaded successfully");
        }
      } catch (err) {
        console.error("Model loading error:", err);
        if (isMounted) {
          setError("Failed to load model: " + (err instanceof Error ? err.message : String(err)));
          setStatus("error");
        }
        return;
      }

      // Now try to get camera
      await initCamera();
    }

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
      // Wait a bit before first attempt
      setStatus("waiting");
      await new Promise((resolve) => setTimeout(resolve, 1000));

      if (!isMounted) return;

      // First attempt
      setStatus("trying");
      const firstAttempt = await tryCamera();

      if (firstAttempt || !isMounted) return;

      // Wait before retry
      setStatus("retrying");
      await new Promise((resolve) => setTimeout(resolve, 2000));

      if (!isMounted) return;

      // Second attempt
      const secondAttempt = await tryCamera();

      if (!secondAttempt && isMounted) {
        setHasCamera(false);
        setError("Camera access denied or not available");
        setStatus("error");
      }
    }

    loadModel();

    return () => {
      isMounted = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  // Extract frame from video and prepare for inference
  const extractFrame = useCallback((): Float32Array | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    // Draw video frame to canvas at model input size
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;

    // Calculate crop to maintain aspect ratio (center crop)
    const videoAspect = video.videoWidth / video.videoHeight;
    const targetAspect = 1; // Square input
    let sx = 0, sy = 0, sw = video.videoWidth, sh = video.videoHeight;

    if (videoAspect > targetAspect) {
      // Video is wider, crop sides
      sw = video.videoHeight * targetAspect;
      sx = (video.videoWidth - sw) / 2;
    } else {
      // Video is taller, crop top/bottom
      sh = video.videoWidth / targetAspect;
      sy = (video.videoHeight - sh) / 2;
    }

    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, INPUT_SIZE, INPUT_SIZE);

    // Get image data and convert to Float32Array
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const data = imageData.data;

    // Convert RGBA to RGB and normalize to [0, 1]
    const floatData = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
      floatData[i * 3] = data[i * 4] / 255.0;     // R
      floatData[i * 3 + 1] = data[i * 4 + 1] / 255.0; // G
      floatData[i * 3 + 2] = data[i * 4 + 2] / 255.0; // B
    }

    return floatData;
  }, []);

  // Draw keypoints on canvas
  const drawKeypoints = useCallback((output: Float32Array) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear previous drawings (but keep the video frame)
    // We'll draw on top of the video

    // The output is a heatmap of shape [64, 64, 39] for 39 keypoints
    // We need to find the max activation location for each keypoint
    const heatmapSize = 64;
    const numKeypoints = 39;

    for (let k = 0; k < Math.min(numKeypoints, KEYPOINT_NAMES.length); k++) {
      let maxVal = -Infinity;
      let maxX = 0, maxY = 0;

      // Find peak in heatmap for this keypoint
      for (let y = 0; y < heatmapSize; y++) {
        for (let x = 0; x < heatmapSize; x++) {
          const idx = (y * heatmapSize + x) * numKeypoints + k;
          if (idx < output.length && output[idx] > maxVal) {
            maxVal = output[idx];
            maxX = x;
            maxY = y;
          }
        }
      }

      // Only draw if confidence is high enough
      if (maxVal > 0.3) {
        // Scale coordinates from heatmap to canvas
        const canvasX = (maxX / heatmapSize) * canvas.width;
        const canvasY = (maxY / heatmapSize) * canvas.height;

        // Draw keypoint
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, 4, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(0, 255, 0, ${Math.min(1, maxVal)})`;
        ctx.fill();
        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }, []);

  // Run inference loop
  const runInference = useCallback(() => {
    if (!wedgeRef.current || status !== "running") return;

    const startTime = performance.now();

    // Extract frame
    const frameData = extractFrame();
    if (!frameData) {
      animationFrameRef.current = requestAnimationFrame(runInference);
      return;
    }

    try {
      // Prepare input for Wedge (handles padding and texture layout)
      const preparedInput = prepareInputForWedge(
        frameData,
        [INPUT_SIZE, INPUT_SIZE, 3],
        { ...defaultOptions, hasBatchDimension: false }
      );

      // Run inference
      const output = wedgeRef.current.predict([preparedInput]);

      // Draw keypoints
      drawKeypoints(output);

      // Update timing stats
      const endTime = performance.now();
      setInferenceTime(endTime - startTime);

      // Update FPS counter
      frameCountRef.current++;
      if (endTime - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastTimeRef.current = endTime;
      }
    } catch (err) {
      console.error("Inference error:", err);
    }

    // Schedule next frame
    animationFrameRef.current = requestAnimationFrame(runInference);
  }, [status, extractFrame, drawKeypoints]);

  // Start/stop inference when status changes
  useEffect(() => {
    if (status === "running") {
      lastTimeRef.current = performance.now();
      frameCountRef.current = 0;
      animationFrameRef.current = requestAnimationFrame(runInference);
    } else if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [status, runInference]);

  // Start inference when camera is ready
  const handleStartInference = useCallback(() => {
    if (status === "ready" && wedgeRef.current) {
      setStatus("running");
    }
  }, [status]);

  const handleStopInference = useCallback(() => {
    if (status === "running") {
      setStatus("ready");
    }
  }, [status]);

  return (
    <main style={styles.main}>
      <div style={styles.header}>
        <Link href="/" style={styles.backLink}>
          ← Back
        </Link>
        <h1 style={styles.title}>BlazePose Lite Demo</h1>
        <p style={styles.subtitle}>
          Real-time pose estimation using Wedge WebGL engine
        </p>
      </div>

      <div style={styles.container}>
        <div style={styles.videoContainer}>
          {(status === "idle" || status === "loading-model") && (
            <div style={styles.overlay}>
              <div style={styles.spinner} />
              <p>Loading BlazePose Lite model...</p>
            </div>
          )}

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
              display: hasCamera && (status === "ready" || status === "running") ? "block" : "none",
            }}
            playsInline
            muted
          />
          <canvas
            ref={canvasRef}
            style={{
              ...styles.canvas,
              display: status === "running" ? "block" : "none",
            }}
            width={INPUT_SIZE}
            height={INPUT_SIZE}
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
                    status === "running"
                      ? "#22c55e"
                      : status === "ready"
                      ? "#3b82f6"
                      : status === "error"
                      ? "#ef4444"
                      : "#eab308",
                }}
              />
              <span style={styles.statusText}>
                {status === "idle" && "Initializing..."}
                {status === "loading-model" && "Loading Model..."}
                {status === "waiting" && "Preparing..."}
                {status === "trying" && "Accessing camera..."}
                {status === "retrying" && "Retrying..."}
                {status === "ready" && "Ready to Start"}
                {status === "running" && "Running Inference"}
                {status === "error" && "Error"}
              </span>
            </div>

            {(status === "ready" || status === "running") && (
              <button
                style={{
                  ...styles.actionButton,
                  background: status === "running"
                    ? "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
                    : "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)",
                }}
                onClick={status === "running" ? handleStopInference : handleStartInference}
              >
                {status === "running" ? "Stop Inference" : "Start Inference"}
              </button>
            )}
          </div>

          {status === "running" && (
            <div style={styles.statusCard}>
              <h3 style={styles.cardTitle}>Performance</h3>
              <div style={styles.perfRow}>
                <span style={styles.perfLabel}>FPS:</span>
                <span style={styles.perfValue}>{fps}</span>
              </div>
              <div style={styles.perfRow}>
                <span style={styles.perfLabel}>Inference:</span>
                <span style={styles.perfValue}>{inferenceTime.toFixed(1)} ms</span>
              </div>
            </div>
          )}

          <div style={styles.statusCard}>
            <h3 style={styles.cardTitle}>Model</h3>
            <p style={styles.modelName}>BlazePose Lite</p>
            <p style={styles.modelInfo}>Pose estimation (39 keypoints)</p>
            <p style={styles.modelPath}>{MODEL_URL}</p>
          </div>

          <div style={styles.statusCard}>
            <h3 style={styles.cardTitle}>About</h3>
            <p style={styles.comingSoon}>
              This demo runs BlazePose Lite entirely on WebGL using the Wedge inference engine.
              All 343 operations are executed as custom GLSL shaders for maximum GPU utilization.
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
  actionButton: {
    marginTop: "1rem",
    padding: "0.75rem 1.5rem",
    borderRadius: "8px",
    border: "none",
    color: "#fff",
    fontWeight: 600,
    fontSize: "0.9rem",
    cursor: "pointer",
    width: "100%",
    transition: "transform 0.2s, box-shadow 0.2s",
  },
  perfRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "0.5rem 0",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
  },
  perfLabel: {
    color: "#9ca3af",
    fontSize: "0.9rem",
  },
  perfValue: {
    color: "#fff",
    fontSize: "1.1rem",
    fontWeight: 600,
    fontFamily: "'Fira Code', monospace",
  },
};
