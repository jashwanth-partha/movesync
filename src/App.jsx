import { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

function App() {
  const [screen, setScreen] = useState("home");
  const [cameraState, setCameraState] = useState("idle");
  const [message, setMessage] = useState("Press Enable Camera to begin");
  const [movementDetected, setMovementDetected] = useState(false);
  const [score, setScore] = useState(0);
  const [poseReady, setPoseReady] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const poseRef = useRef(null);
  const animationRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      if (poseRef.current) {
        poseRef.current.close();
      }
    };
  }, []);

  const drawPose = (landmarks, canvas, ctx) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks) return;

    const points = landmarks.map((landmark) => ({
      x: landmark.x * canvas.width,
      y: landmark.y * canvas.height,
    }));

    const connections = [
      [11, 12],
      [11, 13],
      [13, 15],
      [12, 14],
      [14, 16],
      [11, 23],
      [12, 24],
      [23, 24],
      [23, 25],
      [25, 27],
      [24, 26],
      [26, 28],
      [27, 31],
      [28, 32],
      [15, 17],
      [15, 19],
      [15, 21],
      [16, 18],
      [16, 20],
      [16, 22],
    ];

    ctx.strokeStyle = "rgba(116, 255, 159, 0.9)";
    ctx.lineWidth = 3;

    for (const [a, b] of connections) {
      if (!points[a] || !points[b]) continue;

      ctx.beginPath();
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
      ctx.stroke();
    }

    ctx.fillStyle = "#ffffff";

    for (const point of points) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const processFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const pose = poseRef.current;

    if (!video || !canvas || !pose || video.readyState < 2) {
      animationRef.current = requestAnimationFrame(processFrame);
      return;
    }

    if (video.currentTime !== lastVideoTimeRef.current) {
      const result = pose.detectForVideo(video, performance.now());
      const ctx = canvas.getContext("2d");

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const landmarks = result.landmarks?.[0];

      drawPose(landmarks, canvas, ctx);

      if (landmarks) {
        const rightShoulder = landmarks[12];
        const rightWrist = landmarks[16];
        const rightElbow = landmarks[14];

        const visible =
          rightShoulder?.visibility > 0.45 &&
          rightWrist?.visibility > 0.45 &&
          rightElbow?.visibility > 0.45;

        if (visible) {
          const raised = rightWrist.y < rightShoulder.y - 0.08;

          setMovementDetected(raised);

          if (raised) {
            setMessage("Great! Right hand detected ✓");
            setScore(100);
          } else {
            setMessage("Raise your RIGHT hand above your shoulder");
            setScore(0);
          }
        }
      } else {
        setMovementDetected(false);
        setMessage("Step into the camera frame");
        setScore(0);
      }

      lastVideoTimeRef.current = video.currentTime;
    }

    animationRef.current = requestAnimationFrame(processFrame);
  };

  const startCamera = async () => {
    try {
      setCameraState("loading");
      setMessage("Loading camera and movement tracker...");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      const vision = await FilesetResolver.forVisionTasks(WASM_URL);

      const pose = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      poseRef.current = pose;
      setPoseReady(true);

      const video = videoRef.current;

      video.srcObject = stream;
      await video.play();

      setCameraState("ready");
      setMessage("Raise your RIGHT hand above your shoulder");

      animationRef.current = requestAnimationFrame(processFrame);
    } catch (error) {
      console.error(error);

      setCameraState("error");

      if (error?.name === "NotAllowedError") {
        setMessage(
          "Camera permission was blocked. Allow camera access and try again."
        );
      } else {
        setMessage(
          "Could not start the camera. Check the browser permission and try again."
        );
      }
    }
  };

  const stopCamera = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    animationRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (poseRef.current) {
      poseRef.current.close();
      poseRef.current = null;
    }

    setPoseReady(false);
    setMovementDetected(false);
    setScore(0);
    setCameraState("idle");
    setMessage("Press Enable Camera to begin");

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  if (screen === "training") {
    return (
      <main className="training-page">
        <div className="training-card active-training">
          <div className="topbar">
            <div>
              <p className="eyebrow">MoveSync Training</p>
              <h1>Raise your hand.</h1>
            </div>

            <button
              className="back-button"
              onClick={() => {
                stopCamera();
                setScreen("home");
              }}
            >
              Exit
            </button>
          </div>

          <div className="challenge-layout">
            <section className="camera-panel">
              <video
                ref={videoRef}
                className="camera-video"
                muted
                playsInline
              />

              <canvas ref={canvasRef} className="pose-canvas" />

              <div className="camera-badge">
                <span
                  className={cameraState === "ready" ? "live-dot" : ""}
                ></span>

                {cameraState === "ready" ? "LIVE" : "CAMERA"}
              </div>

              {cameraState === "idle" && (
                <div className="camera-overlay">
                  <div className="camera-icon">◉</div>

                  <h2>Camera check</h2>

                  <p>
                    Your camera stays on this device while MoveSync detects
                    your pose.
                  </p>

                  <button className="start-button" onClick={startCamera}>
                    Enable Camera
                  </button>
                </div>
              )}

              {cameraState === "loading" && (
                <div className="camera-overlay">
                  <div className="loader"></div>

                  <h2>Preparing MoveSync</h2>

                  <p>{message}</p>
                </div>
              )}

              {cameraState === "error" && (
                <div className="camera-overlay">
                  <div className="camera-icon">!</div>

                  <h2>Camera unavailable</h2>

                  <p>{message}</p>

                  <button className="start-button" onClick={startCamera}>
                    Try Again
                  </button>
                </div>
              )}
            </section>

            <aside className="challenge-panel">
              <div>
                <p className="eyebrow">Challenge 01</p>

                <h2>
                  Raise your RIGHT hand above your shoulder.
                </h2>

                <p className="challenge-copy">
                  Keep your body inside the camera frame. The skeleton overlay
                  shows what MoveSync can detect.
                </p>
              </div>

              <div
                className={`result-box ${
                  movementDetected ? "success" : ""
                }`}
              >
                <span className="result-label">STATUS</span>

                <strong>
                  {movementDetected
                    ? "Movement detected"
                    : "Keep going"}
                </strong>

                <p>{message}</p>
              </div>

              <div className="score-row">
                <div>
                  <span>Score</span>
                  <strong>{score}</strong>
                </div>

                <div>
                  <span>Tracker</span>
                  <strong>{poseReady ? "ON" : "OFF"}</strong>
                </div>
              </div>

              {cameraState === "ready" && (
                <button className="stop-button" onClick={stopCamera}>
                  Stop Camera
                </button>
              )}
            </aside>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="app">
      <nav className="navbar">
        <div className="logo">
          Move<span>Sync</span>
        </div>

        <div className="nav-status">
          Movement Training
        </div>
      </nav>

      <section className="hero">
        <div className="hero-content">
          <p className="eyebrow">
            Interactive Movement Training
          </p>

          <h1>
            Move.
            <br />
            Play.
            <br />
            Improve.
          </h1>

          <p className="description">
            MoveSync turns simple movement challenges into an interactive
            training experience using your device camera.
          </p>

          <button
            className="start-button large"
            onClick={() => setScreen("training")}
          >
            Start Training <span>→</span>
          </button>

          <div className="stats">
            <div>
              <strong>01</strong>
              <span>Movement Game</span>
            </div>

            <div>
              <strong>02</strong>
              <span>Real-time Tracking</span>
            </div>

            <div>
              <strong>03</strong>
              <span>Performance Score</span>
            </div>
          </div>
        </div>

        <div className="visual">
          <div className="movement-orb orb-one"></div>
          <div className="movement-orb orb-two"></div>

          <div className="movement-card">
            <span>READY</span>
            <strong>MOVE</strong>
            <small>Camera-based training</small>
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
