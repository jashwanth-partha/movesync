import { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const CHALLENGES = [
  {
    title: "Raise your RIGHT hand",
    short: "Right hand raise",
    instruction: "Raise your right hand above your right shoulder.",
    key: "right-hand",
  },
  {
    title: "Raise your LEFT hand",
    short: "Left hand raise",
    instruction: "Raise your left hand above your left shoulder.",
    key: "left-hand",
  },
  {
    title: "Raise BOTH hands",
    short: "Both hands raise",
    instruction: "Raise both hands above your shoulders.",
    key: "both-hands",
  },
  {
    title: "Move your RIGHT hand",
    short: "Right hand movement",
    instruction: "Move your right hand toward the right side of the screen.",
    key: "right-hand-side",
  },
  {
    title: "Move your LEFT hand",
    short: "Left hand movement",
    instruction: "Move your left hand toward the right side of the screen.",
    key: "left-hand-side",
  },
];

const HOLD_MS = 900;

function calculateScore(seconds) {
  return Math.max(40, Math.round(100 - seconds * 4));
}

function App() {
  const [screen, setScreen] = useState("home");
  const [cameraState, setCameraState] = useState("idle");
  const [message, setMessage] = useState("Press Enable Camera to begin");
  const [movementDetected, setMovementDetected] = useState(false);
  const [poseReady, setPoseReady] = useState(false);
  const [challengeIndex, setChallengeIndex] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [sessionResults, setSessionResults] = useState([]);

  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("movesync-history") || "[]");
    } catch {
      return [];
    }
  });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const poseRef = useRef(null);
  const animationRef = useRef(null);

  const challengeIndexRef = useRef(0);
  const challengeStartRef = useRef(0);
  const holdStartRef = useRef(0);
  const completingRef = useRef(false);
  const movementDetectedRef = useRef(false);

  const challenge = CHALLENGES[challengeIndex];

  const updateMessage = (value) => {
    setMessage(value);
  };

  useEffect(() => {
    if (screen !== "training" || cameraState !== "ready") return;

    const timer = setInterval(() => {
      if (challengeStartRef.current) {
        setElapsed(
          (performance.now() - challengeStartRef.current) / 1000
        );
      }
    }, 100);

    return () => clearInterval(timer);
  }, [screen, cameraState, challengeIndex]);

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

  const stopCamera = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (poseRef.current) {
      poseRef.current.close();
      poseRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setPoseReady(false);
    setMovementDetected(false);
    movementDetectedRef.current = false;
    setCameraState("idle");
    setMessage("Press Enable Camera to begin");
  };

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
      [15, 17],
      [15, 19],
      [15, 21],
      [16, 18],
      [16, 20],
      [16, 22],
    ];

    ctx.strokeStyle = movementDetectedRef.current
      ? "rgba(126, 255, 161, 0.98)"
      : "rgba(255, 255, 255, 0.84)";

    ctx.lineWidth = 3;

    connections.forEach(([a, b]) => {
      if (!points[a] || !points[b]) return;

      ctx.beginPath();
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
      ctx.stroke();
    });

    ctx.fillStyle = movementDetectedRef.current
      ? "#7effa1"
      : "#ffffff";

    points.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  const evaluateMovement = (landmarks, key) => {
    if (!landmarks) return false;

    const rightShoulder = landmarks[12];
    const leftShoulder = landmarks[11];
    const rightWrist = landmarks[16];
    const leftWrist = landmarks[15];

    const visible = (...points) =>
      points.every(
        (point) => point && point.visibility > 0.45
      );

    if (key === "right-hand") {
      return (
        visible(rightShoulder, rightWrist) &&
        rightWrist.y < rightShoulder.y - 0.08
      );
    }

    if (key === "left-hand") {
      return (
        visible(leftShoulder, leftWrist) &&
        leftWrist.y < leftShoulder.y - 0.08
      );
    }

    if (key === "both-hands") {
      return (
        visible(
          leftShoulder,
          rightShoulder,
          leftWrist,
          rightWrist
        ) &&
        leftWrist.y < leftShoulder.y - 0.05 &&
        rightWrist.y < rightShoulder.y - 0.05
      );
    }

    if (key === "right-hand-side") {
      return (
        visible(rightWrist) &&
        rightWrist.x > 0.70
      );
    }

    if (key === "left-hand-side") {
      return (
        visible(leftWrist) &&
        leftWrist.x > 0.70
      );
    }

    return false;
  };

  const finishSession = () => {
    const completed = [...sessionResults];

    const totalScore = completed.length
      ? Math.round(
          completed.reduce((sum, item) => sum + item.score, 0) /
            completed.length
        )
      : 0;

    const session = {
      id: Date.now(),
      date: new Date().toLocaleString(),
      score: totalScore,
      completed: completed.length,
      total: CHALLENGES.length,
      results: completed,
    };

    const previousHistory = (() => {
      try {
        return JSON.parse(localStorage.getItem("movesync-history") || "[]");
      } catch {
        return [];
      }
    })();

    const nextHistory = [session, ...previousHistory].slice(0, 8);

    localStorage.setItem(
      "movesync-history",
      JSON.stringify(nextHistory)
    );

    setHistory(nextHistory);

    stopCamera();
    setScreen("results");
  };

  const completeChallenge = () => {
    if (completingRef.current) return;

    completingRef.current = true;

    const current = CHALLENGES[challengeIndexRef.current];

    const seconds = Math.max(
      0.1,
      (performance.now() - challengeStartRef.current) / 1000
    );

    const result = {
      challenge: current.short,
      time: Number(seconds.toFixed(1)),
      score: calculateScore(seconds),
    };

    const nextResults = [...sessionResults, result];

    setSessionResults(nextResults);

    updateMessage("Nice! Challenge complete ✓");

    setTimeout(() => {
      const nextIndex = challengeIndexRef.current + 1;

      if (nextIndex >= CHALLENGES.length) {
        setSessionResults(nextResults);

        const totalScore = Math.round(
          nextResults.reduce((sum, item) => sum + item.score, 0) /
            nextResults.length
        );

        const session = {
          id: Date.now(),
          date: new Date().toLocaleString(),
          score: totalScore,
          completed: nextResults.length,
          total: CHALLENGES.length,
          results: nextResults,
        };

        const previousHistory = (() => {
          try {
            return JSON.parse(
              localStorage.getItem("movesync-history") || "[]"
            );
          } catch {
            return [];
          }
        })();

        const nextHistory = [session, ...previousHistory].slice(0, 8);

        localStorage.setItem(
          "movesync-history",
          JSON.stringify(nextHistory)
        );

        setHistory(nextHistory);
        stopCamera();
        setScreen("results");
        completingRef.current = false;
        return;
      }

      challengeIndexRef.current = nextIndex;
      challengeStartRef.current = performance.now();
      holdStartRef.current = 0;
      movementDetectedRef.current = false;

      setChallengeIndex(nextIndex);
      setElapsed(0);
      setMovementDetected(false);

      updateMessage(CHALLENGES[nextIndex].instruction);

      completingRef.current = false;
    }, 900);
  };

  const processFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const pose = poseRef.current;

    if (!video || !canvas || !pose || video.readyState < 2) {
      animationRef.current = requestAnimationFrame(processFrame);
      return;
    }

    if (video.currentTime !== Number(video.dataset.lastTime || -1)) {
      video.dataset.lastTime = String(video.currentTime);

      const result = pose.detectForVideo(video, performance.now());

      const ctx = canvas.getContext("2d");

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const landmarks = result.landmarks?.[0];

      drawPose(landmarks, canvas, ctx);

      if (landmarks && !completingRef.current) {
        const currentChallenge =
          CHALLENGES[challengeIndexRef.current];

        const detected = evaluateMovement(
          landmarks,
          currentChallenge.key
        );

        movementDetectedRef.current = detected;
        setMovementDetected(detected);

        if (detected) {
          if (!holdStartRef.current) {
            holdStartRef.current = performance.now();
          }

          const heldFor =
            performance.now() - holdStartRef.current;

          updateMessage("Hold it... ✓");

          if (heldFor >= HOLD_MS) {
            completeChallenge();
          }
        } else {
          holdStartRef.current = 0;
          updateMessage(currentChallenge.instruction);
        }
      } else if (!landmarks) {
        movementDetectedRef.current = false;
        setMovementDetected(false);
        holdStartRef.current = 0;
        updateMessage("Step into the camera frame");
      }
    }

    animationRef.current = requestAnimationFrame(processFrame);
  };

  const startCamera = async () => {
    try {
      setCameraState("loading");
      updateMessage("Loading camera and movement tracker...");

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

      challengeIndexRef.current = 0;
      challengeStartRef.current = performance.now();
      holdStartRef.current = 0;
      completingRef.current = false;

      setChallengeIndex(0);
      setSessionResults([]);
      setElapsed(0);
      setMovementDetected(false);
      setPoseReady(true);

      const video = videoRef.current;

      video.srcObject = stream;
      await video.play();

      setCameraState("ready");
      updateMessage(CHALLENGES[0].instruction);

      animationRef.current = requestAnimationFrame(processFrame);
    } catch (error) {
      console.error(error);

      setCameraState("error");

      if (error?.name === "NotAllowedError") {
        updateMessage(
          "Camera permission was blocked. Allow camera access and try again."
        );
      } else {
        updateMessage(
          "Could not start the camera. Check the browser permission and try again."
        );
      }
    }
  };

  const openTraining = () => {
    setChallengeIndex(0);
    setSessionResults([]);
    setScreen("training");
  };

  if (screen === "results") {
    const averageScore = sessionResults.length
      ? Math.round(
          sessionResults.reduce((sum, item) => sum + item.score, 0) /
            sessionResults.length
        )
      : 0;

    const averageTime = sessionResults.length
      ? (
          sessionResults.reduce((sum, item) => sum + item.time, 0) /
          sessionResults.length
        ).toFixed(1)
      : "0.0";

    return (
      <main className="training-page results-page">
        <div className="results-card">
          <div className="results-header">
            <div>
              <p className="eyebrow">Session complete</p>
              <h1>Great work.</h1>
              <p className="results-subtitle">
                Your movement session has been recorded on this device.
              </p>
            </div>

            <button
              className="back-button"
              onClick={() => setScreen("home")}
            >
              Home
            </button>
          </div>

          <div className="score-hero">
            <div>
              <span>SESSION SCORE</span>
              <strong>{averageScore}</strong>
              <small>/ 100</small>
            </div>

            <div className="score-ring">
              <b>{sessionResults.length}</b>
              <span>of {CHALLENGES.length}</span>
            </div>
          </div>

          <div className="result-grid">
            <div className="metric-card">
              <span>Average score</span>
              <strong>{averageScore}%</strong>
            </div>

            <div className="metric-card">
              <span>Average time</span>
              <strong>{averageTime}s</strong>
            </div>

            <div className="metric-card">
              <span>Completed</span>
              <strong>{sessionResults.length}</strong>
            </div>
          </div>

          <section className="challenge-results">
            <div className="section-heading">
              <span>Challenge breakdown</span>
              <small>Latest session</small>
            </div>

            {sessionResults.map((result, index) => (
              <div
                className="challenge-result-row"
                key={`${result.challenge}-${index}`}
              >
                <div>
                  <span>
                    {String(index + 1).padStart(2, "0")}
                  </span>

                  <strong>{result.challenge}</strong>
                </div>

                <div>
                  <small>{result.time}s</small>
                  <b>{result.score}</b>
                </div>
              </div>
            ))}
          </section>

          <button className="start-button large" onClick={openTraining}>
            Train Again <span>→</span>
          </button>
        </div>
      </main>
    );
  }

  if (screen === "training") {
    return (
      <main className="training-page">
        <div className="training-card active-training">
          <div className="topbar">
            <div>
              <p className="eyebrow">MoveSync Training</p>
              <h1>{challenge.title}</h1>
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
                  className={
                    cameraState === "ready" ? "live-dot" : ""
                  }
                />

                {cameraState === "ready" ? "LIVE" : "CAMERA"}
              </div>

              {cameraState === "idle" && (
                <div className="camera-overlay">
                  <div className="camera-icon">◉</div>

                  <h2>Camera check</h2>

                  <p>
                    Your camera stays on this device while MoveSync
                    detects your hand movements.
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
                <div className="challenge-progress">
                  <span>
                    Challenge{" "}
                    {String(challengeIndex + 1).padStart(2, "0")}
                  </span>

                  <span>{CHALLENGES.length} total</span>
                </div>

                <div className="progress-bar">
                  <div
                    style={{
                      width: `${
                        ((challengeIndex + 1) / CHALLENGES.length) * 100
                      }%`,
                    }}
                  ></div>
                </div>

                <h2>{challenge.instruction}</h2>

                <p className="challenge-copy">
                  Hold the movement steady for a moment. The skeleton overlay
                  shows what MoveSync can see.
                </p>
              </div>

              <div>
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
                    <span>Time</span>
                    <strong>{elapsed.toFixed(1)}s</strong>
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
              </div>
            </aside>
          </div>
        </div>
      </main>
    );
  }

  const bestScore = history.length
    ? Math.max(...history.map((item) => item.score))
    : 0;

  return (
    <main className="app">
      <nav className="navbar">
        <div className="logo">
          Move<span>Sync</span>
        </div>

        <div className="nav-status">Hand Coordination Training</div>
      </nav>

      <section className="hero">
        <div className="hero-content">
          <p className="eyebrow">Interactive Hand Coordination</p>

          <h1>
            Move.
            <br />
            Play.
            <br />
            Improve.
          </h1>

          <p className="description">
            MoveSync turns simple hand coordination challenges into an
            interactive training experience using your device camera.
          </p>

          <button className="start-button large" onClick={openTraining}>
            Start Training <span>→</span>
          </button>

          <div className="stats">
            <div>
              <strong>{CHALLENGES.length}</strong>
              <span>Hand Challenges</span>
            </div>

            <div>
              <strong>Live</strong>
              <span>Movement Tracking</span>
            </div>

            <div>
              <strong>{bestScore || "—"}</strong>
              <span>Best Session Score</span>
            </div>
          </div>
        </div>

        <div className="visual">
          <div className="movement-orb orb-one"></div>
          <div className="movement-orb orb-two"></div>

          <div className="movement-card">
            <span>READY</span>
            <strong>MOVE</strong>
            <small>Hand coordination training</small>
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
