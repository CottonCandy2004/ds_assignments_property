const STAGES = ["auth", "console", "predict"];

function inferDefaultBase() {
  const { protocol, hostname, port } = window.location;
  if (port === "9000") {
    return `${protocol}//${hostname}:8000`;
  }
  if (!port && (hostname === "localhost" || hostname === "127.0.0.1")) {
    return `${protocol}//${hostname}:5000`;
  }
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return `${protocol}//${hostname}:${port === "80" ? "5000" : port}`;
  }
  return "http://localhost:5000";
}

const configuredBase = window.PropertyClientConfig?.apiBase;

const state = {
  baseUrl: configuredBase || inferDefaultBase(),
  token: null,
  user: null,
  stage: "auth",
  progress: "auth",
  lastResult: null,
};

const els = {
  pingHealth: document.getElementById("pingHealth"),
  healthStatus: document.getElementById("healthStatus"),
  modelSummary: document.getElementById("modelSummary"),
  registerForm: document.getElementById("registerForm"),
  loginForm: document.getElementById("loginForm"),
  tokenPreview: document.getElementById("tokenPreview"),
  copyToken: document.getElementById("copyToken"),
  predictForm: document.getElementById("predictForm"),
  roundResult: document.getElementById("roundResult"),
  views: document.querySelectorAll("[data-view]"),
  journeySteps: document.querySelectorAll("[data-stage-indicator]"),
  logoutBtn: document.getElementById("logoutBtn"),
  logoutInline: document.getElementById("logoutInline"),
  userWelcome: document.getElementById("userWelcome"),
  consoleUser: document.getElementById("consoleUser"),
  consoleMessage: document.getElementById("consoleMessage"),
  resultValue: document.getElementById("resultValue"),
  resultMeta: document.getElementById("resultMeta"),
};

function setHealthStatus(text, tone = "neutral") {
  els.healthStatus.textContent = text;
  els.healthStatus.dataset.tone = tone;
}

function setToken(token) {
  state.token = token || null;
  if (token) {
    els.tokenPreview.textContent = `${token.slice(0, 28)}…`;
    els.copyToken.disabled = false;
  } else {
    els.tokenPreview.textContent = "尚未登录";
    els.copyToken.disabled = true;
  }
}

function setUser(user) {
  state.user = user || null;
  const name = user?.username || "--";
  els.userWelcome.textContent = user ? `欢迎，${name}` : "尚未登录";
  els.consoleUser.textContent = name;
}

function setConsoleMessage(message, tone = "info") {
  if (!els.consoleMessage) return;
  els.consoleMessage.textContent = message;
  els.consoleMessage.dataset.tone = tone;
}

function updateResultCard(result) {
  if (!result) {
    els.resultValue.textContent = "尚无结果";
    els.resultMeta.textContent = "填写表单后即可看到预估价格。";
    state.lastResult = null;
    return;
  }
  const { prediction, currency, overrides } = result;
  els.resultValue.textContent = `${prediction} ${currency}`;
  const overrideKeys = Object.keys(overrides || {});
  els.resultMeta.textContent = overrideKeys.length
    ? `输入特征：${overrideKeys.join(", ")}`
    : "使用默认特征完成计算。";
  state.lastResult = result;
}

function updateJourney() {
  const progressIndex = STAGES.indexOf(state.progress);
  const currentStep = state.stage === "auth" ? "auth" : state.progress;
  els.journeySteps.forEach((step) => {
    const value = step.dataset.stageIndicator;
    const idx = STAGES.indexOf(value);
    step.classList.toggle("active", idx <= progressIndex);
    step.classList.toggle("current", value === currentStep);
  });
}

function goToStage(stage) {
  state.stage = stage;
  if (stage === "auth") {
    state.progress = "auth";
  } else if (state.progress === "auth") {
    state.progress = "console";
  }
  els.views.forEach((view) => {
    view.classList.toggle("active", view.dataset.view === stage);
  });
  const disabled = stage === "auth";
  els.logoutBtn.disabled = disabled;
  els.logoutInline.disabled = disabled;
  updateJourney();
}

function markProgress(step) {
  state.progress = step;
  updateJourney();
}

function logout() {
  setToken(null);
  setUser(null);
  state.progress = "auth";
  goToStage("auth");
  updateResultCard(null);
  setConsoleMessage("已退出登录", "info");
}

async function apiRequest(path, { method = "GET", data, auth = false } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (!data) delete headers["Content-Type"];
  if (auth && state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }
  const response = await fetch(`${state.baseUrl}${path}`, {
    method,
    headers,
    body: data ? JSON.stringify(data) : undefined,
  });
  const contentType = response.headers.get("content-type") || "";
  const payload =
    contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const error = new Error("API request failed");
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function parseCustomFeatures(text) {
  const features = {};
  text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .forEach((line) => {
      const [key, ...rest] = line.split("=");
      if (key && rest.length) {
        features[key.trim()] = rest.join("=").trim();
      }
    });
  return features;
}

function buildPredictQuery(formData) {
  const params = new URLSearchParams();
  ["Rooms", "Bathroom", "Car", "Distance"].forEach((key) => {
    const value = formData.get(key);
    if (value) params.set(key, value);
  });
  const custom = parseCustomFeatures(formData.get("custom") || "");
  Object.entries(custom).forEach(([key, value]) => params.append("feature", `${key}=${value}`));
  return params.toString() ? `?${params.toString()}` : "";
}

els.copyToken.addEventListener("click", async () => {
  if (!state.token) return;
  await navigator.clipboard.writeText(state.token);
  els.copyToken.textContent = "已复制";
  setTimeout(() => (els.copyToken.textContent = "复制 Bearer Token"), 1500);
});

els.logoutBtn.addEventListener("click", logout);
els.logoutInline.addEventListener("click", logout);

els.pingHealth.addEventListener("click", async () => {
  setHealthStatus("检测中…", "info");
  try {
    const data = await apiRequest("/health");
    setHealthStatus("正常", "success");
    els.modelSummary.textContent = `${data.feature_count} features`;
    setConsoleMessage("服务健康检查通过", "success");
  } catch (error) {
    setHealthStatus(`异常 (${error.status || "网络错误"})`, "error");
    setConsoleMessage("健康检查失败，请稍后再试", "error");
  }
});

els.registerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  const payload = {
    username: formData.get("username")?.trim(),
    password: formData.get("password"),
  };
  try {
    const data = await apiRequest("/auth/register", { method: "POST", data: payload });
    setToken(data.token);
    setUser(data.user);
    goToStage("console");
    markProgress("console");
    setConsoleMessage("注册并登录成功", "success");
  } catch (error) {
    const message = error.payload?.error || error.payload?.message || "注册失败";
    setConsoleMessage(message, "error");
  }
});

els.loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  const payload = {
    username: formData.get("username")?.trim(),
    password: formData.get("password"),
  };
  try {
    const data = await apiRequest("/auth/login", { method: "POST", data: payload });
    setToken(data.token);
    setUser(data.user);
    goToStage("console");
    markProgress("console");
    setConsoleMessage("登录成功", "success");
  } catch (error) {
    const message = error.payload?.error || error.payload?.message || "登录失败";
    setConsoleMessage(message, "error");
  }
});

els.predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  const query = buildPredictQuery(formData);
  const round = els.roundResult.checked;
  try {
    const data = await apiRequest(`/predict${query}`);
    if (round) {
      data.prediction = Number(data.prediction).toFixed(2);
    }
    markProgress("predict");
    updateResultCard(data);
    setConsoleMessage("预测完成", "success");
  } catch (error) {
    const message = error.payload?.error || error.payload?.message || "预测失败";
    setConsoleMessage(message, "error");
    els.resultMeta.textContent = message;
    els.resultValue.textContent = "--";
  }
});

// Initial render
setToken(null);
setUser(null);
setConsoleMessage("请先登录以开始计算", "info");
updateResultCard(null);
goToStage("auth");
