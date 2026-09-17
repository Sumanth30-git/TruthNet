const $ = (selector) => document.querySelector(selector);
const setHidden = (element, hidden) => element.classList.toggle("hidden", hidden);

function showResult(target, data) {
  target.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = data.verdict || data.error || "request failed";
  const message = document.createElement("p");
  message.textContent = data.message || data.detail || "No message returned.";
  const uncertainty = document.createElement("p");
  const uncertaintyLabel = document.createElement("strong");
  uncertaintyLabel.textContent = "Uncertainty: ";
  uncertainty.append(uncertaintyLabel, document.createTextNode(data.uncertainty || "not assessed"));
  target.append(heading, message, uncertainty);
  if (data.verdict === "inconclusive") {
    const note = document.createElement("p");
    note.textContent = "Model classification signal only. This is not live fact verification and is not a final authentic/fake verdict.";
    target.append(note);
  }
  if (data.signals?.length) {
    const list = document.createElement("ul");
    data.signals.forEach((signal) => {
      const item = document.createElement("li");
      const label = document.createElement("strong");
      label.textContent = signal.model_id || signal.name || "signal";
      const score = signal.raw_score == null ? "" : `, score ${signal.raw_score}`;
      const inferenceTime = signal.inference_time_ms == null
        ? ""
        : `, inference ${signal.inference_time_ms} ms`;
      item.append(label, document.createTextNode(`: ${signal.prediction} (${signal.status}${score}${inferenceTime})`));
      list.append(item);
    });
    target.append(list);
  }
  if (data.limitations?.length) {
    const limitationsHeading = document.createElement("p");
    const limitationsLabel = document.createElement("strong");
    limitationsLabel.textContent = "Limitations";
    limitationsHeading.append(limitationsLabel);
    const limitations = document.createElement("ul");
    data.limitations.forEach((limitation) => {
      const item = document.createElement("li");
      item.textContent = limitation;
      limitations.append(item);
    });
    target.append(limitationsHeading, limitations);
  }
  setHidden(target, false);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw data;
  return data;
}

document.querySelectorAll(".nav-link").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".nav-link, .tab-panel").forEach((element) => element.classList.remove("active"));
  button.classList.add("active"); $("#" + button.dataset.tab).classList.add("active");
}));

async function checkHealth() {
  const status = $("#health-status");
  try {
    const data = await request("/api/health");
    const inference = data.model_inference_enabled ? "inference on" : "inference off";
    status.textContent = `Service online · v${data.version} · ${inference}`;
    status.classList.add("online");
  }
  catch { status.textContent = "Service unavailable"; }
}

$("#news-submit").addEventListener("click", async () => {
  const result = $("#news-result"), text = $("#news-text").value.trim();
  if (!text) return showResult(result, { detail: "Enter news text before analysis." });
  try { showResult(result, await request("/api/news/analyze", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ text }) })); }
  catch (error) { showResult(result, error); }
});

let selectedImage;
function selectImage(file) {
  const status = $("#image-status"), preview = $("#image-preview");
  if (!file) return;
  if (!['image/jpeg','image/png','image/webp'].includes(file.type) || file.size > 10 * 1024 * 1024) { selectedImage = undefined; $("#image-submit").disabled = true; status.textContent = "Choose a JPEG, PNG, or WebP image under 10 MB."; return; }
  selectedImage = file; status.textContent = `${file.name} ready for analysis.`; $("#image-submit").disabled = false; preview.src = URL.createObjectURL(file); setHidden(preview, false);
}
$("#choose-image").addEventListener("click", () => $("#image-input").click());
$("#image-input").addEventListener("change", (event) => selectImage(event.target.files[0]));
const dropZone = $("#drop-zone");
['dragenter','dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); }));
['dragleave','drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", (event) => selectImage(event.dataTransfer.files[0]));
$("#image-submit").addEventListener("click", async () => { const result = $("#image-result"); if (!selectedImage) return; $("#image-status").textContent = "Processing request…"; const form = new FormData(); form.append("image", selectedImage); try { showResult(result, await request("/api/image/analyze", { method:"POST", body:form })); $("#image-status").textContent = "Analysis response received."; } catch (error) { showResult(result, error); $("#image-status").textContent = "Analysis could not be completed."; } });
$("#video-submit").addEventListener("click", async () => { try { showResult($("#video-result"), await request("/api/video/analyze", { method:"POST" })); } catch (error) { showResult($("#video-result"), error); } });
checkHealth();
