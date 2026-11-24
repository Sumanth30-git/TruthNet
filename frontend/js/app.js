const $ = (selector) => document.querySelector(selector);
const setHidden = (element, hidden) => element.classList.toggle("hidden", hidden);

function resultSection(target, title) {
  const section = document.createElement("section");
  section.className = "result-section";
  const heading = document.createElement("h4");
  heading.textContent = title;
  section.append(heading);
  target.append(section);
  return section;
}

function addResultLine(target, label, value) {
  if (value == null || value === "") return;
  const line = document.createElement("p");
  const name = document.createElement("strong");
  name.textContent = `${label}: `;
  line.append(name, document.createTextNode(String(value)));
  target.append(line);
}

function safeHttpUrl(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const parsed = new URL(value);
    if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password) return null;
    return parsed.href;
  } catch { return null; }
}

function addSourceLink(target, title, rawUrl) {
  if (typeof rawUrl !== "string" || !rawUrl) return;
  const safeUrl = safeHttpUrl(rawUrl);
  if (!safeUrl) {
    addResultLine(target, title || "Source URL", rawUrl);
    return;
  }
  const line = document.createElement("p");
  const link = document.createElement("a");
  link.href = safeUrl;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = title || rawUrl;
  line.append(link);
  target.append(line);
}

function evidenceStateLabel(evidence) {
  if (evidence.analysis_status === "unavailable" || evidence.source_content_origin === "unavailable") return "Evidence unavailable";
  if (evidence.stance === "contradicts") return "Evidence contradicts the claim text";
  if (evidence.analysis_status === "insufficient" || evidence.stance === "insufficient") return "Evidence insufficient";
  return null;
}

function renderNewsDetails(target, data) {
  if (Array.isArray(data.claims)) {
    const claimsSection = resultSection(target, `Claims (${data.claims.length})`);
    if (!data.claims.length) addResultLine(claimsSection, "Status", "No usable claims were extracted.");
    data.claims.forEach((claim) => {
      const item = document.createElement("article");
      item.className = "result-item";
      addResultLine(item, claim.claim_id || "Claim", claim.text || "Claim text unavailable.");
      const context = claim.context;
      if (context) {
        addResultLine(item, "Subject", context.subject);
        addResultLine(item, "Event", context.event);
        addResultLine(item, "Location", context.location);
        addResultLine(item, "Time", context.time_reference);
        if (context.numerical_values?.length) {
          addResultLine(item, "Numbers", context.numerical_values.map((entry) => entry.raw_text).filter(Boolean).join(", "));
        }
        if (context.entity_mentions?.length) {
          addResultLine(item, "Entities", context.entity_mentions.map((entry) => `${entry.text} (${entry.entity_type})`).join(", "));
        }
      }
      claimsSection.append(item);
    });
  }

  if (data.search) {
    const search = resultSection(target, `Search (${data.search.status || "not assessed"})`);
    addResultLine(search, "Provider", data.search.provider);
    addResultLine(search, "Message", data.search.message);
    const results = Array.isArray(data.search.results) ? data.search.results : [];
    if (!results.length) addResultLine(search, "Sources", "No search results were available.");
    results.forEach((source) => {
      addSourceLink(search, source.title || "Source", source.url);
      addResultLine(search, "Publisher", source.publisher);
      addResultLine(search, "Published", source.published_at);
    });
  }

  if (Array.isArray(data.evidence)) {
    const evidenceSection = resultSection(target, `Evidence (${data.evidence.length}) · ${data.evidence_analysis_status || "not assessed"}`);
    if (!data.evidence.length) addResultLine(evidenceSection, "Evidence", "No evidence items were available.");
    data.evidence.forEach((evidence) => {
      const item = document.createElement("article");
      item.className = "result-item";
      addResultLine(item, "Claim", evidence.claim_id || evidence.claim || "Unassigned");
      addResultLine(item, "Evidence assessment", evidenceStateLabel(evidence));
      addResultLine(item, "Assessment", evidence.summary);
      addResultLine(item, "Evidence status", evidence.analysis_status);
      addResultLine(item, "Stance", evidence.stance);
      addResultLine(item, "Reason", evidence.reason);
      addSourceLink(item, evidence.source_title || evidence.publisher || evidence.source || "Source", evidence.source_url);
      addResultLine(item, "Publisher", evidence.publisher);
      addResultLine(item, "Published", evidence.published_at);
      addResultLine(item, "Retrieved", evidence.retrieved_at);
      addResultLine(item, "Text origin", evidence.source_content_origin);
      const retrievalFailures = new Set([
        "invalid_url", "unsafe_host", "dns_failed", "timeout", "http_error",
        "redirect_rejected", "redirect_limit", "body_too_large", "unsupported_content_type",
        "empty_content", "extraction_failed", "failed", "budget_exceeded",
      ]);
      const retrieval = evidence.source_fetch_status;
      addResultLine(item, "Retrieval", retrievalFailures.has(retrieval) ? `Failed (${retrieval})` : retrieval);
      addSourceLink(item, "Final source URL", evidence.source_final_url);
      if (typeof evidence.source_snippet === "string" && evidence.source_snippet) {
        const maxDisplayChars = 1200;
        const preview = evidence.source_snippet.slice(0, maxDisplayChars);
        addResultLine(item, "Evidence text", preview + (evidence.source_snippet.length > maxDisplayChars ? " … [display truncated]" : ""));
      }

      const signal = evidence.nli_signal;
      if (signal) {
        addResultLine(item, "NLI availability", signal.status === "failed" ? "NLI unavailable" : signal.status === "not_run" ? "NLI not run" : signal.status);
        addResultLine(item, "NLI relation", `${signal.prediction || "not assessed"} (${signal.status || "unknown"})`);
        addResultLine(item, "NLI model", `${signal.model_id || "unknown"} · ${signal.model_version || "unknown version"}`);
        addResultLine(item, "NLI inference time", signal.inference_time_ms == null ? null : `${signal.inference_time_ms} ms`);
        addResultLine(item, "NLI source text truncated", signal.source_text_truncated == null ? null : (signal.source_text_truncated ? "yes" : "no"));
        if (signal.probabilities) {
          addResultLine(item, "NLI model scores (not truth probabilities)", [
            `contradiction ${signal.probabilities.contradiction ?? "unavailable"}`,
            `entailment ${signal.probabilities.entailment ?? "unavailable"}`,
            `neutral ${signal.probabilities.neutral ?? "unavailable"}`,
          ].join(" · "));
        }
      }

      const quality = evidence.evidence_quality;
      if (quality) {
        addResultLine(item, "Source quality", quality.source_profile?.source_type);
        addResultLine(item, "Source authority category", quality.source_profile?.authority_level);
        addResultLine(item, "Source relationship", quality.source_profile?.source_relationship);
        addResultLine(item, "Authority note", quality.source_profile?.authority_reason);
        addResultLine(item, "Evidence relevance", quality.match?.relevance);
        addResultLine(item, "Entity / event / location / time match", [
          quality.match?.entity_match, quality.match?.event_match,
          quality.match?.location_match, quality.match?.time_match,
        ].filter(Boolean).join(" / "));
        addResultLine(item, "Match note", quality.match?.reason);
        addResultLine(item, "Freshness", quality.freshness);
        addResultLine(item, "Independence", quality.independence);
        const numerical = quality.numerical_check;
        if (numerical) {
          addResultLine(item, "Numerical comparison", numerical.status);
          addResultLine(item, "Compared values", [numerical.claim_raw, numerical.source_raw].filter(Boolean).join(" / "));
          addResultLine(item, "Numerical comparison note", numerical.reason);
        }
        if (quality.evidence_weight != null) addResultLine(item, "Evidence weight (uncalibrated)", quality.evidence_weight);
      }
      evidenceSection.append(item);
    });
  }

  if (Array.isArray(data.evidence_summaries)) {
    const summariesSection = resultSection(target, "Per-claim evidence summary");
    if (!data.evidence_summaries.length) addResultLine(summariesSection, "Summary", "No summaries were available.");
    data.evidence_summaries.forEach((summary) => {
      const item = document.createElement("article");
      item.className = "result-item";
      addResultLine(item, summary.claim_id || "Claim", summary.overall_strength);
      addResultLine(item, "Support", summary.support_strength);
      addResultLine(item, "Contradiction", summary.contradiction_strength);
      addResultLine(item, "Independent support groups", summary.independent_support_groups);
      addResultLine(item, "Independent contradiction groups", summary.independent_contradiction_groups);
      addResultLine(item, "Summary note", summary.reason);
      summariesSection.append(item);
    });
  }
}

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
    note.textContent = "Review the available evidence and limitations. NLI model scores describe text relations; they are not probabilities that a claim is true.";
    target.append(note);
  }
  if (data.signals?.length) {
    const list = document.createElement("ul");
    data.signals.forEach((signal) => {
      const item = document.createElement("li");
      const label = document.createElement("strong");
      label.textContent = signal.model_id || signal.name || "signal";
      const score = signal.raw_score == null ? "" : `, raw model score ${signal.raw_score}`;
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
  if (data.search || data.claims || data.evidence || data.evidence_summaries) renderNewsDetails(target, data);
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
