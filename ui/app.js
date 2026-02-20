// ============================================
// RAG_AI — Frontend Application
// ============================================

const API_BASE = "";

// ============ DOM Elements ============
const navBtns = document.querySelectorAll(".nav-btn");
const views = document.querySelectorAll(".view");
const chatContainer = document.getElementById("chatContainer");
const welcomeMessage = document.getElementById("welcomeMessage");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const docTypeSelect = document.getElementById("docTypeSelect");
const sourceFilter = document.getElementById("sourceFilter");
const llmProviderSelect = document.getElementById("llmProviderSelect");
const llmModelSelect = document.getElementById("llmModelSelect");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadProgress = document.getElementById("uploadProgress");
const progressFileName = document.getElementById("progressFileName");
const progressBar = document.getElementById("progressBar");
const progressStatus = document.getElementById("progressStatus");
const uploadResult = document.getElementById("uploadResult");
const historyList = document.getElementById("historyList");
const documentsGrid = document.getElementById("documentsGrid");
const refreshDocsBtn = document.getElementById("refreshDocsBtn");
const docCount = document.getElementById("docCount");


// ============ Navigation ============
navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        const targetView = btn.dataset.view;

        navBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        views.forEach(v => v.classList.remove("active"));
        document.getElementById(`view-${targetView}`).classList.add("active");

        // Load documents when switching to that tab
        if (targetView === "documents") {
            loadDocuments();
        }
        if (targetView === "settings") {
            loadProviderStatus();
        }
    });
});


// ============ Chat — Send Question ============
sendBtn.addEventListener("click", sendQuestion);
questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

// Auto-resize textarea
questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = questionInput.scrollHeight + "px";
});

async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    // Hide welcome
    if (welcomeMessage) welcomeMessage.style.display = "none";

    // Add user message
    addMessage("user", question);
    questionInput.value = "";
    questionInput.style.height = "auto";

    // Show thinking
    const thinkingEl = addThinking();
    sendBtn.disabled = true;

    try {
        const body = { question };
        const docType = docTypeSelect.value;
        if (docType) body.doc_type = docType;
        const source = sourceFilter.value;
        if (source) body.source_filter = source;
        const provider = llmProviderSelect.value;
        if (provider) body.provider = provider;
        const model = llmModelSelect.value;
        if (model) body.model_name = model;

        const res = await fetch(`${API_BASE}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        const data = await res.json();

        thinkingEl.remove();

        if (!res.ok) {
            addMessage("ai", `⚠️ Error: ${data.detail || "Something went wrong"}`, []);
            return;
        }

        const providerTag = data.provider_used ? ` [${data.provider_used}]` : "";
        addMessage("ai", data.answer, data.sources || [], providerTag);

    } catch (err) {
        thinkingEl.remove();
        addMessage("ai", `⚠️ Connection error: ${err.message}. Make sure the server is running.`, []);
    } finally {
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

function addMessage(role, text, sources = [], providerTag = "") {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message message-${role}`;

    if (role === "user") {
        msgDiv.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
    } else {
        let sourcesHtml = "";
        if (sources.length > 0) {
            const chips = sources.map(s => {
                const preview = escapeHtml(s.text || "");
                return `<span class="source-chip" title="${preview}">${preview.slice(0, 80)}${preview.length > 80 ? '...' : ''}</span>`;
            }).join("");
            sourcesHtml = `
                <div class="sources-section">
                    <div class="sources-title">📎 Sources (${sources.length})</div>
                    ${chips}
                </div>
            `;
        }

        const providerHtml = providerTag
            ? `<div class="provider-tag">${escapeHtml(providerTag)}</div>`
            : "";

        msgDiv.innerHTML = `
            <div class="ai-avatar">🧠</div>
            <div class="message-bubble">
                ${formatAnswer(text)}
                ${sourcesHtml}
                ${providerHtml}
            </div>
        `;
    }

    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addThinking() {
    const div = document.createElement("div");
    div.className = "thinking";
    div.innerHTML = `
        <div class="ai-avatar">🧠</div>
        <div class="thinking-dots">
            <span></span><span></span><span></span>
        </div>
    `;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return div;
}

function formatAnswer(text) {
    // Basic markdown: bold, newlines
    return text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}


// ============ Upload — Drag & Drop ============
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) uploadFile(e.target.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
});

async function uploadFile(file) {
    // Show progress
    uploadProgress.style.display = "block";
    uploadResult.style.display = "none";
    progressFileName.textContent = `📄 ${file.name}`;
    progressBar.style.width = "0%";
    progressStatus.textContent = "Uploading & ingesting...";

    // Animate progress bar (fake progress since we can't track server-side)
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressBar.style.width = progress + "%";
    }, 500);

    try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });

        clearInterval(interval);
        progressBar.style.width = "100%";

        const data = await res.json();

        if (res.ok) {
            progressStatus.textContent = "✅ Complete!";
            uploadResult.style.display = "block";
            uploadResult.className = "upload-result success";
            uploadResult.textContent = `✅ ${data.message}`;
            addToHistory(file.name, true);
            loadSourceFilter();  // Refresh the source dropdown
        } else {
            progressStatus.textContent = "❌ Failed";
            uploadResult.style.display = "block";
            uploadResult.className = "upload-result error";
            uploadResult.textContent = `❌ Error: ${data.detail || "Upload failed"}`;
            addToHistory(file.name, false);
        }

    } catch (err) {
        clearInterval(interval);
        progressBar.style.width = "100%";
        progressStatus.textContent = "❌ Failed";
        uploadResult.style.display = "block";
        uploadResult.className = "upload-result error";
        uploadResult.textContent = `❌ Connection error: ${err.message}`;
        addToHistory(file.name, false);
    }

    fileInput.value = "";
}

const uploadHistoryItems = [];

function addToHistory(filename, success) {
    uploadHistoryItems.unshift({ filename, success, time: new Date() });
    renderHistory();
}

function renderHistory() {
    if (uploadHistoryItems.length === 0) {
        historyList.innerHTML = '<p class="empty-state">No files uploaded yet</p>';
        return;
    }

    historyList.innerHTML = uploadHistoryItems.map(item => {
        const icon = getFileIcon(item.filename);
        const status = item.success ? "✅ Ingested" : "❌ Failed";
        const statusClass = item.success ? "file-status" : "file-status error";
        return `
            <div class="history-item">
                <span class="file-icon">${icon}</span>
                <div class="file-info">
                    <div class="file-name">${escapeHtml(item.filename)}</div>
                    <div class="${statusClass}">${status}</div>
                </div>
            </div>
        `;
    }).join("");
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        pdf: "📄", html: "🌐", htm: "🌐", json: "📋",
        eml: "📧", png: "🖼️", jpg: "🖼️", jpeg: "🖼️",
        wav: "🎵", mp3: "🎵", txt: "📝"
    };
    return icons[ext] || "📁";
}


// ============ Documents View ============
refreshDocsBtn.addEventListener("click", loadDocuments);

async function loadDocuments() {
    documentsGrid.innerHTML = '<div class="loading-spinner">Loading documents...</div>';
    docCount.textContent = "Loading...";

    try {
        const res = await fetch(`${API_BASE}/documents`);
        const data = await res.json();

        const docs = data.documents || [];
        docCount.textContent = `${docs.length} document${docs.length !== 1 ? 's' : ''}`;

        if (docs.length === 0) {
            documentsGrid.innerHTML = '<div class="loading-spinner">No documents ingested yet. Upload a file to get started!</div>';
            return;
        }

        documentsGrid.innerHTML = docs.map(doc => {
            const icon = getFileIcon(doc.source || "");
            const docType = (doc.doc_type || doc.type || "unknown").toUpperCase();
            const source = doc.source || "Unknown source";
            const fileName = source.split('/').pop();

            // Build meta items from available fields
            let metaHtml = "";
            if (doc.source) {
                metaHtml += `<div class="doc-meta-item">📂 ${escapeHtml(fileName)}</div>`;
            }
            if (doc.doc_type || doc.type) {
                metaHtml += `<div class="doc-meta-item">🏷️ Type: ${escapeHtml(doc.doc_type || doc.type)}</div>`;
            }
            if (doc.chunk_count) {
                metaHtml += `<div class="doc-meta-item">🧩 ${doc.chunk_count} chunks</div>`;
            }
            if (doc.entities) {
                metaHtml += `<div class="doc-meta-item">🔎 Entities: ${escapeHtml(doc.entities)}</div>`;
            }

            return `
                <div class="doc-card" data-source="${escapeHtml(source)}">
                    <div class="doc-card-header">
                        <span style="font-size:24px">${icon}</span>
                        <span class="doc-type-badge">${escapeHtml(docType)}</span>
                        <button class="delete-doc-btn" title="Delete this document" onclick="deleteDocument('${escapeHtml(source).replace(/'/g, "\\'")}')">🗑️</button>
                    </div>
                    <div class="doc-card-title">${escapeHtml(fileName)}</div>
                    <div class="doc-card-meta">${metaHtml}</div>
                </div>
            `;
        }).join("");

    } catch (err) {
        documentsGrid.innerHTML = `<div class="loading-spinner">⚠️ Failed to load documents: ${escapeHtml(err.message)}</div>`;
        docCount.textContent = "Error";
    }
}


// ============ Delete Document ============
async function deleteDocument(source) {
    if (!confirm(`Are you sure you want to delete "${source}" and all its chunks?\n\nThis cannot be undone.`)) {
        return;
    }

    // Find the card and show deleting state
    const card = document.querySelector(`.doc-card[data-source="${source}"]`);
    if (card) {
        card.style.opacity = "0.5";
        card.style.pointerEvents = "none";
    }

    try {
        const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(source)}`, {
            method: "DELETE",
        });

        const data = await res.json();

        if (res.ok) {
            // Remove card with animation
            if (card) {
                card.style.transition = "all 0.3s ease";
                card.style.transform = "scale(0.9)";
                card.style.opacity = "0";
                setTimeout(() => {
                    card.remove();
                    // Update count
                    const remaining = documentsGrid.querySelectorAll(".doc-card").length;
                    docCount.textContent = `${remaining} document${remaining !== 1 ? 's' : ''}`;
                    if (remaining === 0) {
                        documentsGrid.innerHTML = '<div class="loading-spinner">No documents ingested yet. Upload a file to get started!</div>';
                    }
                }, 300);
            }
            // Refresh source filter dropdown
            loadSourceFilter();
        } else {
            alert(`Failed to delete: ${data.detail || "Unknown error"}`);
            if (card) {
                card.style.opacity = "1";
                card.style.pointerEvents = "auto";
            }
        }
    } catch (err) {
        alert(`Connection error: ${err.message}`);
        if (card) {
            card.style.opacity = "1";
            card.style.pointerEvents = "auto";
        }
    }
}


// ============ Source Filter Dropdown ============
async function loadSourceFilter() {
    try {
        const res = await fetch(`${API_BASE}/documents`);
        if (!res.ok) return;

        const data = await res.json();
        const docs = data.documents || [];

        // Get unique source names
        const sources = [...new Set(docs.map(d => d.source).filter(Boolean))];

        // Keep the current selection
        const currentValue = sourceFilter.value;

        // Rebuild options
        sourceFilter.innerHTML = '<option value="">📂 All Documents</option>';
        sources.forEach(src => {
            const opt = document.createElement("option");
            opt.value = src;
            opt.textContent = `📄 ${src}`;
            sourceFilter.appendChild(opt);
        });

        // Restore selection if still valid
        if (currentValue && sources.includes(currentValue)) {
            sourceFilter.value = currentValue;
        }
    } catch (e) {
        // Silently fail — dropdown stays as "All Documents"
    }
}

// Load sources on page load
loadSourceFilter();


// ============ LLM Provider / Model Selection ============
let modelsData = {};

llmProviderSelect.addEventListener("change", () => {
    const provider = llmProviderSelect.value;
    if (provider && modelsData[provider]) {
        llmModelSelect.innerHTML = '<option value="">Default</option>';
        modelsData[provider].forEach(m => {
            const opt = document.createElement("option");
            opt.value = m;
            opt.textContent = m;
            llmModelSelect.appendChild(opt);
        });
        llmModelSelect.style.display = "inline-block";
    } else {
        llmModelSelect.style.display = "none";
        llmModelSelect.innerHTML = "";
    }
});

async function loadModels() {
    try {
        const res = await fetch(`${API_BASE}/models`);
        if (!res.ok) return;
        const data = await res.json();
        modelsData = {};
        (data.providers || []).forEach(p => {
            modelsData[p.name] = p.models || [];
        });
    } catch (e) {
        // Silently fail
    }
}

loadModels();


// ============ Settings View ============
const saveGroqKeyBtn = document.getElementById("saveGroqKeyBtn");
const groqKeyInput = document.getElementById("groqApiKeyInput");
const settingsResult = document.getElementById("settingsResult");

if (saveGroqKeyBtn) {
    saveGroqKeyBtn.addEventListener("click", async () => {
        const key = groqKeyInput.value.trim();
        if (!key) {
            settingsResult.textContent = "⚠️ Please enter an API key.";
            settingsResult.className = "settings-result error";
            settingsResult.style.display = "block";
            return;
        }

        saveGroqKeyBtn.disabled = true;
        saveGroqKeyBtn.textContent = "Saving...";

        try {
            const res = await fetch(`${API_BASE}/settings`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ groq_api_key: key }),
            });
            const data = await res.json();

            if (res.ok) {
                settingsResult.textContent = "✅ " + (data.message || "API key saved!");
                settingsResult.className = "settings-result success";
                groqKeyInput.value = "";
                loadProviderStatus();
                loadModels();
            } else {
                settingsResult.textContent = "❌ " + (data.detail || "Failed to save.");
                settingsResult.className = "settings-result error";
            }
        } catch (err) {
            settingsResult.textContent = "❌ Connection error: " + err.message;
            settingsResult.className = "settings-result error";
        } finally {
            settingsResult.style.display = "block";
            saveGroqKeyBtn.disabled = false;
            saveGroqKeyBtn.textContent = "💾 Save API Key";
        }
    });
}

async function loadProviderStatus() {
    const statusEl = document.getElementById("providerStatus");
    const modelsListEl = document.getElementById("modelsList");
    if (!statusEl) return;

    try {
        const res = await fetch(`${API_BASE}/models`);
        if (!res.ok) return;
        const data = await res.json();
        const providers = data.providers || [];

        statusEl.innerHTML = providers.map(p => {
            const icon = p.name === "groq" ? "☁️" : "🏠";
            const status = p.available
                ? '<span style="color: #4caf50;">● Available</span>'
                : '<span style="color: #f44336;">● Unavailable</span>';
            return `<div class="provider-status-item">${icon} <strong>${p.name}</strong> — ${status}</div>`;
        }).join("");

        if (modelsListEl) {
            modelsListEl.innerHTML = providers.map(p => {
                const models = (p.models || []).map(m => `<span class="model-chip">${m}</span>`).join(" ");
                return `<div class="models-section"><strong>${p.name}:</strong> ${models}</div>`;
            }).join("");
        }
    } catch (e) {
        statusEl.innerHTML = '<div class="provider-status-item">⚠️ Could not load provider info</div>';
    }
}

// Settings info is loaded via the nav handler above.
