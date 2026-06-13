// Frontend state variables
let activeDocId = null;
let activeDocType = 'pdf';
let activePageNum = 1;
let currentTab = 'image'; // 'image' or 'text'
let documents = [];

// DOM Elements
const themeToggle = document.getElementById('theme-toggle');
const statusOllama = document.getElementById('status-ollama');
const statusXinference = document.getElementById('status-xinference');

const settingsProvider = document.getElementById('settings-provider');
const settingsModel = document.getElementById('settings-model');
const settingsBase = document.getElementById('settings-base');
const settingsKey = document.getElementById('settings-key');
const settingsUseVlm = document.getElementById('settings-use-vlm');
const settingsVlmProvider = document.getElementById('settings-vlm-provider');
const settingsVlmModel = document.getElementById('settings-vlm-model');
const settingsVlmBase = document.getElementById('settings-vlm-base');
const settingsVlmKey = document.getElementById('settings-vlm-key');
const saveChatBtn = document.getElementById('save-chat-btn');
const saveVlmBtn = document.getElementById('save-vlm-btn');
const vlmModelContainer = document.getElementById('vlm-model-container');
const apiKeyContainer = document.getElementById('api-key-container');
const apiBaseContainer = document.getElementById('api-base-container');
const vlmApiKeyContainer = document.getElementById('vlm-api-key-container');
const vlmApiBaseContainer = document.getElementById('vlm-api-base-container');

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const docList = document.getElementById('doc-list');
const uploadProgressContainer = document.getElementById('upload-progress-container');
const uploadFilename = document.getElementById('upload-filename');
const uploadPercent = document.getElementById('upload-percent');
const uploadProgressBar = document.getElementById('upload-progress-bar');

const activeDocIcon = document.getElementById('active-doc-icon');
const activeDocTitle = document.getElementById('active-doc-title');
const activeDocPages = document.getElementById('active-doc-pages');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatFallback = document.getElementById('chat-fallback');
const chatForceSearch = document.getElementById('chat-force-search');

const treeContainer = document.getElementById('tree-container');
const tabImageBtn = document.getElementById('tab-image-btn');
const tabTextBtn = document.getElementById('tab-text-btn');
const tabImageContent = document.getElementById('tab-image-content');
const tabTextContent = document.getElementById('tab-text-content');
const pageImageDisplay = document.getElementById('page-image-display');
const pageTextDisplay = document.getElementById('page-text-display');
const imageViewerPlaceholder = document.getElementById('image-viewer-placeholder');
const textViewerPlaceholder = document.getElementById('text-viewer-placeholder');

// Configure Marked options
marked.setOptions({
  breaks: true,
  gfm: true
});

// Init functions
document.addEventListener('DOMContentLoaded', () => {
  setupTheme();
  setupSettingsUI();
  setupUploadUI();
  setupTabsUI();
  setupChatUI();
  
  // Initial Status and Library fetch
  fetchStatus();
  fetchLibrary();
  
  // Poll connection status every 15s
  setInterval(fetchStatus, 15000);
});

// ── Theme Switcher (Dark / Light) ───────────────────────────────────────────
function setupTheme() {
  const isDark = localStorage.getItem('theme') !== 'light';
  if (isDark) {
    document.documentElement.classList.add('dark');
    themeToggle.innerHTML = '<span class="material-symbols-outlined">light_mode</span>';
  } else {
    document.documentElement.classList.remove('dark');
    themeToggle.innerHTML = '<span class="material-symbols-outlined">dark_mode</span>';
  }
  
  themeToggle.addEventListener('click', () => {
    const isCurrentlyDark = document.documentElement.classList.contains('dark');
    if (isCurrentlyDark) {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
      themeToggle.innerHTML = '<span class="material-symbols-outlined">dark_mode</span>';
    } else {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
      themeToggle.innerHTML = '<span class="material-symbols-outlined">light_mode</span>';
    }
  });
}

// ── Provider Settings ────────────────────────────────────────────────────────
function setupSettingsUI() {
  settingsProvider.addEventListener('change', () => {
    const provider = settingsProvider.value;
    if (provider === 'openai') {
      apiKeyContainer.classList.remove('hidden');
      apiBaseContainer.classList.remove('hidden');
      settingsBase.placeholder = "https://api.openai.com/v1";
    } else if (provider === 'xinference') {
      apiKeyContainer.classList.add('hidden');
      apiBaseContainer.classList.remove('hidden');
      settingsBase.placeholder = "http://localhost:9997";
    } else { // ollama
      apiKeyContainer.classList.add('hidden');
      apiBaseContainer.classList.remove('hidden');
      settingsBase.placeholder = "http://localhost:11434";
    }
  });

  settingsVlmProvider.addEventListener('change', () => {
    const provider = settingsVlmProvider.value;
    if (provider === 'openai') {
      vlmApiKeyContainer.classList.remove('hidden');
      vlmApiBaseContainer.classList.remove('hidden');
      settingsVlmBase.placeholder = "https://api.openai.com/v1";
    } else if (provider === 'xinference') {
      vlmApiKeyContainer.classList.add('hidden');
      vlmApiBaseContainer.classList.remove('hidden');
      settingsVlmBase.placeholder = "http://localhost:9997";
    } else { // ollama
      vlmApiKeyContainer.classList.add('hidden');
      vlmApiBaseContainer.classList.remove('hidden');
      settingsVlmBase.placeholder = "http://localhost:11434";
    }
  });

  settingsUseVlm.addEventListener('change', () => {
    if (settingsUseVlm.checked) {
      vlmModelContainer.classList.remove('hidden');
    } else {
      vlmModelContainer.classList.add('hidden');
    }
  });

  saveChatBtn.addEventListener('click', saveChatSettings);
  saveVlmBtn.addEventListener('click', saveVlmSettings);
}

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    
    // Update active settings UI if empty or initial load
    if (!settingsModel.value) {
      settingsProvider.value = data.active_provider.provider_type;
      settingsProvider.dispatchEvent(new Event('change'));
      
      settingsModel.value = data.active_provider.model_name;
      settingsBase.value = data.active_provider.api_base || '';
      if (data.active_provider.api_key) {
        settingsKey.value = data.active_provider.api_key;
      }
      
      settingsUseVlm.checked = data.vlm_provider.use_vlm;
      settingsUseVlm.dispatchEvent(new Event('change'));
      
      settingsVlmProvider.value = data.vlm_provider.provider_type || 'ollama';
      settingsVlmProvider.dispatchEvent(new Event('change'));
      
      settingsVlmModel.value = data.vlm_provider.model_name || '';
      settingsVlmBase.value = data.vlm_provider.api_base || '';
      if (data.vlm_provider.api_key) {
        settingsVlmKey.value = data.vlm_provider.api_key;
      }
    }
    
    // Update Ollama badge
    if (data.ollama.status === 'online') {
      statusOllama.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-green-500/10 text-green-500 font-medium border border-green-500/20";
      statusOllama.innerHTML = `<span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> Ollama: Online`;
      statusOllama.title = `Available models: ${data.ollama.models.join(', ')}`;
    } else {
      statusOllama.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/10 text-red-500 font-medium border border-red-500/20";
      statusOllama.innerHTML = `<span class="w-2 h-2 rounded-full bg-red-500"></span> Ollama: Offline`;
      statusOllama.title = "Could not connect to Ollama service.";
    }

    // Update Xinference badge
    if (data.xinference.status === 'online') {
      statusXinference.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-green-500/10 text-green-500 font-medium border border-green-500/20";
      statusXinference.innerHTML = `<span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> Xinference: Online`;
      statusXinference.title = `Available models: ${data.xinference.models.join(', ')}`;
    } else {
      statusXinference.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/10 text-red-500 font-medium border border-red-500/20";
      statusXinference.innerHTML = `<span class="w-2 h-2 rounded-full bg-red-500"></span> Xinference: Offline`;
      statusXinference.title = "Could not connect to Xinference service.";
    }
  } catch (e) {
    console.error("Failed to query status api", e);
  }
}

async function saveChatSettings() {
  saveChatBtn.disabled = true;
  saveChatBtn.innerHTML = '<span class="material-symbols-outlined text-sm animate-spin">sync</span> Saving...';
  
  const payload = {
    provider_type: settingsProvider.value,
    model_name: settingsModel.value,
    api_base: settingsBase.value || null,
    api_key: settingsKey.value || null
  };
  
  try {
    const res = await fetch('/api/settings/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (res.ok) {
      showToast("Chat settings saved and applied successfully!", "success");
      fetchStatus();
    } else {
      const err = await res.json();
      showToast(`Failed to update Chat settings: ${err.detail}`, "error");
    }
  } catch (e) {
    showToast(`Error updating Chat settings: ${e}`, "error");
  } finally {
    saveChatBtn.disabled = false;
    saveChatBtn.innerHTML = '<span class="material-symbols-outlined text-sm">save</span> Save Chat Settings';
  }
}

async function saveVlmSettings() {
  saveVlmBtn.disabled = true;
  saveVlmBtn.innerHTML = '<span class="material-symbols-outlined text-sm animate-spin">sync</span> Saving...';
  
  const payload = {
    use_vlm: settingsUseVlm.checked,
    vlm_provider_type: settingsVlmProvider.value,
    vlm_model: settingsVlmModel.value || null,
    vlm_api_base: settingsVlmBase.value || null,
    vlm_api_key: settingsVlmKey.value || null
  };
  
  try {
    const res = await fetch('/api/settings/vlm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (res.ok) {
      showToast("VLM settings saved and applied successfully!", "success");
      fetchStatus();
    } else {
      const err = await res.json();
      showToast(`Failed to update VLM settings: ${err.detail}`, "error");
    }
  } catch (e) {
    showToast(`Error updating VLM settings: ${e}`, "error");
  } finally {
    saveVlmBtn.disabled = false;
    saveVlmBtn.innerHTML = '<span class="material-symbols-outlined text-sm">save</span> Save VLM Settings';
  }
}

// ── File Upload & Drag-and-Drop ─────────────────────────────────────────────
function setupUploadUI() {
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-primary', 'bg-primary/5');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-primary', 'bg-primary/5');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-primary', 'bg-primary/5');
    if (e.dataTransfer.files.length > 0) {
      handleUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleUpload(fileInput.files[0]);
    }
  });
}

function handleUpload(file) {
  uploadProgressContainer.classList.remove('hidden');
  uploadFilename.textContent = file.name;
  uploadPercent.textContent = '0%';
  uploadProgressBar.style.width = '0%';
  uploadProgressBar.classList.remove('bg-red-500', 'bg-green-500', 'animate-pulse');
  uploadProgressBar.classList.add('bg-primary');
  
  const formData = new FormData();
  formData.append('file', file);
  
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload', true);
  
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const percent = Math.round((e.loaded / e.total) * 100);
      if (percent === 100) {
        uploadPercent.textContent = '100% (Parsing & Indexing...)';
        uploadProgressBar.classList.add('animate-pulse');
      } else {
        uploadPercent.textContent = `${percent}%`;
      }
      uploadProgressBar.style.width = `${percent}%`;
    }
  };
  
  xhr.onload = () => {
    uploadProgressBar.classList.remove('animate-pulse');
    if (xhr.status === 200) {
      const res = JSON.parse(xhr.responseText);
      uploadPercent.textContent = 'Completed!';
      uploadProgressBar.classList.remove('bg-primary');
      uploadProgressBar.classList.add('bg-green-500');
      uploadProgressBar.style.width = '100%';
      showToast(`Uploaded "${file.name}" successfully! PageIndex index tree generated.`, "success");
      fetchLibrary().then(() => {
        selectDocument(res.doc_id);
      });
    } else {
      let err_msg = "Unknown parsing error";
      try {
        err_msg = JSON.parse(xhr.responseText).detail || err_msg;
      } catch (e) {}
      uploadPercent.textContent = 'Failed';
      uploadProgressBar.classList.remove('bg-primary');
      uploadProgressBar.classList.add('bg-red-500');
      showToast(`Upload failed: ${err_msg}`, "error");
    }
    setTimeout(() => {
      uploadProgressContainer.classList.add('hidden');
    }, 4000);
  };
  
  xhr.onerror = () => {
    uploadProgressBar.classList.remove('animate-pulse');
    uploadPercent.textContent = 'Failed';
    uploadProgressBar.classList.remove('bg-primary');
    uploadProgressBar.classList.add('bg-red-500');
    showToast("Upload failed due to connection error.", "error");
    setTimeout(() => {
      uploadProgressContainer.classList.add('hidden');
    }, 4000);
  };
  
  xhr.send(formData);
}

// ── Library List ────────────────────────────────────────────────────────────
async function fetchLibrary() {
  try {
    const res = await fetch('/api/documents');
    documents = await res.json();
    renderLibrary();
  } catch (e) {
    console.error("Failed to query documents library", e);
  }
}

function renderLibrary() {
  if (documents.length === 0) {
    docList.innerHTML = '<div class="text-xs text-on-surface-variant/70 text-center py-6">No documents indexed yet. Upload one to get started!</div>';
    return;
  }
  
  docList.innerHTML = documents.map(doc => {
    const isSelected = doc.doc_id === activeDocId;
    const activeClass = isSelected ? 'bg-primary/10 border-primary/40' : 'bg-surface border-outline/10 hover:bg-outline/5';
    
    // Choose icon based on file type
    let icon = 'draft';
    const ext = doc.doc_name.split('.').pop().toLowerCase();
    if (ext === 'pdf') icon = 'picture_as_pdf';
    else if (ext === 'docx') icon = 'description';
    else if (['png', 'jpg', 'jpeg'].includes(ext)) icon = 'image';
    else if (ext === 'md' || ext === 'markdown') icon = 'markdown';
    
    const metricText = doc.page_count > 0 ? `${doc.page_count} pages` : `${doc.line_count} lines`;
    
    return `
      <div onclick="selectDocument('${doc.doc_id}')" class="p-3 border rounded-xl flex items-center justify-between cursor-pointer transition-all duration-200 ${activeClass}">
        <div class="flex items-center gap-3 min-w-0 flex-1">
          <span class="material-symbols-outlined text-primary shrink-0">${icon}</span>
          <div class="min-w-0 flex-1">
            <div class="text-xs font-semibold text-on-surface truncate pr-1" title="${doc.doc_name}">${doc.doc_name}</div>
            <div class="text-[10px] text-on-surface-variant/80 font-medium mt-0.5">${metricText}</div>
          </div>
        </div>
        <button onclick="event.stopPropagation(); deleteDocument('${doc.doc_id}')" class="p-1.5 rounded-lg hover:bg-red-500/10 text-on-surface-variant hover:text-red-500 transition-colors" title="Delete from index">
          <span class="material-symbols-outlined text-base">delete</span>
        </button>
      </div>
    `;
  }).join('');
}

async function deleteDocument(docId) {
  if (!confirm("Are you sure you want to delete this document from index? This will remove all structural caching and page visual assets.")) {
    return;
  }
  
  try {
    const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    if (res.ok) {
      showToast("Document deleted successfully.", "success");
      if (activeDocId === docId) {
        activeDocId = null;
        activeDocTitle.textContent = "Select a Document to Chat";
        activeDocPages.textContent = "";
        treeContainer.innerHTML = '<div class="text-on-surface-variant text-center py-10">Select a document to see its structural tree outline.</div>';
        clearPageDisplay();
      }
      fetchLibrary();
    } else {
      showToast("Failed to delete document.", "error");
    }
  } catch (e) {
    showToast(`Error deleting document: ${e}`, "error");
  }
}

async function selectDocument(docId) {
  activeDocId = docId;
  renderLibrary();
  
  const doc = documents.find(d => d.doc_id === docId);
  if (!doc) return;
  
  // Set title
  let icon = 'draft';
  const ext = doc.doc_name.split('.').pop().toLowerCase();
  if (ext === 'pdf') icon = 'picture_as_pdf';
  else if (ext === 'docx') icon = 'description';
  else if (['png', 'jpg', 'jpeg'].includes(ext)) icon = 'image';
  else if (ext === 'md' || ext === 'markdown') icon = 'markdown';
  
  activeDocIcon.textContent = icon;
  activeDocTitle.textContent = doc.doc_name;
  activeDocPages.textContent = doc.page_count > 0 ? `(${doc.page_count} pages)` : `(${doc.line_count} lines)`;
  activeDocType = doc.type;
  
  // Fetch details and outline tree
  treeContainer.innerHTML = '<div class="text-xs text-on-surface-variant/80 text-center py-6"><span class="animate-spin inline-block mr-1">sync</span>Loading tree structure...</div>';
  
  try {
    const res = await fetch(`/api/documents/${docId}`);
    const details = await res.json();
    renderTree(details.structure);
    // Select first page automatically
    inspectPage(1);
  } catch (e) {
    treeContainer.innerHTML = `<div class="text-xs text-red-500 text-center py-6">Failed to load structure: ${e}</div>`;
  }
}

// ── PageIndex Tree outline ───────────────────────────────────────────────────
function renderTree(structure) {
  if (!structure || structure.length === 0) {
    treeContainer.innerHTML = '<div class="text-on-surface-variant text-center py-4">No structure outlines found.</div>';
    return;
  }
  
  function buildTreeNodeHTML(node) {
    const hasChildren = node.nodes && node.nodes.length > 0;
    const targetIdx = node.start_index !== undefined ? node.start_index : (node.line_num !== undefined ? node.line_num : 1);
    const metricLabel = activeDocType === 'pdf' ? `p. ${targetIdx}` : `L ${targetIdx}`;
    
    let html = `
      <div class="tree-node flex flex-col pl-2 border-l border-outline/10 ml-1 mt-1">
        <div class="flex items-center justify-between p-1.5 rounded-lg hover:bg-outline/10 group cursor-pointer transition-colors" onclick="event.stopPropagation(); inspectPage(${targetIdx})">
          <div class="flex items-center gap-1.5 min-w-0 flex-1">
            ${hasChildren ? `
              <button onclick="event.stopPropagation(); toggleNodeCollapse(this)" class="p-0.5 rounded hover:bg-outline/20 flex items-center justify-center shrink-0">
                <span class="material-symbols-outlined text-sm font-semibold transition-transform">keyboard_arrow_down</span>
              </button>
            ` : '<span class="w-4 shrink-0"></span>'}
            <span class="text-[11px] font-semibold text-on-surface truncate group-hover:text-primary transition-colors">${node.title}</span>
          </div>
          <span class="text-[9px] font-bold text-on-surface-variant/80 shrink-0 bg-outline/10 border border-outline/15 px-1.5 py-0.5 rounded">${metricLabel}</span>
        </div>
    `;
    
    if (hasChildren) {
      html += `<div class="node-children flex flex-col pl-2 mt-0.5 space-y-0.5">`;
      node.nodes.forEach(child => {
        html += buildTreeNodeHTML(child);
      });
      html += `</div>`;
    }
    
    html += `</div>`;
    return html;
  }
  
  let treeHTML = '';
  structure.forEach(node => {
    treeHTML += buildTreeNodeHTML(node);
  });
  treeContainer.innerHTML = treeHTML;
}

function toggleNodeCollapse(btn) {
  const childrenContainer = btn.closest('.tree-node').querySelector('.node-children');
  const icon = btn.querySelector('.material-symbols-outlined');
  
  if (childrenContainer.classList.contains('hidden')) {
    childrenContainer.classList.remove('hidden');
    icon.style.transform = 'rotate(0deg)';
  } else {
    childrenContainer.classList.add('hidden');
    icon.style.transform = 'rotate(-90deg)';
  }
}

// ── Inspector Tab Navigation & Dual View ────────────────────────────────────
function setupTabsUI() {
  tabImageBtn.addEventListener('click', () => switchTab('image'));
  tabTextBtn.addEventListener('click', () => switchTab('text'));
}

function switchTab(tab) {
  currentTab = tab;
  if (tab === 'image') {
    tabImageBtn.className = "flex-1 py-2 text-xs font-semibold text-primary border-b-2 border-primary focus:outline-none flex items-center justify-center gap-1";
    tabTextBtn.className = "flex-1 py-2 text-xs font-semibold text-on-surface-variant border-b-2 border-transparent hover:bg-outline/5 focus:outline-none flex items-center justify-center gap-1";
    tabImageContent.classList.remove('hidden');
    tabTextContent.classList.add('hidden');
  } else {
    tabTextBtn.className = "flex-1 py-2 text-xs font-semibold text-primary border-b-2 border-primary focus:outline-none flex items-center justify-center gap-1";
    tabImageBtn.className = "flex-1 py-2 text-xs font-semibold text-on-surface-variant border-b-2 border-transparent hover:bg-outline/5 focus:outline-none flex items-center justify-center gap-1";
    tabImageContent.classList.add('hidden');
    tabTextContent.classList.remove('hidden');
  }
  
  // Reload content
  if (activeDocId) {
    loadPageContent();
  }
}

function clearPageDisplay() {
  pageImageDisplay.classList.add('hidden');
  pageTextDisplay.classList.add('hidden');
  imageViewerPlaceholder.classList.remove('hidden');
  textViewerPlaceholder.classList.remove('hidden');
}

function inspectPage(idx) {
  activePageNum = idx;
  
  // Highlighting active visual in tree could be done, but let's load content first
  if (activeDocId) {
    loadPageContent();
  }
}

async function loadPageContent() {
  imageViewerPlaceholder.classList.add('hidden');
  textViewerPlaceholder.classList.add('hidden');
  
  // Load page image (only works for physical pdf pages or image uploads)
  const imagePageNum = activeDocType === 'pdf' ? activePageNum : 1;
  const imageApiUrl = `/api/documents/${activeDocId}/pages/${imagePageNum}/image`;
  
  // Setup loading state
  pageImageDisplay.classList.add('hidden');
  pageTextDisplay.classList.add('hidden');
  
  if (currentTab === 'image') {
    imageViewerPlaceholder.innerHTML = '<span class="animate-spin inline-block text-base mr-1">sync</span>Loading original page image...';
    imageViewerPlaceholder.classList.remove('hidden');
    
    // Test if image exists first
    const testImg = new Image();
    testImg.onload = () => {
      imageViewerPlaceholder.classList.add('hidden');
      pageImageDisplay.src = imageApiUrl;
      pageImageDisplay.classList.remove('hidden');
    };
    testImg.onerror = () => {
      imageViewerPlaceholder.textContent = `Original page image not available for page ${imagePageNum}`;
      imageViewerPlaceholder.classList.remove('hidden');
    };
    testImg.src = imageApiUrl;
  } else {
    textViewerPlaceholder.innerHTML = '<span class="animate-spin inline-block text-base mr-1">sync</span>Loading structured content...';
    textViewerPlaceholder.classList.remove('hidden');
    
    try {
      const res = await fetch(`/api/documents/${activeDocId}/pages/${activePageNum}/text`);
      if (res.ok) {
        const data = await res.json();
        textViewerPlaceholder.classList.add('hidden');
        pageTextDisplay.innerHTML = marked.parse(data.content || '');
        pageTextDisplay.classList.remove('hidden');
      } else {
        textViewerPlaceholder.textContent = `Content not found at page/line index ${activePageNum}`;
        textViewerPlaceholder.classList.remove('hidden');
      }
    } catch (e) {
      textViewerPlaceholder.textContent = `Failed to fetch text: ${e}`;
      textViewerPlaceholder.classList.remove('hidden');
    }
  }
}

// ── Agentic Chat UI & Streaming ─────────────────────────────────────────────
function setupChatUI() {
  sendBtn.addEventListener('click', handleChatSubmit);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleChatSubmit();
    }
  });
}

async function handleChatSubmit() {
  if (!activeDocId) {
    showToast("Please select a document first before asking questions.", "error");
    return;
  }
  
  const query = chatInput.value.trim();
  if (!query) return;
  
  // Clear input
  chatInput.value = '';
  chatInput.style.height = 'auto';
  
  // Append user message
  appendMessage('user', query);
  
  // Create AI message container for streaming
  const messageId = `ai-msg-${Date.now()}`;
  const bubbleContainer = appendAIStreamingMessageContainer(messageId);
  const statusContainer = bubbleContainer.querySelector('.reasoning-status-box');
  const deltaTextContainer = bubbleContainer.querySelector('.markdown-body');
  const citationsContainer = bubbleContainer.querySelector('.citations-box');
  
  // Setup streaming POST payload
  const payload = {
    doc_id: activeDocId,
    query: query,
    force_search: chatForceSearch.checked
  };
  
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`Server returned status code: ${response.status}`);
    }
    
    // Read steam reader
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let accumulatedText = "";
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Retain last partial line
      buffer = lines.pop();
      
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);
          
          if (data.type === 'status') {
            // Update reasoning steps
            appendReasoningStep(statusContainer, data.content);
          } else if (data.type === 'delta') {
            // Accumulate response text
            accumulatedText += data.content;
            deltaTextContainer.innerHTML = marked.parse(accumulatedText);
            // Auto scroll chat
            chatMessages.scrollTop = chatMessages.scrollHeight;
          } else if (data.type === 'error') {
            appendReasoningStep(statusContainer, `Error: ${data.content}`, true);
          } else if (data.type === 'result') {
            // Stream complete. Render final markdown & citations
            if (accumulatedText) {
              deltaTextContainer.innerHTML = renderCitationsInText(accumulatedText, data.sources);
            }
            renderCitationsFooter(citationsContainer, data.sources, data.fallback);
          }
        } catch (err) {
          console.warn("Error parsing stream line:", line, err);
        }
      }
    }
    
  } catch (e) {
    loggerErrorBubble(messageId, `Chat execution failed: ${e.message}`);
  }
}

function appendMessage(sender, text) {
  const isAI = sender === 'ai';
  const icon = isAI ? 'chat' : 'person';
  const bgClass = isAI ? 'bg-surface-container' : 'bg-primary text-on-primary';
  const label = isAI ? 'AI ASSISTANT' : 'YOU';
  
  const html = `
    <div class="flex items-start gap-4 chat-bubble">
      <div class="w-8 h-8 rounded-full ${isAI ? 'bg-primary/10 border border-primary/20' : 'bg-primary/20 border border-primary/40'} flex items-center justify-center shrink-0">
        <span class="material-symbols-outlined text-primary text-lg">${icon}</span>
      </div>
      <div class="space-y-1 max-w-[80%]">
        <div class="text-[10px] text-on-surface-variant font-bold tracking-wider">${label}</div>
        <div class="p-4 rounded-2xl ${bgClass} text-sm leading-relaxed shadow-sm">
          ${isAI ? marked.parse(text) : text.replace(/\n/g, '<br>')}
        </div>
      </div>
    </div>
  `;
  chatMessages.insertAdjacentHTML('beforeend', html);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendAIStreamingMessageContainer(messageId) {
  const html = `
    <div class="flex items-start gap-4 chat-bubble" id="${messageId}">
      <div class="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
        <span class="material-symbols-outlined text-primary text-lg">chat</span>
      </div>
      <div class="space-y-2 max-w-[80%] flex-1">
        <div class="text-[10px] text-on-surface-variant font-bold tracking-wider">AI ASSISTANT</div>
        
        <!-- Reasoning steps logs box -->
        <div class="reasoning-status-box flex flex-col gap-1.5 p-3 rounded-xl bg-outline/5 border border-outline/10 text-[11px] font-medium text-on-surface-variant hidden">
          <!-- Populated in real-time -->
        </div>
        
        <!-- Stream content -->
        <div class="p-4 rounded-2xl bg-surface-container text-sm leading-relaxed shadow-sm">
          <div class="markdown-body prose dark:prose-invert text-sm max-w-none text-on-surface">
            <span class="animate-pulse inline-block w-2 h-4 bg-primary"></span>
          </div>
          
          <!-- Citations list footer -->
          <div class="citations-box flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-outline/10 hidden">
          </div>
        </div>
      </div>
    </div>
  `;
  chatMessages.insertAdjacentHTML('beforeend', html);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return document.getElementById(messageId);
}

function appendReasoningStep(container, stepText, isError = false) {
  container.classList.remove('hidden');
  const icon = isError ? 'error' : 'check_circle';
  const color = isError ? 'text-red-500' : 'text-primary';
  const html = `
    <div class="flex items-center gap-1.5">
      <span class="material-symbols-outlined text-sm shrink-0 ${color}">${icon}</span>
      <span class="truncate">${stepText}</span>
    </div>
  `;
  container.insertAdjacentHTML('beforeend', html);
}

function loggerErrorBubble(messageId, errorMsg) {
  const bubble = document.getElementById(messageId);
  if (bubble) {
    const statusBox = bubble.querySelector('.reasoning-status-box');
    appendReasoningStep(statusBox, errorMsg, true);
    
    const mdBody = bubble.querySelector('.markdown-body');
    mdBody.innerHTML = `<span class="text-red-500 font-semibold">${errorMsg}</span>`;
  }
}

// ── Interactive Citation Pill Replacement ───────────────────────────────────
function renderCitationsInText(text, sources) {
  // Matches references like [Page 4], [4], [L 4], [Line 4]
  text = text.replace(/\[(?:Page\s+)?(\d+)\]/gi, (match, pageNum) => {
    return `<span class="citation-pill" onclick="inspectPage(${pageNum})"><span class="material-symbols-outlined text-[10px]">auto_stories</span>Page ${pageNum}</span>`;
  });
  text = text.replace(/\[(?:Line\s+|L\s+)?(\d+)\]/gi, (match, lineNum) => {
    return `<span class="citation-pill" onclick="inspectPage(${lineNum})"><span class="material-symbols-outlined text-[10px]">subject</span>L ${lineNum}</span>`;
  });
  return marked.parse(text);
}

function renderCitationsFooter(container, sources, isFallback) {
  if (!sources || sources.length === 0) {
    container.classList.add('hidden');
    return;
  }
  
  container.classList.remove('hidden');
  
  if (isFallback) {
    // Web search fallback sources
    container.innerHTML = `
      <div class="w-full text-[10px] text-on-surface-variant font-bold mb-1.5 flex items-center gap-1">
        <span class="material-symbols-outlined text-xs text-primary">public</span> CITATIONS FROM WEB FALLBACK
      </div>
    ` + sources.map(src => {
      return `
        <a href="${src.url}" target="_blank" class="citation-pill">
          <span class="material-symbols-outlined text-[10px]">link</span> [${src.id}] ${src.title.substring(0, 25)}...
        </a>
      `;
    }).join('');
  } else {
    // Normal document sources
    container.innerHTML = `
      <div class="w-full text-[10px] text-on-surface-variant font-bold mb-1.5 flex items-center gap-1">
        <span class="material-symbols-outlined text-xs text-primary">article</span> REFERENCED OUTLINE PAGES
      </div>
    ` + sources.map(src => {
      const idxLabel = activeDocType === 'pdf' ? `Page ${src.page}` : `Line ${src.page}`;
      return `
        <span onclick="inspectPage(${src.page})" class="citation-pill">
          <span class="material-symbols-outlined text-[10px]">auto_stories</span> ${idxLabel} (${src.doc_name.substring(0, 15)}...)
        </span>
      `;
    }).join('');
  }
}

// ── Toast Notifications Helper ──────────────────────────────────────────────
function showToast(message, type = "success") {
  const toastId = `toast-${Date.now()}`;
  const bgClass = type === 'success' ? 'bg-primary text-on-primary' : 'bg-error text-on-primary';
  const icon = type === 'success' ? 'check_circle' : 'error';
  
  const html = `
    <div id="${toastId}" class="fixed bottom-6 right-6 flex items-center gap-3 px-4 py-3 rounded-xl shadow-lg ${bgClass} font-medium text-xs z-50 animate-bounce transition-all duration-300">
      <span class="material-symbols-outlined text-lg">${icon}</span>
      <span>${message}</span>
    </div>
  `;
  document.body.insertAdjacentHTML('beforeend', html);
  
  setTimeout(() => {
    const element = document.getElementById(toastId);
    if (element) {
      element.classList.add('opacity-0', 'translate-y-2');
      setTimeout(() => element.remove(), 300);
    }
  }, 4000);
}
