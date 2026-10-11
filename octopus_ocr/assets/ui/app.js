'use strict';

(() => {
// Keep app bindings out of the global scope used by pywebview's DOM scripts.

const element = (id) => document.getElementById(id);
const state = { revision: -1, phase: 'idle', paths: [], bridgeReady: false, startedAt: null, elapsedTimer: null, progress: 0 };
const svgNamespace = 'http://www.w3.org/2000/svg';

function icon(name) {
  const svg = document.createElementNS(svgNamespace, 'svg');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(svgNamespace, 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.appendChild(use);
  return svg;
}

function showError(message = '') {
  element('error').textContent = message;
  element('error').hidden = !message;
}

function setWorkflow(stage) {
  ['files', 'extract', 'review'].forEach((name, index) => {
    const step = element(`step-${name}`);
    step.classList.toggle('current', index === stage);
    step.classList.toggle('done', index < stage);
    if (index === stage) step.setAttribute('aria-current', 'step');
    else step.removeAttribute('aria-current');
    step.querySelector('.step-marker').textContent = index < stage ? '\u2713' : String(index + 1);
  });
}

function updateControls() {
  const running = state.phase === 'running';
  const unavailable = !state.bridgeReady || running;
  document.querySelectorAll('[data-source], .remove-file').forEach((button) => { button.disabled = unavailable; });
  element('clear').disabled = unavailable || state.paths.length === 0;
  element('ocr-engine').disabled = unavailable;
  element('paddle-model').disabled = unavailable;
  element('run').disabled = unavailable || state.paths.length === 0;
  element('run-label').textContent = running ? 'Extracting transactions' : 'Extract transactions';
  const count = state.paths.length;
  const noun = count === 1 ? 'file' : 'files';
  element('selection-summary').textContent = count ? `${count} ${noun} ${running ? 'being processed' : state.phase === 'complete' ? 'processed' : 'ready'}` : '';
  element('selection-summary').hidden = count === 0;
  element('drop-hint').textContent = running ? 'Processing files' : 'Drop more files';
  element('dropzone').setAttribute('aria-busy', String(running));
}

function setPhase(phase) {
  state.phase = phase;
  document.body.dataset.state = phase;
  const labels = { idle: 'Ready to import', running: 'Processing', complete: 'Complete', error: 'Needs attention' };
  element('session-label').textContent = state.bridgeReady ? labels[phase] : 'Connecting';
  setWorkflow(phase === 'complete' ? 2 : phase === 'running' ? 1 : 0);
  updateControls();
}

function setStatus(title, message, iconName) {
  element('status-title').textContent = title;
  element('status').textContent = message;
  element('status').hidden = !message;
  element('status-icon').setAttribute('href', `#i-${iconName}`);
}

function resetRun() {
  stopElapsedTimer();
  state.progress = 0;
  element('results').hidden = true;
  element('output-preview').hidden = false;
  element('elapsed').hidden = true;
  element('progress-track').hidden = true;
  element('progress-percent').hidden = true;
  document.querySelector('.other-exports').open = false;
  setPhase('idle');
  setStatus('Ready', '', 'file');
}

function renderInputs(payload) {
  const paths = payload.paths || [];
  state.paths = paths;
  element('input-count').textContent = paths.length;
  element('dropzone').style.setProperty('--file-count', paths.length);
  element('drop-empty').hidden = paths.length > 0;
  element('populated-sources').hidden = paths.length === 0;
  element('queue-actions').hidden = paths.length === 0;

  const list = element('input-list');
  list.replaceChildren();
  for (const path of paths) {
    const parts = path.split(/[\\/]/);
    const name = parts.pop();
    const extension = name.split('.').pop().toLowerCase();
    const video = ['mov', 'mp4', 'm4v'].includes(extension);
    const item = document.createElement('li');
    item.className = 'file-row';
    const fileIcon = document.createElement('span');
    fileIcon.className = `source-file-icon${video ? ' video' : ''}`;
    fileIcon.appendChild(icon(video ? 'video' : 'image'));
    const info = document.createElement('div');
    info.className = 'file-info';
    const fileName = document.createElement('span');
    fileName.className = 'file-name';
    fileName.textContent = name;
    fileName.title = path;
    const folder = document.createElement('span');
    folder.className = 'file-folder';
    folder.textContent = parts.join('/') || 'Current folder';
    folder.title = path;
    info.append(fileName, folder);
    const type = document.createElement('span');
    type.className = 'file-type';
    type.textContent = extension.toUpperCase();
    const remove = document.createElement('button');
    remove.className = 'quiet-button remove-file';
    remove.setAttribute('aria-label', `Remove ${name}`);
    remove.title = `Remove ${name}`;
    remove.appendChild(icon('close'));
    remove.addEventListener('click', async () => {
      const response = await invoke('remove_input', path);
      if (response) applySnapshot(response);
    });
    item.append(fileIcon, info, type, remove);
    list.appendChild(item);
  }
  updateControls();
}

function progressPercent(event) {
  if (event.phase === 'complete') return 99;
  if (event.phase === 'export') return 97;
  if (event.phase === 'ocr' && event.total_rows > 0) return 20 + Math.round(event.completed_rows / event.total_rows * 76);
  if (event.phase === 'detect' && event.image_count > 0) return Math.min(20, Math.round(event.image_index / event.image_count * 20));
  return 3;
}

function setProgress(percent) {
  state.progress = Math.max(state.progress, Math.min(100, percent));
  element('progress-fill').style.width = `${state.progress}%`;
  element('progress-track').setAttribute('aria-valuenow', String(state.progress));
  element('progress-percent').textContent = `${state.progress}%`;
}

function setElapsed(seconds) {
  const minutes = Math.floor(seconds / 60);
  element('elapsed-label').textContent = minutes ? `${minutes}m ${Math.floor(seconds % 60)}s` : `${seconds.toFixed(1)}s`;
}

function stopElapsedTimer(finalSeconds = null) {
  if (state.elapsedTimer !== null) window.clearInterval(state.elapsedTimer);
  if (finalSeconds !== null) setElapsed(finalSeconds);
  else if (state.startedAt !== null) setElapsed((Date.now() - state.startedAt) / 1000);
  state.elapsedTimer = null;
  state.startedAt = null;
}

function startElapsedTimer() {
  stopElapsedTimer();
  state.startedAt = Date.now();
  setElapsed(0);
  element('elapsed').hidden = false;
  state.elapsedTimer = window.setInterval(() => setElapsed((Date.now() - state.startedAt) / 1000), 250);
}

async function invoke(method, ...args) {
  try {
    if (!state.bridgeReady) throw new Error('The desktop connection is not ready. Relaunch Octopus OCR if this continues.');
    const response = await window.pywebview.api[method](...args);
    return response;
  } catch (error) {
    showError(error.message || 'The desktop action failed. Try again.');
    return null;
  }
}

const runViews = {
  started() {
    state.progress = 0;
    showError();
    element('results').hidden = true;
    element('output-preview').hidden = true;
    element('progress-track').hidden = false;
    element('progress-percent').hidden = false;
    setPhase('running');
    setStatus('Preparing your files', '', 'scan');
    setProgress(2);
    startElapsedTimer();
  },
  progress(event) {
    setProgress(progressPercent(event));
    const titles = { prepare: 'Preparing your files', detect: 'Finding transactions', ocr: 'Reading transactions', export: 'Creating your export files', complete: 'Finishing up' };
    let message = event.message || 'Processing your files.';
    if (event.phase === 'ocr') message = `${event.completed_rows} of ${event.total_rows} rows read. Screenshot ${event.image_index} of ${event.image_count}.`;
    if (event.phase === 'detect' && event.image_index > 0) message = `Looking for transaction rows in screenshot ${event.image_index} of ${event.image_count}.`;
    setStatus(titles[event.phase] || 'Extracting transactions', message, 'scan');
  },
  complete(result) {
    stopElapsedTimer(result.total_seconds);
    setPhase('complete');
    setProgress(100);
    element('progress-track').hidden = true;
    element('progress-percent').hidden = true;
    const count = result.exported_transactions;
    const warnings = result.warning_rows;
    const message = warnings ? `${warnings} ${warnings === 1 ? 'row needs' : 'rows need'} a check in the review CSV before importing.` : '';
    setStatus(count ? 'Your export files are ready' : 'No transactions found', count ? message : 'Check that the screenshots show Octopus transaction activity, then try again.', count ? 'check' : 'warning');
    element('rows').textContent = result.detected_rows;
    element('transactions').textContent = count;
    element('warnings').textContent = warnings;
    element('warning-metric').classList.toggle('has-warnings', warnings > 0);
    element('output-path').textContent = `Saved to ${result.output_dir}`;
    element('output-path').title = result.output_dir;
    element('results').hidden = false;
    element('output-preview').hidden = true;
    showError((result.video_warnings || []).join('\n'));
  },
  error(message) {
    stopElapsedTimer();
    setPhase('error');
    setStatus('Extraction stopped', '', 'warning');
    element('progress-track').hidden = true;
    element('progress-percent').hidden = true;
    element('output-preview').hidden = true;
    element('results').hidden = true;
    showError(message || 'Extraction failed. Try again.');
  }
};

function applySnapshot(snapshot) {
  // Native callbacks and API Promise replies can arrive in either order.
  if (!snapshot || snapshot.revision < state.revision) return false;
  const initial = state.revision === -1;
  state.revision = snapshot.revision;
  const changed = JSON.stringify(snapshot.paths) !== JSON.stringify(state.paths);
  const previousPhase = state.phase;
  if (initial || changed) renderInputs(snapshot);
  if (snapshot.phase === 'running') {
    if (previousPhase !== 'running' || state.startedAt === null) runViews.started();
    if (snapshot.progress) runViews.progress(snapshot.progress);
  } else if (snapshot.phase === 'complete') {
    runViews.complete(snapshot.result);
  } else if (snapshot.phase === 'error') {
    runViews.error(snapshot.run_error);
  } else if (changed || previousPhase !== 'idle') {
    resetRun();
    showError();
  }
  setPhase(snapshot.phase);
  if (snapshot.error) showError(snapshot.error);
  return true;
}

window.OctopusOCR = { onStateChanged: applySnapshot };

for (const id of ['pick-files', 'add-files']) {
  element(id).addEventListener('click', async () => {
    showError();
    const response = await invoke('choose_files');
    if (response) applySnapshot(response);
  });
}
for (const id of ['pick-folder', 'add-folder']) {
  element(id).addEventListener('click', async () => {
    showError();
    const response = await invoke('choose_folder');
    if (response) applySnapshot(response);
  });
}
element('clear').addEventListener('click', async () => {
  const response = await invoke('clear_inputs');
  if (response) applySnapshot(response);
});
element('run').addEventListener('click', async () => {
  // Lock immediately; the Python callback confirms that the run actually started.
  setPhase('running');
  const response = await invoke('run', { ocr_engine: element('ocr-engine').value, paddle_model: element('paddle-model').value });
  if (response?.snapshot && !applySnapshot(response.snapshot)) return;
  if (!response) runViews.error(element('error').textContent || 'Could not start extraction.');
  else if (!response.accepted) showError(response.error || 'Could not start extraction.');
});
document.querySelectorAll('[data-output]').forEach((button) => {
  button.addEventListener('click', async () => {
    const response = await invoke('open_output', button.dataset.output);
    if (response && !response.ok) showError(response.error || 'Could not open output.');
  });
});
element('ocr-engine').addEventListener('change', () => {
  const paddle = element('ocr-engine').value === 'paddle';
  element('paddle-settings').hidden = !paddle;
});

let dragDepth = 0;
document.addEventListener('dragenter', (event) => {
  if (state.phase === 'running') return;
  if (Array.from(event.dataTransfer?.types || []).includes('Files')) {
    dragDepth += 1;
    element('dropzone').classList.add('dragging');
  }
});
document.addEventListener('dragover', (event) => {
  event.preventDefault();
  if (event.dataTransfer) event.dataTransfer.dropEffect = state.phase === 'running' ? 'none' : 'copy';
});
document.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) element('dropzone').classList.remove('dragging');
});
document.addEventListener('drop', (event) => {
  event.preventDefault();
  dragDepth = 0;
  element('dropzone').classList.remove('dragging');
  // Full file paths are supplied by pywebview's native DOM drop handler.
});
window.addEventListener('blur', () => {
  dragDepth = 0;
  element('dropzone').classList.remove('dragging');
});

async function connectBridge() {
  if (state.bridgeReady) return;
  state.bridgeReady = true;
  setPhase('idle');
  const response = await invoke('get_inputs');
  if (response) applySnapshot(response);
}
window.addEventListener('pywebviewready', connectBridge);
if (window.pywebview?.api) connectBridge();

})();
