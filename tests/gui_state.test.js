'use strict';

const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

class Element {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.textContent = '';
    this.style = { setProperty() {} };
    this.dataset = {};
    this.value = '';
    this.classList = { toggle() {}, add() {}, remove() {} };
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  querySelector() { return this.marker ||= new Element(); }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
}

function snapshot(revision, paths, phase = 'idle', extra = {}) {
  return { revision, paths, count: paths.length, phase, progress: null, result: null, run_error: null, ...extra };
}

const result = { total_seconds: 3, detected_rows: 2, exported_transactions: 2, warning_rows: 0, output_dir: '/out', video_warnings: [] };

async function application(apiOverrides = {}) {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const document = {
    body: new Element(),
    getElementById: element,
    createElement: () => new Element(),
    createElementNS: () => new Element(),
    querySelector: element,
    querySelectorAll: () => [],
    addEventListener() {},
  };
  const window = {
    pywebview: { api: { get_inputs: async () => snapshot(0, []), ...apiOverrides } },
    addEventListener() {},
    setInterval: () => 1,
    clearInterval() {},
  };
  const context = vm.createContext({ document, window, Date });
  vm.runInContext(readFileSync(join(__dirname, '../octopus_ocr/assets/ui/app.js'), 'utf8'), context);
  await Promise.resolve();
  await Promise.resolve();
  return { window, document, element, apply: window.OctopusOCR.onStateChanged };
}

test('a late queue callback cannot replace the newer queue or reset a running job', async () => {
  const app = await application();
  app.apply(snapshot(2, ['/first.png', '/second.png']));
  app.apply(snapshot(3, ['/first.png', '/second.png'], 'running'));
  app.apply(snapshot(1, ['/first.png']));
  assert.equal(app.element('input-count').textContent, 2);
  assert.equal(app.document.body.dataset.state, 'running');
  assert.equal(app.element('run').disabled, true);
});

test('a late completion cannot restore invalidated exports', async () => {
  const app = await application();
  app.apply(snapshot(5, ['/new.png']));
  app.apply(snapshot(4, ['/old.png'], 'complete', { result }));
  assert.equal(app.document.body.dataset.state, 'idle');
  assert.equal(app.element('results').hidden, true);
  assert.equal(app.element('input-list').children[0].children[1].children[0].textContent, 'new.png');
});

test('a delayed remove Promise uses the same version check as native callbacks', async () => {
  let resolveRemove;
  const app = await application({ remove_input: () => new Promise((resolve) => { resolveRemove = resolve; }) });
  app.apply(snapshot(1, ['/first.png', '/second.png']));
  const removal = app.element('input-list').children[0].children[3].listeners.click();
  app.apply(snapshot(3, ['/third.png'], 'idle', { error: 'Current warning' }));
  resolveRemove(snapshot(2, ['/second.png'], 'idle', { error: 'Stale warning' }));
  await removal;
  assert.equal(app.element('input-list').children[0].children[1].children[0].textContent, 'third.png');
  assert.equal(app.element('error').textContent, 'Current warning');
});

test('progress preserves existing queue nodes and their focus', async () => {
  const app = await application();
  app.apply(snapshot(1, ['/first.png']));
  const row = app.element('input-list').children[0];
  app.apply(snapshot(2, ['/first.png'], 'running'));
  app.apply(snapshot(3, ['/first.png'], 'running', { progress: { phase: 'ocr', completed_rows: 1, total_rows: 2, image_index: 1, image_count: 1 } }));
  assert.equal(app.element('input-list').children[0], row);
  assert.equal(app.document.body.dataset.state, 'running');
});

test('rejected runs restore the authoritative completed state', async () => {
  const app = await application({ run: async () => ({ accepted: false, error: 'Choose an engine', snapshot: snapshot(2, ['/first.png'], 'complete', { result }) }) });
  app.apply(snapshot(2, ['/first.png'], 'complete', { result }));
  await app.element('run').listeners.click();
  assert.equal(app.document.body.dataset.state, 'complete');
  assert.equal(app.element('results').hidden, false);
  assert.equal(app.element('error').textContent, 'Choose an engine');
});

test('video extraction warnings remain visible beside completed exports', async () => {
  const app = await application();
  app.apply(snapshot(2, ['/recording.mov'], 'complete', { result: { ...result, video_warnings: ['Possible skipped transactions'] } }));
  assert.equal(app.element('error').hidden, false);
  assert.equal(app.element('error').textContent, 'Possible skipped transactions');
  assert.equal(app.element('warnings').textContent, 0);
  assert.equal(app.element('results').hidden, false);
});
