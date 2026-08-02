const state = {
  files: [], mode: null, jobId: null, pollTimer: null, activeJob: null,
  reviewIssues: [], reviewIndex: 0, creatingJob: false, pollFailures: 0,
  revision: -1, utilityReturnFocus: null, language: 'en',
  resourceTimer: null, clockTimer: null, jobStartedAt: null,
  etaSeconds: null, etaSamples: [], etaPhase: null, cancelArmed: false,
  previewUrl: null, previewLoadedUrl: null
};
const $ = (id) => document.getElementById(id);
const accessToken = new URLSearchParams(window.location.search).get('token') || '';

const messages = {
  en: {
    appToolbar: 'Application toolbar', language: 'Language',
    systemStatus: 'System status', close: 'Close', conversionWorkflow: 'Conversion workflow', import: 'Import',
    recognize: 'Recognize', output: 'Output', runtimeCheck: 'Runtime check', closeSystemStatus: 'Close system status',
    checking: 'Checking…', allChecks: 'All checks', downloadDiagnostics: 'Download diagnostics', unfinished: 'Unfinished',
    resumePrevious: 'Resume previous conversion', resume: 'Resume', dismiss: 'Dismiss', importLabel: '01 / Import',
    newConversion: 'New conversion', importScoreScans: 'Import score scans', dropScans: 'Drop scans here',
    imagesOrPdf: 'Images or PDF', addImages: 'Add images', addPdf: 'Add PDF', pageOrder: 'Page order', clear: 'Clear',
    orientationCorrect: 'Page orientation is correct', noAutoRotation: 'Pages will not be rotated or deskewed automatically',
    conversionDetails: 'Conversion details', thisOutput: 'This output', oneContinuousScore: 'One continuous score', files: 'Files',
    order: 'Order', mergeOrder: 'Top-to-bottom into one MusicXML file', expandByPage: 'Expanded by page number', runtime: 'Runtime',
    quickSettings: 'Quick settings', conversionSettings: 'Conversion settings', outputFileName: 'Output file name',
    automaticFileName: 'Automatic', outputFiles: 'Output files', correctOrientation: 'Import correctly oriented score pages.',
    supported: 'Supported', supportedScope: 'Printed Western staff notation; monophonic instruments; common piano scores; aligned full scores. Instruments do not need note-by-note horizontal alignment.',
    notSupported: 'Not supported', unsupportedScope: 'Handwriting, percussion, tablature, lyrics, chord symbols, figured bass, and graphic notation. Convert separately scanned parts as separate jobs.',
    scenarioLimits: 'Scenarios and feature limits', fullScore: 'Full score', fullScoreLimit: 'Up to 16 physical staves per system and one keyboard part',
    piano: 'Piano', pianoLimit: 'Up to 4 physical staves; up to 8 independent rhythmic voices per measure and staff; temporary staves, ossia, cross-staff notes, and cross-staff beams',
    otherParts: 'Other parts', otherPartsLimit: 'One independent rhythmic voice per staff', multipleScans: 'Multiple scans',
    multipleScansLimit: 'Consecutive pages of the same score only', noFiles: 'No files added', noPages: 'No pages yet',
    startConversion: 'Start conversion', reviewLabel: '03 / Review', reviewItems: 'Review items', backToOutput: 'Back to output',
    scanDetail: 'Scan detail', content: 'Content', apply: 'Apply', skip: 'Skip', outputLabel: '03 / Output',
    conversionResult: 'Conversion result', checkDetails: 'Check details', handleChecks: 'Handle checks', downloadMusicXml: 'Download MusicXML',
    downloadMxl: 'Download MXL', exportMxl: 'Export MXL', exportMusicXml: 'Export MusicXML',
    mxlMuseScoreNote: 'MXL files can be viewed and edited in MuseScore', preview: 'Preview', closePreview: 'Close preview',
    museScoreInstrumentNote: 'After export, assign the correct instrument to each part in MuseScore. Otherwise, displayed pitches may be incorrect.',
    loadingPreview: 'Loading preview…', previewUnavailable: 'Preview could not be loaded.', previewPage: 'Score preview page {count}',
    conversionReport: 'Conversion report', openMusicXml: 'Open MusicXML', openMxl: 'Open MXL', openFolder: 'Open folder',
    closeProduct: 'Close Pigeon Score Scan?', returnToApp: 'Return to application', chooseCloseAction: 'The service will stop.',
    chooseCloseRunning: 'The active conversion will stop.', cancel: 'Cancel', exitCompletely: 'Exit completely',
    unsupportedFile: 'This file type is not supported. Add PNG, JPG, TIFF, BMP, WebP, or PDF.',
    mixedFiles: 'Images and PDF files must be converted separately.', imagesOnly: 'Add image files here.', pdfOnly: 'Add PDF files here.',
    differentMode: 'The list already contains another file type. Clear it before adding these files.', duplicateFile: 'The file is already in the list.',
    emptyFile: 'Empty files cannot be added.', sizeLimit: 'The total file size cannot exceed 1 GB.', imageFiles: 'images', inputFiles: 'input files',
    moveUp: 'Move up', moveDown: 'Move down', remove: 'Remove', couldNotCreate: 'Could not create the conversion.',
    couldNotReadStatus: 'Could not read conversion status.', couldNotReadResult: 'Could not read the conversion result.',
    couldNotReadChecks: 'Could not read review items.', couldNotApply: 'Could not apply the change.', operationFailed: 'The operation failed.',
    item: 'item', items: 'items', confirmResult: 'Confirm recognition result', markReviewed: 'Mark as reviewed', keepCurrent: 'Keep current result',
    conversionIncomplete: 'Conversion incomplete', conversionFailed: 'Conversion failed.', cancelled: 'Conversion cancelled.',
    outputChecksPassed: 'Output checks passed', checksPassed: 'Checks passed', checksPresent: 'Checks required', checksRemain: 'Checks remain',
    outputLimited: 'Manual review required', conversionComplete: 'Conversion complete', page: 'page', pages: 'pages', pageRecognitionFailed: 'page recognition failed',
    pageRecognitionFailedPlural: 'pages failed recognition', checksNotPassed: 'Some output checks did not pass', handleCount: 'Handle {count} checks',
    structurePassed: 'Structure, rhythm, notation, text, and file integrity: passed', outputConditions: '{count} output conditions not met',
    fileIntegrityPassed: 'File integrity: passed', fileIntegrityIncomplete: 'File integrity: incomplete', verificationId: 'Verification ID',
    exportCurrentMusicXml: 'Export current MusicXML', exportCurrentMxl: 'Export current MXL', checkDetailCount: 'Check details ({count})',
    normal: 'Normal', notEnabled: 'Not enabled', failed: 'Failed', passed: 'Passed', issueCount: '{count} issue(s)',
    storageWorkspace: 'Storage and workspace', recognitionComponents: 'Recognition components', scorePreview: 'Score preview', modelFiles: 'Model files', programFiles: 'Program files',
    runningNormally: 'Running normally · Pigeon Score Scan {version}', criticalIssues: '{count} critical issue(s)', viewAllChecks: 'View all checks ({count})',
    systemCheckFailed: 'System check could not be completed.', trayUnavailable: 'Tray controls are available in the desktop application.',
    textReviewTitle: 'Check score text or dynamics', textReviewMessage: 'Page {page}, near measure {measure}. Compare the scan with the recognized text.',
    measureReviewTitle: 'Check measure content', measureReviewMessage: 'Page {page}, measure {measure}. Compare the highlighted measure with the current output.',
    notationReviewTitle: 'Check notation coverage', notationReviewMessage: 'Page {page}. Compare the highlighted marks with the current output.',
    currentResultChecked: 'Checked — keep current result'
  },
  zh: {
    museScoreInstrumentNote: '导出后请在 MuseScore 中为声部选择正确的乐器，否则可能出现音高显示错误。',
    appToolbar: '应用工具栏', language: '语言', systemStatus: '系统状态', close: '关闭',
    conversionWorkflow: '转换流程', import: '导入', recognize: '识别', output: '输出', runtimeCheck: '运行检查', closeSystemStatus: '关闭系统状态',
    checking: '正在检查…', allChecks: '全部项目', downloadDiagnostics: '下载诊断包', unfinished: '未完成', resumePrevious: '继续上次转换',
    resume: '继续', dismiss: '忽略', importLabel: '01 / 导入', newConversion: '新建转换', importScoreScans: '导入乐谱扫描件',
    dropScans: '拖放扫描件', imagesOrPdf: '图片或 PDF', addImages: '添加图片', addPdf: '添加 PDF', pageOrder: '页面顺序', clear: '清空',
    orientationCorrect: '页面方向正确', noAutoRotation: '不会自动旋转或自动纠斜', conversionDetails: '转换说明', thisOutput: '本次输出',
    oneContinuousScore: '一份连续乐谱', files: '文件', order: '顺序', mergeOrder: '列表从上到下合并为一份 MusicXML', expandByPage: '按页码展开', runtime: '运行',
    quickSettings: '快速设置', conversionSettings: '转换设置', outputFileName: '输出文件名', automaticFileName: '自动命名',
    outputFiles: '输出文件', correctOrientation: '请导入方向正确的乐谱。',
    supported: '支持', supportedScope: '印刷五线谱；单声部乐器谱；常见钢琴谱；同页总谱。各乐器不需要逐音符横向对齐。',
    notSupported: '不支持', unsupportedScope: '手写谱、打击乐、TAB、歌词、和弦符号、数字低音、图形谱。分别扫描的独立分谱请分开转换。',
    scenarioLimits: '场景与功能限制', fullScore: '总谱', fullScoreLimit: '每系统最多 16 行物理谱表，最多 1 个键盘声部', piano: '钢琴',
    pianoLimit: '最多 4 行物理谱表；每谱表每小节最多 8 个独立节奏声部；支持临时谱表、ossia、跨谱表音符与连梁',
    otherParts: '其他乐器', otherPartsLimit: '每谱表仅支持 1 个独立节奏声部', multipleScans: '多份扫描', multipleScansLimit: '仅用于同一份乐谱的连续页面',
    noFiles: '尚未添加文件', noPages: '暂无页面', startConversion: '开始转换', reviewLabel: '03 / 检查', reviewItems: '检查项', backToOutput: '返回输出',
    scanDetail: '扫描件局部', content: '内容', apply: '应用', skip: '跳过', outputLabel: '03 / 输出', conversionResult: '转换结果', checkDetails: '检查明细',
    handleChecks: '处理检查项', downloadMusicXml: '下载 MusicXML', downloadMxl: '下载 MXL', exportMxl: '导出 MXL',
    exportMusicXml: '导出 MusicXML', mxlMuseScoreNote: 'MXL 文件可用 MuseScore 软件查看和编辑', preview: '预览', closePreview: '关闭预览',
    loadingPreview: '正在加载预览…', previewUnavailable: '无法加载预览。', previewPage: '乐谱预览第 {count} 页',
    conversionReport: '转换报告', openMusicXml: '打开 MusicXML', openMxl: '打开 MXL', openFolder: '打开文件夹',
    closeProduct: '关闭 Pigeon Score Scan？', returnToApp: '返回应用', chooseCloseAction: '服务将停止。',
    chooseCloseRunning: '正在进行的转换将停止。', cancel: '取消', exitCompletely: '彻底退出',
    unsupportedFile: '不支持该文件格式。可添加 PNG、JPG、TIFF、BMP、WebP 或 PDF。', mixedFiles: '图片与 PDF 需分开转换。',
    imagesOnly: '此处仅添加图片。', pdfOnly: '此处仅添加 PDF。', differentMode: '当前列表已使用另一种文件类型。清空后再添加。', duplicateFile: '文件已在列表中。',
    emptyFile: '空文件无法添加。', sizeLimit: '本次文件总量不能超过 1 GB。', imageFiles: '图片', inputFiles: '个输入文件', moveUp: '上移', moveDown: '下移', remove: '删除',
    couldNotCreate: '无法创建转换。', couldNotReadStatus: '无法读取转换状态。', couldNotReadResult: '无法读取转换结果。', couldNotReadChecks: '无法读取检查项。',
    couldNotApply: '无法应用修改。', operationFailed: '操作失败。', item: '项', items: '项', confirmResult: '确认识别结果', markReviewed: '标记为已检查',
    keepCurrent: '保留当前结果', conversionIncomplete: '转换未完成', conversionFailed: '转换失败。', cancelled: '转换已取消。', outputChecksPassed: '输出检查通过',
    checksPassed: '检查通过', checksPresent: '存在检查项', checksRemain: '检查项未清除', outputLimited: '需要人工复查', conversionComplete: '转换完成', page: '页', pages: '页',
    pageRecognitionFailed: '页识别失败', pageRecognitionFailedPlural: '页识别失败', checksNotPassed: '存在未通过的输出检查', handleCount: '处理 {count} 个检查项',
    structurePassed: '结构、节奏、记号、文字与文件完整性：通过', outputConditions: '{count} 项输出条件未满足', fileIntegrityPassed: '文件完整性：通过',
    fileIntegrityIncomplete: '文件完整性：未完成', verificationId: '校验标识', exportCurrentMusicXml: '导出当前 MusicXML', exportCurrentMxl: '导出当前 MXL',
    checkDetailCount: '检查明细（{count}）', normal: '正常', notEnabled: '未启用', failed: '失败', passed: '通过', issueCount: '{count} 项异常',
    storageWorkspace: '存储与工作目录', recognitionComponents: '识别组件', scorePreview: '排版预览', modelFiles: '模型文件', programFiles: '程序文件',
    runningNormally: '运行正常 · Pigeon Score Scan {version}', criticalIssues: '{count} 个关键问题', viewAllChecks: '查看全部项目（{count}）',
    systemCheckFailed: '系统检查无法完成。', trayUnavailable: '托盘功能仅在桌面应用中可用。', textReviewTitle: '核对谱面文字或力度',
    textReviewMessage: '第 {page} 页，第 {measure} 小节附近。对照扫描件与识别文字。', measureReviewTitle: '核对小节内容',
    measureReviewMessage: '第 {page} 页，第 {measure} 小节。对照高亮小节与当前输出。', notationReviewTitle: '核对记号覆盖',
    notationReviewMessage: '第 {page} 页。对照高亮记号与当前输出。', currentResultChecked: '已核对，保留当前结果'
  }
};

function t(key, values = {}) {
  const table = messages[state.language] || messages.en;
  let value = table[key] ?? messages.en[key] ?? key;
  Object.entries(values).forEach(([name, replacement]) => { value = value.replaceAll(`{${name}}`, String(replacement)); });
  return value;
}

function applyLanguage(language, persist = true) {
  state.language = language === 'zh' ? 'zh' : 'en';
  document.documentElement.lang = state.language === 'zh' ? 'zh-CN' : 'en';
  $('languageSelect').value = state.language;
  document.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll('[data-i18n-aria-label]').forEach(node => { node.setAttribute('aria-label', t(node.dataset.i18nAriaLabel)); });
  document.querySelectorAll('[data-i18n-alt]').forEach(node => { node.setAttribute('alt', t(node.dataset.i18nAlt)); });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(node => { node.setAttribute('placeholder', t(node.dataset.i18nPlaceholder)); });
  $('fileList').dataset.emptyText = t('noPages');
  if (persist) localStorage.setItem('pigeon-score-scan-language', state.language);
  renderFiles();
  if (!$('reviewPanel').classList.contains('hidden') && state.reviewIssues.length) renderReviewIssue();
  if (!$('resultPanel').classList.contains('hidden') && state.activeJob) showResult(state.activeJob);
  if (!$('closeDialog').classList.contains('hidden')) updateCloseDialogMessage();
}

function apiUrl(path) {
  const url = new URL(path, window.location.origin);
  if (accessToken) url.searchParams.set('token', accessToken);
  return `${url.pathname}${url.search}${url.hash}`;
}
function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (accessToken) headers.set('X-ScoreScan-Token', accessToken);
  return window.fetch(apiUrl(path), { ...options, headers });
}
function showToast(message) {
  const toast = $('toast');
  toast.textContent = message;
  toast.classList.remove('hidden');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.add('hidden'), 3200);
}
function formatBytes(value) {
  if (!Number.isFinite(value) || value < 0) return '—';
  if (value < 1024 * 1024) return `${Math.max(1, Math.round(value / 1024))} KB`;
  if (value >= 1024 * 1024 * 1024) return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  return `${(value / (1024 * 1024)).toFixed(value < 10 * 1024 * 1024 ? 1 : 0)} MB`;
}
function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const remainder = value % 60;
  return hours ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}` : `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}
function localizedError(message, fallbackKey) {
  const text = String(message || '').trim();
  if (state.language === 'zh' && text) return text;
  if (/cancel|取消/i.test(text)) return 'Conversion cancelled.';
  if (/input|输入|file|文件/i.test(text)) return 'The input files could not be processed.';
  if (/space|磁盘|空间/i.test(text)) return 'There is not enough free disk space.';
  if (/orientation|方向/i.test(text)) return 'Check the page orientation and try again.';
  if (/engine|模型|识别/i.test(text)) return 'The recognition service could not complete this conversion.';
  return t(fallbackKey);
}
function setWorkflow(stage) {
  const order = ['import', 'process', 'output'];
  const activeIndex = order.indexOf(stage);
  order.forEach((name, index) => {
    const item = $(`workflow${name[0].toUpperCase()}${name.slice(1)}`);
    item.classList.toggle('active', index === activeIndex);
    item.classList.toggle('complete', index < activeIndex);
    if (index === activeIndex) item.setAttribute('aria-current', 'step');
    else item.removeAttribute('aria-current');
  });
}

function openUtilityPanel(panelId, trigger = null) {
  state.utilityReturnFocus = trigger || document.activeElement;
  $('systemPanel').classList.toggle('hidden', panelId !== 'systemPanel');
  $('panelScrim').classList.remove('hidden');
  document.body.classList.add('panel-open');
  requestAnimationFrame(() => $(panelId).querySelector('.icon-button')?.focus());
}
function syncBodyLock() {
  const overlayOpen = !$("systemPanel").classList.contains('hidden')
    || !$("closeDialog").classList.contains('hidden')
    || !$("previewDialog").classList.contains('hidden');
  document.body.classList.toggle('panel-open', overlayOpen);
}
function closeUtilityPanels() {
  const wasOpen = !$('systemPanel').classList.contains('hidden');
  $('systemPanel').classList.add('hidden');
  $('panelScrim').classList.add('hidden');
  syncBodyLock();
  if (wasOpen && state.utilityReturnFocus instanceof HTMLElement) state.utilityReturnFocus.focus();
  state.utilityReturnFocus = null;
}

function naturalParts(name) { return name.toLocaleLowerCase().split(/(\d+)/).map(x => /^\d+$/.test(x) ? Number(x) : x); }
function naturalCompare(a, b) {
  const aa = naturalParts(a.name), bb = naturalParts(b.name);
  for (let i = 0; i < Math.max(aa.length, bb.length); i++) {
    if (aa[i] === bb[i]) continue;
    if (aa[i] === undefined) return -1;
    if (bb[i] === undefined) return 1;
    return aa[i] < bb[i] ? -1 : 1;
  }
  return 0;
}
function extension(name) { return name.includes('.') ? name.slice(name.lastIndexOf('.')).toLowerCase() : ''; }
function detectMode(files) {
  const modes = new Set();
  files.forEach(file => {
    const suffix = extension(file.name);
    if (suffix === '.pdf') modes.add('pdf');
    else if (['.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'].includes(suffix)) modes.add('images');
    else modes.add('unsupported');
  });
  if (modes.has('unsupported')) return 'unsupported';
  return modes.size === 1 ? Array.from(modes)[0] : 'mixed';
}
function showMessage(text) { $('modeMessage').textContent = text; $('modeMessage').classList.remove('hidden'); }
function hideMessage() { $('modeMessage').classList.add('hidden'); }
function setFiles(newFiles, requestedMode = null) {
  const list = Array.from(newFiles);
  if (!list.length) return;
  const mode = detectMode(list);
  if (mode === 'unsupported') return showMessage(t('unsupportedFile'));
  if (mode === 'mixed') return showMessage(t('mixedFiles'));
  if (requestedMode && mode !== requestedMode) return showMessage(t(requestedMode === 'images' ? 'imagesOnly' : 'pdfOnly'));
  if (state.mode && state.mode !== mode) return showMessage(t('differentMode'));
  const existing = new Set(state.files.map(file => `${file.name}\0${file.size}\0${file.lastModified}`));
  const unique = list.filter(file => {
    const key = `${file.name}\0${file.size}\0${file.lastModified}`;
    if (existing.has(key)) return false;
    existing.add(key);
    return true;
  });
  if (!unique.length) return showMessage(t('duplicateFile'));
  if (unique.some(file => file.size === 0)) return showMessage(t('emptyFile'));
  const totalBytes = [...state.files, ...unique].reduce((sum, file) => sum + file.size, 0);
  if (totalBytes > 1024 * 1024 * 1024) return showMessage(t('sizeLimit'));
  state.mode = mode;
  state.files.push(...unique);
  state.files.sort(naturalCompare);
  renderFiles();
}
function renderFiles() {
  hideMessage();
  const list = $('fileList'); list.textContent = ''; list.dataset.emptyText = t('noPages');
  state.files.forEach((file, index) => {
    const row = document.createElement('div'); row.className = 'file-row';
    const number = document.createElement('div'); number.className = 'file-index'; number.textContent = index + 1;
    const name = document.createElement('div'); name.className = 'file-name'; name.title = file.name; name.textContent = file.name;
    const meta = document.createElement('span'); meta.className = 'file-meta'; meta.textContent = formatBytes(file.size); name.append(' ', meta);
    const actions = document.createElement('div'); actions.className = 'file-actions';
    [['↑','up','moveUp'],['↓','down','moveDown'],[t('remove'),'remove','remove']].forEach(([label, key, actionKey]) => {
      const button = document.createElement('button'); button.textContent = label; button.dataset[key] = index;
      button.title = t(actionKey); button.setAttribute('aria-label', `${t(actionKey)} ${file.name}`);
      if (key === 'up' && index === 0) button.disabled = true;
      if (key === 'down' && index === state.files.length - 1) button.disabled = true;
      actions.appendChild(button);
    });
    row.append(number, name, actions); list.appendChild(row);
  });
  const totalBytes = state.files.reduce((sum, file) => sum + file.size, 0);
  $('fileSummary').textContent = state.files.length
    ? `${state.mode === 'pdf' ? 'PDF' : t('imageFiles')} · ${state.files.length} ${t('inputFiles')} · ${formatBytes(totalBytes)}`
    : t('noFiles');
  $('clearButton').disabled = !state.files.length;
  $('startButton').disabled = !state.files.length;
  $('imageInput').disabled = state.mode === 'pdf';
  $('pdfInput').disabled = state.mode === 'images';
}
$('fileList').addEventListener('click', event => {
  const button = event.target.closest('button'); if (!button) return;
  for (const key of ['up','down','remove']) if (button.dataset[key] !== undefined) {
    const index = Number(button.dataset[key]);
    if (key === 'remove') state.files.splice(index, 1);
    if (key === 'up' && index > 0) [state.files[index - 1], state.files[index]] = [state.files[index], state.files[index - 1]];
    if (key === 'down' && index < state.files.length - 1) [state.files[index + 1], state.files[index]] = [state.files[index], state.files[index + 1]];
    if (!state.files.length) state.mode = null;
    renderFiles(); break;
  }
});
$('imageInput').addEventListener('change', event => { setFiles(event.target.files, 'images'); event.target.value = ''; });
$('pdfInput').addEventListener('change', event => { setFiles(event.target.files, 'pdf'); event.target.value = ''; });
$('clearButton').addEventListener('click', () => { state.files = []; state.mode = null; renderFiles(); });
const dropZone = $('dropZone');
['dragenter','dragover'].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.add('drag'); }));
['dragleave','drop'].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.remove('drag'); }));
dropZone.addEventListener('drop', event => setFiles(event.dataTransfer.files));

function resetRuntimeEstimate() {
  state.jobStartedAt = null; state.etaSeconds = null; state.etaSamples = []; state.etaPhase = null;
  $('elapsedTime').textContent = '00:00'; $('remainingTime').textContent = 'ESTIMATING';
}
function showProgress() {
  $('importPanel').classList.add('hidden'); $('resumePanel').classList.add('hidden');
  $('resultPanel').classList.add('hidden'); $('progressPanel').classList.remove('hidden');
  setWorkflow('process');
  startRuntimeMonitoring();
}
$('startButton').addEventListener('click', async () => {
  if (state.creatingJob || !state.files.length) return;
  state.creatingJob = true; $('startButton').disabled = true; resetRuntimeEstimate();
  const data = new FormData(); state.files.forEach(file => data.append('files', file, file.name));
  data.append('output_name', $('outputName').value.trim());
  showProgress();
  try {
    const response = await apiFetch('/api/jobs', { method: 'POST', body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(localizedError(payload.error, 'couldNotCreate'));
    state.jobId = payload.id; state.pollFailures = 0; state.revision = -1; poll();
  } catch (error) { showFailure(error.message); }
  finally { state.creatingJob = false; }
});

function englishStage(stage) {
  const value = String(stage || '').trim();
  const direct = {
    '等待开始': 'Waiting to start', '恢复上次任务': 'Resuming previous conversion', '正在取消': 'Cancelling', '已取消': 'Cancelled',
    '转换发生错误': 'Conversion failed', '整理输入页面': 'Preparing input pages', '检查扫描与谱面结构': 'Checking scans and score layout',
    '准备本地识别模型': 'Loading recognition models', '合并页面并保持分页与换行': 'Merging pages and preserving layout',
    '验证 MusicXML 与基本排版': 'Validating MusicXML and layout', '转换完成': 'Conversion complete'
  };
  if (direct[value]) return direct[value];
  let match = value.match(/^检查扫描与谱面结构（(\d+)\s*\/\s*(\d+)）$/);
  if (match) return `Checking scans and score layout (${match[1]} / ${match[2]})`;
  match = value.match(/^识别第\s*(\d+)\s*\/\s*(\d+)\s*页的音符、节奏与记号$/);
  if (match) return `Recognizing notes, rhythm, and notation — page ${match[1]} / ${match[2]}`;
  match = value.match(/^识别第\s*(\d+)\s*\/\s*(\d+)\s*页的速度、力度与文字$/);
  if (match) return `Recognizing tempo, dynamics, and text — page ${match[1]} / ${match[2]}`;
  match = value.match(/^识别第\s*(\d+)\s*\/\s*(\d+)\s*页：/);
  if (match) return `Recognizing page ${match[1]} / ${match[2]}`;
  return /[\u3400-\u9fff]/.test(value) ? 'Processing score' : (value || 'Preparing conversion');
}
function updateEta(job) {
  const now = Date.now();
  const parsedStart = Date.parse(job.created_at || '');
  state.jobStartedAt = Number.isFinite(parsedStart) ? parsedStart : (state.jobStartedAt || now);
  const progress = Math.max(0, Math.min(1, Number(job.progress) || 0));
  if (job.status === 'completed') {
    state.etaSeconds = 0; state.etaPhase = 'completed'; return;
  }
  if (job.status !== 'running') { state.etaSeconds = null; return; }
  const phase = progress < 0.20 ? 'preparing' : (progress < 0.80 ? 'recognizing' : 'finalizing');
  if (phase !== state.etaPhase) {
    state.etaPhase = phase;
    state.etaSamples = [];
    state.etaSeconds = null;
  }
  if (phase !== 'recognizing') return;
  const previous = state.etaSamples.at(-1);
  if (previous && progress <= previous.progress + 0.0001) return;
  state.etaSamples.push({ progress, at: now });
  state.etaSamples = state.etaSamples.filter(sample => now - sample.at <= 300000).slice(-30);
  const firstUseful = state.etaSamples.find(sample => now - sample.at >= 15000 && progress - sample.progress >= 0.01);
  if (!firstUseful) return;
  const observedSeconds = Math.max(1, (now - firstUseful.at) / 1000);
  const rate = (progress - firstUseful.progress) / observedSeconds;
  if (rate <= 0) return;
  const recognitionSeconds = Math.max(0, 0.80 - progress) / rate;
  const finalizingBuffer = Math.max(20, Math.min(180, observedSeconds * 0.20));
  const raw = Math.max(15, Math.min(7 * 24 * 3600, (recognitionSeconds + finalizingBuffer) * 1.15));
  if (state.etaSeconds === null) state.etaSeconds = raw;
  else if (raw > state.etaSeconds) state.etaSeconds = state.etaSeconds * 0.35 + raw * 0.65;
  else state.etaSeconds = state.etaSeconds * 0.85 + raw * 0.15;
}
function roundedEta(seconds) {
  const step = seconds <= 120 ? 15 : (seconds <= 600 ? 30 : 60);
  return Math.max(step, Math.ceil(seconds / step) * step);
}
function updateRuntimeClock() {
  const now = Date.now();
  const elapsed = state.jobStartedAt ? Math.max(0, (now - state.jobStartedAt) / 1000) : 0;
  $('elapsedTime').textContent = formatDuration(elapsed);
  if (state.activeJob?.status === 'completed') $('remainingTime').textContent = '00:00';
  else if (state.activeJob?.queue_position) $('remainingTime').textContent = 'WAITING';
  else if (state.etaPhase === 'finalizing') $('remainingTime').textContent = 'FINALIZING';
  else if (state.etaSeconds === null) $('remainingTime').textContent = 'ESTIMATING';
  else $('remainingTime').textContent = `~${formatDuration(roundedEta(state.etaSeconds))}`;
}
async function updateResourceUsage() {
  if ($('progressPanel').classList.contains('hidden')) return;
  try {
    const response = await apiFetch('/api/runtime');
    const payload = await response.json();
    if (!response.ok) return;
    $('systemCpu').textContent = Number.isFinite(payload.system_cpu_percent) ? `${Math.round(payload.system_cpu_percent)}%` : '—';
    $('systemMemory').textContent = Number.isFinite(payload.memory_percent) ? `${Math.round(payload.memory_percent)}%` : '—';
    $('serviceMemory').textContent = formatBytes(payload.process_memory_bytes);
  } catch (_) {}
}
function startRuntimeMonitoring() {
  if (!state.clockTimer) state.clockTimer = setInterval(updateRuntimeClock, 1000);
  if (!state.resourceTimer) state.resourceTimer = setInterval(updateResourceUsage, 2000);
  updateRuntimeClock(); updateResourceUsage();
}
function stopRuntimeMonitoring() {
  clearInterval(state.clockTimer); clearInterval(state.resourceTimer);
  state.clockTimer = null; state.resourceTimer = null;
}

async function poll() {
  if (!state.jobId) return;
  try {
    const query = state.revision >= 0 ? `?compact=1&after=${state.revision}&wait=15` : '?compact=1';
    const response = await apiFetch(`/api/jobs/${state.jobId}${query}`); const job = await response.json();
    if (!response.ok) throw new Error(localizedError(job.error, 'couldNotReadStatus'));
    state.pollFailures = 0; state.activeJob = job;
    state.revision = Number.isFinite(job.revision) ? job.revision : state.revision;
    $('stageText').textContent = job.queue_position ? 'Waiting for an available processing slot' : englishStage(job.stage);
    $('queueText').textContent = job.queue_position ? `${Math.max(0, job.queue_position - 1)} job(s) ahead` : '';
    $('pageText').textContent = job.total_pages ? `PAGE ${Math.max(1, job.current_page || 1)} / ${job.total_pages}` : 'WAITING FOR PAGE INFORMATION';
    const percentage = Math.round((job.progress || 0) * 100);
    $('progressBar').style.width = `${percentage}%`; $('progressNumber').textContent = `${percentage}%`;
    $('jobProgressTrack').setAttribute('aria-valuenow', String(percentage));
    $('cancelButton').disabled = job.status === 'cancelling';
    updateEta(job); updateRuntimeClock();
    if (job.status === 'completed') {
      const fullResponse = await apiFetch(`/api/jobs/${state.jobId}`); const fullJob = await fullResponse.json();
      if (!fullResponse.ok) throw new Error(localizedError(fullJob.error, 'couldNotReadResult'));
      state.activeJob = fullJob; return showResult(fullJob);
    }
    if (job.status === 'cancelled') return showFailure(t('cancelled'));
    if (job.status === 'failed') return showFailure(localizedError(job.error, 'conversionFailed'));
    state.pollTimer = setTimeout(poll, 60);
  } catch (error) {
    state.pollFailures += 1;
    if (state.pollFailures >= 3) {
      $('stageText').textContent = 'Reconnecting to local service';
      $('pageText').textContent = 'BACKGROUND PROCESSING CONTINUES';
    }
    state.pollTimer = setTimeout(poll, Math.min(5000, 1200 + state.pollFailures * 500));
  }
}
$('cancelButton').addEventListener('click', async () => {
  if (!state.jobId) return;
  if (!state.cancelArmed) {
    state.cancelArmed = true; $('cancelButton').textContent = 'Confirm cancel'; $('cancelButton').classList.add('cancel-armed');
    setTimeout(() => { state.cancelArmed = false; $('cancelButton').textContent = 'Cancel conversion'; $('cancelButton').classList.remove('cancel-armed'); }, 4000);
    return;
  }
  state.cancelArmed = false; $('cancelButton').disabled = true; $('cancelButton').textContent = 'Cancelling…';
  await apiFetch(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' });
});

function qualityLabel(stateName) {
  if (stateName === 'verified') return [t('outputChecksPassed'), 'ok'];
  if (stateName === 'verified_after_review') return [t('checksPassed'), 'ok'];
  if (stateName === 'review_recommended') return [t('outputLimited'), 'review'];
  if (stateName === 'reviewed_with_warnings') return [t('outputLimited'), 'review'];
  if (stateName === 'best_effort') return [t('outputLimited'), 'review'];
  return [t('conversionComplete'), 'review'];
}
function userFacingWarning(message) {
  const text = String(message || '').trim();
  const pageNumber = text.match(/第\s*(\d+)\s*页/)?.[1];
  const pagePrefix = state.language === 'zh' ? (pageNumber ? `第 ${pageNumber} 页：` : '') : (pageNumber ? `Page ${pageNumber}: ` : '');
  if (/自动校正了\s*\d+\s*处/.test(text)) return null;
  if (state.language === 'en') {
    if (/语义记号检测器未获准启用|语义记号检测失败/.test(text)) return `${pagePrefix}Notation model unavailable.`;
    if (/源图连梁关系恢复失败/.test(text)) return `${pagePrefix}Beam validation incomplete.`;
    if (/文字识别进程失败|文字标记无法写入|文字角色模型不可用/.test(text)) return `${pagePrefix}Text validation incomplete.`;
    if (/小节线.*模型不可用/.test(text)) return `${pagePrefix}Barline validation incomplete.`;
    if (/多个识别候选得分接近/.test(text)) return `${pagePrefix}Recognition result is unstable.`;
    if (/识别模型初始化异常/.test(text)) return 'Recognition model initialization failed.';
    if (/没有检测到 homr 识别引擎/.test(text)) return 'Recognition engine unavailable.';
    if (/小节时值疑点/.test(text)) return `${pagePrefix}Rhythm validation did not pass.`;
    if (/延音线结构疑点/.test(text)) return `${pagePrefix}Tie validation did not pass.`;
    if (/连音线结构疑点/.test(text)) return `${pagePrefix}Slur validation did not pass.`;
    if (/装饰音审计失败|记号检测失败|发夹关系审计失败|连奏线关系审计失败|记号覆盖审计失败/.test(text)) return `${pagePrefix}Notation validation incomplete.`;
    if (/语义符号审计失败|位置级语义符号审计/.test(text)) return `${pagePrefix}Symbol placement validation incomplete.`;
    if (/漏识别/.test(text)) return `${pagePrefix}Notation may be missing.`;
    if (/模型资源完整性|模型资源审计/.test(text)) return 'Recognition model files are incomplete.';
    if (/连续三连音.*事务/.test(text)) return 'Triplet correction was not applied.';
    return `${pagePrefix}A validation check did not pass.`;
  }
  const prefix = pageNumber ? `第 ${pageNumber} 页：` : '';
  if (/语义记号检测器未获准启用|语义记号检测失败/.test(text)) return `${prefix}符号模型未启用`;
  if (/源图连梁关系恢复失败/.test(text)) return `${prefix}连梁检查未完成`;
  if (/文字识别进程失败|文字标记无法写入|文字角色模型不可用/.test(text)) return `${prefix}文字检查未完成`;
  if (/小节线.*模型不可用/.test(text)) return `${prefix}小节线检查未完成`;
  if (/多个识别候选得分接近/.test(text)) return `${prefix}识别结果不稳定`;
  if (/识别模型初始化异常/.test(text)) return '识别模型初始化失败';
  if (/没有检测到 homr 识别引擎/.test(text)) return '识别引擎不可用';
  if (/小节时值疑点/.test(text)) return `${prefix}节奏检查未通过`;
  if (/延音线结构疑点/.test(text)) return `${prefix}延音线检查未通过`;
  if (/连音线结构疑点/.test(text)) return `${prefix}连音线检查未通过`;
  if (/装饰音审计失败|记号检测失败|发夹关系审计失败|连奏线关系审计失败|记号覆盖审计失败/.test(text)) return `${prefix}记号检查未完成`;
  if (/语义符号审计失败|位置级语义符号审计/.test(text)) return `${prefix}符号位置检查未完成`;
  if (/漏识别/.test(text)) return `${prefix}记号可能缺失`;
  if (/模型资源完整性|模型资源审计/.test(text)) return '识别模型文件不完整';
  if (/连续三连音.*事务/.test(text)) return '三连音修正未应用';
  return text.replaceAll('源扫描', '扫描件').replaceAll('审计', '检查').replaceAll('静默', '').replace(/，将启用.*$/, '').replace(/，程序仍会.*$/, '').replace(/；已保留分页并继续$/, '').replace(/：(?:RuntimeError|ValueError|OSError|Exception):.*$/, '');
}
function pendingReviewIssues(job) { return (job.review_issues || []).filter(issue => issue.status !== 'resolved'); }
async function loadReview(job = state.activeJob) {
  if (!job) return;
  try {
    const response = await apiFetch(`/api/jobs/${job.id}/review`); const payload = await response.json();
    if (!response.ok) throw new Error(localizedError(payload.error, 'couldNotReadChecks'));
    state.reviewIssues = (payload.issues || []).filter(issue => issue.status !== 'resolved'); state.reviewIndex = 0;
    if (!state.reviewIssues.length) return showResult(state.activeJob || job);
    $('resultPanel').classList.add('hidden'); $('reviewPanel').classList.remove('hidden'); setWorkflow('output'); renderReviewIssue();
  } catch (error) { showToast(error.message); }
}
function localizedReview(issue) {
  if (state.language === 'zh') return [issue.title || t('confirmResult'), issue.message || ''];
  const values = { page: issue.page_index || '—', measure: issue.global_measure_number || '—' };
  if (issue.category === 'music_text') return [t('textReviewTitle'), t('textReviewMessage', values)];
  if (issue.category === 'measure_consensus') return [t('measureReviewTitle'), t('measureReviewMessage', values)];
  if (issue.category === 'notation_coverage') return [t('notationReviewTitle'), t('notationReviewMessage', values)];
  return [t('confirmResult'), `Page ${values.page}. Compare the scan with the current output.`];
}
function renderReviewIssue() {
  const issue = state.reviewIssues[state.reviewIndex];
  if (!issue) { $('reviewPanel').classList.add('hidden'); if (state.activeJob) showResult(state.activeJob); return; }
  $('reviewProgress').textContent = `${state.reviewIndex + 1} / ${state.reviewIssues.length}`;
  const [title, message] = localizedReview(issue); $('reviewTitle').textContent = title; $('reviewMessage').textContent = message;
  $('reviewCrop').src = apiUrl(`/api/jobs/${state.jobId}/review/${issue.id}/crop?ts=${Date.now()}`);
  $('reviewCustom').value = issue.suggested_value || issue.raw_value || '';
  const needsValue = issue.requires_value !== false; $('reviewCustomLabel').classList.toggle('hidden', !needsValue);
  $('reviewApply').textContent = needsValue ? t('apply') : t('markReviewed'); $('reviewIgnore').textContent = needsValue ? t('keepCurrent') : t('skip');
  const options = $('reviewOptions'); options.textContent = '';
  (issue.options || []).forEach((value, index) => {
    const displayValue = state.language === 'en' && value === '已核对，保留当前结果' ? t('currentResultChecked') : value;
    const label = document.createElement('label'); label.className = 'review-option';
    const radio = document.createElement('input'); radio.type = 'radio'; radio.name = 'reviewChoice'; radio.value = value;
    if (index === 0) radio.checked = true;
    radio.addEventListener('change', () => { $('reviewCustom').value = value; });
    const span = document.createElement('span'); span.textContent = displayValue; label.append(radio, span); options.appendChild(label);
  });
}
async function resolveCurrentReview(ignore = false) {
  const issue = state.reviewIssues[state.reviewIndex]; if (!issue) return;
  const value = issue.requires_value === false ? '' : $('reviewCustom').value.trim();
  $('reviewApply').disabled = true; $('reviewIgnore').disabled = true;
  try {
    const response = await apiFetch(`/api/jobs/${state.jobId}/review/${issue.id}`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ value, ignore }) });
    const payload = await response.json(); if (!response.ok) throw new Error(localizedError(payload.error, 'couldNotApply'));
    if (payload.job) state.activeJob = payload.job; state.reviewIndex += 1; renderReviewIssue();
  } catch (error) { showToast(error.message); }
  finally { $('reviewApply').disabled = false; $('reviewIgnore').disabled = false; }
}
$('reviewApply').addEventListener('click', () => resolveCurrentReview(false));
$('reviewIgnore').addEventListener('click', () => resolveCurrentReview(true));
$('reviewBack').addEventListener('click', () => { $('reviewPanel').classList.add('hidden'); if (state.activeJob) showResult(state.activeJob); });
$('reviewButton').addEventListener('click', () => loadReview());

async function showPreviewDialog() {
  if (!state.previewUrl) return;
  $('previewDialog').classList.remove('hidden');
  syncBodyLock();
  requestAnimationFrame(() => $('previewCloseButton').focus());
  if (state.previewLoadedUrl === state.previewUrl) return;
  const viewport = $('previewViewport');
  viewport.textContent = '';
  const loading = document.createElement('div'); loading.className = 'preview-message'; loading.textContent = t('loadingPreview'); viewport.appendChild(loading);
  try {
    const response = await apiFetch(state.previewUrl); const payload = await response.json();
    if (!response.ok || !Number.isInteger(payload.page_count) || payload.page_count < 1) throw new Error();
    viewport.textContent = '';
    for (let page = 1; page <= payload.page_count; page += 1) {
      const section = document.createElement('section'); section.className = 'preview-page';
      const image = document.createElement('img'); image.loading = page === 1 ? 'eager' : 'lazy'; image.alt = t('previewPage', {count: page});
      image.src = apiUrl(`${state.previewUrl}/${page}`); section.appendChild(image); viewport.appendChild(section);
    }
    state.previewLoadedUrl = state.previewUrl;
  } catch (_) {
    viewport.textContent = '';
    const message = document.createElement('div'); message.className = 'preview-message'; message.textContent = t('previewUnavailable'); viewport.appendChild(message);
  }
}
function hidePreviewDialog() {
  $('previewDialog').classList.add('hidden');
  syncBodyLock();
  $('previewButton').focus();
}
$('previewButton').addEventListener('click', showPreviewDialog);
$('previewCloseButton').addEventListener('click', hidePreviewDialog);
$('previewDialog').addEventListener('click', event => { if (event.target === $('previewDialog')) hidePreviewDialog(); });

function showResult(job) {
  clearTimeout(state.pollTimer); stopRuntimeMonitoring();
  $('progressPanel').classList.add('hidden'); $('reviewPanel').classList.add('hidden'); $('resultPanel').classList.remove('hidden'); setWorkflow('output');
  $('downloadActions').classList.remove('hidden'); $('downloadReport').classList.remove('hidden');
  ['downloadXml','downloadMxl','openMxlButton','openFolderButton'].forEach(id => $(id).classList.remove('hidden'));
  const fallback = (job.pages || []).filter(page => page.omr_status === 'fallback').length; const pending = pendingReviewIssues(job);
  $('resultStatus').className = 'result-status';
  if (fallback) { $('resultStatus').classList.add('partial'); $('resultStatus').textContent = `${t('conversionIncomplete')} · ${fallback} ${fallback === 1 ? t('pageRecognitionFailed') : t('pageRecognitionFailedPlural')}`; }
  else if (job.production_ready) $('resultStatus').textContent = `${t('conversionComplete')} · ${job.pages.length} ${t('pages')} · ${t('outputChecksPassed')}`;
  else { $('resultStatus').classList.add('needs-review'); $('resultStatus').textContent = `${t('conversionComplete')} · ${t('checksNotPassed')}`; }
  $('reviewButton').classList.toggle('hidden', pending.length === 0); $('reviewButton').textContent = pending.length ? t('handleCount', {count: pending.length}) : t('handleChecks');
  const [label, css] = qualityLabel(job.quality_state); const badge = $('qualityBadge'); badge.textContent = label; badge.className = `quality-badge ${css}`;
  $('qualityHelp').textContent = job.production_ready ? t('structurePassed') : (job.release_blocker_count ? t('outputConditions', {count: job.release_blocker_count}) : t('checksNotPassed'));
  $('qualityHelp').classList.remove('hidden'); const integrity = $('integrityStatus');
  if (job.artifact_bundle_id) { integrity.textContent = t('fileIntegrityPassed'); integrity.title = `${t('verificationId')}: ${job.artifact_bundle_id}`; integrity.className = 'integrity-status ok'; }
  else { integrity.textContent = t('fileIntegrityIncomplete'); integrity.className = 'integrity-status warning'; }
  const box = $('warningList'); box.textContent = '';
  const warnings = Array.from(new Set((job.warnings || []).map(userFacingWarning).filter(Boolean)));
  warnings.forEach(warning => { const div = document.createElement('div'); div.className = 'warning'; div.textContent = warning; box.appendChild(div); });
  $('warningDetails').classList.toggle('hidden', !warnings.length); $('warningSummary').textContent = t('checkDetailCount', {count: warnings.length});
  $('downloadXml').href = apiUrl(`/api/jobs/${job.id}/download/musicxml`); $('downloadMxl').href = apiUrl(`/api/jobs/${job.id}/download/mxl`);
  $('downloadReport').href = apiUrl(`/api/jobs/${job.id}/download/report`); $('openMxlButton').dataset.job = job.id; $('openFolderButton').dataset.job = job.id;
  $('downloadMxl').textContent = t('exportMxl'); $('downloadXml').textContent = t('exportMusicXml');
  state.previewUrl = job.preview_svg ? `/api/jobs/${job.id}/preview` : null;
  if (state.previewLoadedUrl !== state.previewUrl) state.previewLoadedUrl = null;
  $('previewButton').classList.toggle('hidden', !state.previewUrl);
}
function showFailure(message) {
  clearTimeout(state.pollTimer); stopRuntimeMonitoring();
  $('progressPanel').classList.add('hidden'); $('reviewPanel').classList.add('hidden'); $('resultPanel').classList.remove('hidden'); setWorkflow('output');
  $('reviewButton').classList.add('hidden'); $('resultStatus').className = 'result-status error'; $('resultStatus').textContent = `${t('conversionIncomplete')}: ${message}`;
  $('qualityBadge').classList.add('hidden'); $('qualityHelp').classList.add('hidden'); $('integrityStatus').classList.add('hidden');
  $('warningDetails').classList.add('hidden'); $('warningList').textContent = ''; $('downloadActions').classList.add('hidden');
  $('openMxlButton').classList.add('hidden'); $('openFolderButton').classList.add('hidden');
}
async function postAction(url) {
  try { const response = await apiFetch(url, { method: 'POST' }); const payload = await response.json(); if (!response.ok) throw new Error(localizedError(payload.error, 'operationFailed')); }
  catch (error) { showToast(error.message); }
}
$('openMxlButton').addEventListener('click', () => { const id = $('openMxlButton').dataset.job; if (id) postAction(`/api/jobs/${id}/open/mxl`); });
$('openFolderButton').addEventListener('click', () => { const id = $('openFolderButton').dataset.job; if (id) postAction(`/api/jobs/${id}/open/folder`); });
$('newJobButton').addEventListener('click', () => location.reload());

function updateCloseDialogMessage() {
  const running = state.activeJob && ['running','queued','cancelling'].includes(state.activeJob.status);
  $('closeDialogMessage').textContent = t(running ? 'chooseCloseRunning' : 'chooseCloseAction');
}
function showCloseDialog() {
  updateCloseDialogMessage(); $('closeDialog').classList.remove('hidden'); syncBodyLock();
  requestAnimationFrame(() => $('cancelCloseButton').focus());
}
function hideCloseDialog() {
  $('closeDialog').classList.add('hidden');
  syncBodyLock();
  $('quitButton').focus();
}
window.PigeonScoreScan = { showCloseDialog };
$('quitButton').addEventListener('click', showCloseDialog);
$('cancelCloseButton').addEventListener('click', hideCloseDialog);
$('closeDialog').addEventListener('click', event => { if (event.target === $('closeDialog')) hideCloseDialog(); });
$('exitCompletelyButton').addEventListener('click', async () => {
  $('exitCompletelyButton').disabled = true;
  if (window.pywebview?.api?.exit_completely) { await window.pywebview.api.exit_completely(); return; }
  await apiFetch('/api/shutdown', { method: 'POST' }); window.close();
  document.body.innerHTML = `<main class="shell"><section class="workspace-card"><header class="workspace-heading"><div><span class="section-label">STATUS</span><h2>Pigeon Score Scan has exited</h2></div></header></section></main>`;
});

async function discoverRecent() {
  try {
    const response = await apiFetch('/api/jobs'); const payload = await response.json();
    const job = (payload.jobs || []).find(item => ['queued','running','interrupted','cancelling'].includes(item.status)); if (!job) return;
    $('resumeText').textContent = `${job.source_files.length} ${t('inputFiles')} · ${englishStage(job.stage)}`; $('resumePanel').classList.remove('hidden');
    $('resumeButton').onclick = () => { state.jobId = job.id; state.revision = -1; resetRuntimeEstimate(); showProgress(); poll(); };
    $('dismissResumeButton').onclick = () => $('resumePanel').classList.add('hidden');
  } catch (_) {}
}
async function showSystemCheck() {
  openUtilityPanel('systemPanel', $('systemCheckButton')); $('systemSummary').textContent = t('checking');
  $('systemHighlights').textContent = ''; $('systemChecks').textContent = '';
  try {
    const response = await apiFetch('/api/system-check'); const payload = await response.json();
    $('systemSummary').textContent = payload.ok ? t('runningNormally', {version: payload.version}) : t('criticalIssues', {count: payload.critical_failures || 0});
    const checks = payload.checks || [];
    const groups = [
      ['storageWorkspace', item => item.key.startsWith('filesystem:')], ['recognitionComponents', item => item.key.startsWith('module:') || item.key.startsWith('accelerator:')],
      ['scorePreview', item => item.key.startsWith('render:')], ['modelFiles', item => item.key.startsWith('model:') || item.key.startsWith('models:')],
      ['programFiles', item => item.key.startsWith('bootstrap:') || item.key.startsWith('release:')]
    ];
    groups.forEach(([nameKey, matches]) => {
      const items = checks.filter(matches); if (!items.length) return;
      const failures = items.filter(item => !item.ok); const criticalFailures = failures.filter(item => item.critical !== false);
      const stateClass = criticalFailures.length ? 'fail' : (failures.length ? 'optional' : 'ok');
      const card = document.createElement('div'); card.className = `system-highlight ${stateClass}`;
      const title = document.createElement('strong'); title.textContent = t(nameKey); const status = document.createElement('span');
      status.textContent = criticalFailures.length ? t('issueCount', {count: criticalFailures.length}) : (failures.length ? `${failures.length} ${t('notEnabled')}` : t('normal'));
      card.append(title, status); $('systemHighlights').appendChild(card);
    });
    $('systemDetailsSummary').textContent = t('viewAllChecks', {count: checks.length}); $('systemDetails').open = !payload.ok;
    checks.forEach(item => {
      const optional = !item.ok && item.critical === false; const row = document.createElement('div');
      row.className = `system-check ${item.ok ? 'ok' : (optional ? 'optional' : 'fail')}`;
      const text = document.createElement('span'); text.textContent = String(item.key || 'check').replaceAll(':', ' · ');
      const status = document.createElement('span'); status.className = 'status'; status.textContent = item.ok ? t('passed') : (optional ? t('notEnabled') : t('failed'));
      row.append(text, status); $('systemChecks').appendChild(row);
    });
  } catch (_) { $('systemSummary').textContent = t('systemCheckFailed'); }
}
$('systemCheckButton').addEventListener('click', showSystemCheck);
$('systemCloseButton').addEventListener('click', closeUtilityPanels);
$('panelScrim').addEventListener('click', closeUtilityPanels);
$('languageSelect').addEventListener('change', event => applyLanguage(event.target.value));
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  if (!$('previewDialog').classList.contains('hidden')) hidePreviewDialog();
  else if (!$('closeDialog').classList.contains('hidden')) hideCloseDialog();
  else closeUtilityPanels();
});

$('diagnosticsButton').href = apiUrl('/api/diagnostics');
const savedLanguage = localStorage.getItem('pigeon-score-scan-language');
applyLanguage(savedLanguage === 'zh' ? 'zh' : 'en', false);
setWorkflow('import'); discoverRecent();
