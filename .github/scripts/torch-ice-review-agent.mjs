#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const TRUSTED = new Set(['OWNER', 'MEMBER', 'COLLABORATOR']);
const HISTORY_MAX_ITEMS = 30;
const HISTORY_MAX_CHARS = 32_000;
const DIFF_MAX_CHARS = 160_000;
const COMMAND_MAX_CHARS = 2_000;
const PR_TITLE_MAX_CHARS = 2_000;
const PR_BODY_MAX_CHARS = 8_000;
const METADATA_MAX_CHARS = PR_TITLE_MAX_CHARS + PR_BODY_MAX_CHARS + 256;
const FILES_MAX_CHARS = 16_000;
const FILE_CONTEXT_MAX_CHARS = 24_000;
const EXPLORATION_MAX_CALLS = 64;
const EXPLORATION_MAX_CHARS = 96_000;
const EXPLORATION_FILE_MAX_CHARS = 12_000;
const EXPLORATION_SEARCH_MAX_FILES = 200;
const EXPLORATION_SEARCH_MAX_MATCHES = 40;
const OUTPUT_MAX_CHARS = 32_000;
const OPENAI_REQUEST_TIMEOUT_MS = 180_000;
// Finish model work before the 20-minute workflow timeout so failure handling
// can still post its advisory comment.
const REVIEW_DEADLINE_MS = 14 * 60 * 1_000;
// Responses has no input-token limit parameter. This ceiling targets roughly
// 64k input tokens while giving the current diff its own non-competing budget.
const INPUT_MAX_CHARS = 256_000;
const INITIAL_MAX_OUTPUT_TOKENS = 6_144;
const RETRY_MAX_OUTPUT_TOKENS = 8_192;
const FORCE_COOLDOWN_MS = 15 * 60 * 1_000;
const FORCE_MAX_PER_HEAD = 2;
const BLOCKED_LABELS = new Set(['security', 'private', 'do-not-ai-review']);
const BOT_MARKER = '<!-- torch-ice-review-agent: success head_sha=';
const FINAL_MARKER = /(?:^|\r?\n)<!-- torch-ice-review-agent: [^\r\n]* -->\r?$/;

const EXPLORATION_TOOLS = [
  {
    type: 'function', name: 'list_files', description: 'List up to 100 files in the read-only PR base or head snapshot.', strict: true,
    parameters: { type: 'object', additionalProperties: false, required: ['snapshot', 'path', 'limit'], properties: {
      snapshot: { type: 'string', enum: ['base', 'head'] }, path: { type: ['string', 'null'] }, limit: { type: ['integer', 'null'], minimum: 1, maximum: 100 },
    } },
  },
  {
    type: 'function', name: 'read_file', description: 'Read at most 200 lines from one regular file in the read-only PR base or head snapshot.', strict: true,
    parameters: { type: 'object', additionalProperties: false, required: ['snapshot', 'path', 'line_start', 'line_end'], properties: {
      snapshot: { type: 'string', enum: ['base', 'head'] }, path: { type: 'string' }, line_start: { type: ['integer', 'null'], minimum: 1 }, line_end: { type: ['integer', 'null'], minimum: 1 },
    } },
  },
  {
    type: 'function', name: 'search_code', description: 'Search literal text in up to 200 files in the read-only PR base or head snapshot.', strict: true,
    parameters: { type: 'object', additionalProperties: false, required: ['snapshot', 'query', 'path'], properties: {
      snapshot: { type: 'string', enum: ['base', 'head'] }, query: { type: 'string', minLength: 1, maxLength: 160 }, path: { type: ['string', 'null'] },
    } },
  },
];

export function parseReviewCommand(body = '') {
  const match = /(?:^|\r?\n)[ \t]*@torch-ice-review-agent(?=$|[ \t])(?:[ \t]*(.*))?/.exec(body);
  if (!match) return null;
  const commandEnd = match.index + match[0].length;
  const firstLinePrompt = match[1] ?? '';
  const remainder = body.slice(commandEnd);
  const prompt = `${firstLinePrompt}${remainder}`.trim();
  const force = /(?:^|\s)--force(?=$|\s)/.test(prompt);
  return { force, prompt: prompt.replace(/(?:^|\s)--force(?=$|\s)/g, ' ').trim() };
}

function changedPath(file) {
  return typeof file === 'string' ? file : file.filename ?? file.path ?? '';
}

export function selectReviewMode(files = []) {
  if (files.some((file) => /^frameworks\/[^/]+\/[^/]+\//.test(changedPath(file)))) return 'framework-assessment';
  if (files.some((file) => file.status === 'added' && /^skills\/[^/]+\/SKILL\.md$/.test(changedPath(file)))) return 'framework-assessment';

  const addedFrameworkFiles = new Map();
  for (const file of files) {
    const match = /^frameworks\/([^/]+)\/(EVAL|checklist)\.md$/.exec(changedPath(file));
    if (!match || file.status !== 'added') continue;
    const filesForFramework = addedFrameworkFiles.get(match[1]) ?? new Set();
    filesForFramework.add(match[2]);
    addedFrameworkFiles.set(match[1], filesForFramework);
  }
  return [...addedFrameworkFiles.values()].some((filesForFramework) => filesForFramework.has('EVAL') && filesForFramework.has('checklist'))
    ? 'framework-assessment'
    : 'general';
}

export function isSuccessfulReviewResult(comment, headSha) {
  if (comment?.user?.login !== 'github-actions[bot]') return false;
  const escaped = String(headSha).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(?:^|\\r?\\n)${BOT_MARKER.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}${escaped}(?: attempt=force)? -->\\r?$`).test(String(comment.body ?? ''));
}

export function extractResponseText(response) {
  const sdkText = typeof response?.output_text === 'string' ? response.output_text.trim() : '';
  if (sdkText) return sdkText;

  const parts = Array.isArray(response?.output) ? response.output.flatMap((item) => {
    if (item?.type !== 'message' || !Array.isArray(item.content)) return [];
    return item.content
      .filter((part) => part?.type === 'output_text' && typeof part.text === 'string')
      .map((part) => part.text);
  }) : [];
  return parts.join('\n').trim();
}

export function formatDeduplicationComment(headSha) {
  return `No changes have been made since the previous successful review of this PR head, so no new review was run.\n\n<!-- torch-ice-review-agent: skipped head_sha=${headSha} -->`;
}

export function redactSensitiveText(value) {
  let count = 0;
  const redact = (text, pattern) => text.replace(pattern, () => { count += 1; return '[REDACTED]'; });
  let text = String(value ?? '');
  text = redact(text, /-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----[\s\S]*?-----END(?: [A-Z]+)? PRIVATE KEY-----/g);
  text = redact(text, /\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk(?:-proj)?-[A-Za-z0-9_-]{20,})\b/g);
  text = redact(text, /\bAKIA[0-9A-Z]{16}\b/g);
  return { text, count };
}

export function sanitizeReviewOutput(value) {
  const output = String(value ?? '').trim()
    .replace(/<!--\s*torch-ice-review-agent:[\s\S]*?-->/gi, '')
    .replace(/<picture\b[^>]*>[\s\S]*?<\/picture>/gi, '[external image omitted]')
    .replace(/<img\b[^>]*>/gi, '[external image omitted]')
    .replace(/!\[[^\]]*\][ \t]*(?:\([^\r\n)]*\)|\[[^\r\n\]]*\])?/g, '[external image omitted]')
    .replace(/@(?=[A-Za-z0-9-]{1,39}\b)/g, '@\u200B')
    .trim();
  if (output.length > OUTPUT_MAX_CHARS) throw new Error('OpenAI returned review text that exceeded the safe output limit.');
  return output;
}

function escapeUntrustedSection(value) {
  return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function buildReviewInput({ commandPrompt, pr, headSha, files, fileContext = '', diff, history, checklist, reviewMode = 'general' }) {
  const raw = {
    command: String(commandPrompt ?? '') || '(No additional prompt.)',
    metadata: JSON.stringify({ number: pr.number, title: truncate(pr.title, PR_TITLE_MAX_CHARS), body: truncate(pr.body, PR_BODY_MAX_CHARS), head_sha: headSha }),
    files: files.map((file) => `${file.filename} (+${file.additions}/-${file.deletions})`).join('\n'),
    fileContext: String(fileContext ?? ''),
    history: formatHistory(history),
    diff: String(diff ?? ''),
  };
  const escaped = Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, escapeUntrustedSection(value)]));
  const values = {
    command: truncate(escaped.command, COMMAND_MAX_CHARS), metadata: truncate(escaped.metadata, METADATA_MAX_CHARS),
    files: truncate(escaped.files, FILES_MAX_CHARS), history: truncate(escaped.history, HISTORY_MAX_CHARS),
    fileContext: truncate(escaped.fileContext, FILE_CONTEXT_MAX_CHARS), diff: truncate(escaped.diff, DIFF_MAX_CHARS),
    checklist: reviewMode === 'framework-assessment' ? String(checklist ?? '') : '',
  };
  const section = (tag, value) => `<${tag}>\n${value}\n</${tag}>`;
  const frameworkCategories = 'Skill Structure\nFramework Nesting\nScoring Consistency\nDispatch & Orchestration';
  const parts = [section('trusted_review_dispatch', reviewMode), section('untrusted_command', values.command), section('untrusted_pr_metadata', values.metadata),
    section('untrusted_changed_files', values.files),
    ...(values.fileContext ? [section('untrusted_pr_file_context', values.fileContext)] : []),
    ...(reviewMode === 'framework-assessment' ? [section('trusted_framework_assessment_categories', frameworkCategories)] : []),
    ...(reviewMode === 'framework-assessment' ? [section('trusted_architecture_checklist', values.checklist)] : []),
    section('untrusted_review_history', values.history), section('untrusted_pr_diff', values.diff)];
  const input = parts.join('\n\n');
  if (input.length > INPUT_MAX_CHARS) throw new Error('Review input exceeded its fixed section budgets.');
  return { input, truncated: Object.keys(raw).some((key) => values[key] !== escaped[key]) };
}

export function shouldRetryForOutputLimit(response) {
  return response?.status === 'incomplete' && response?.incomplete_details?.reason === 'max_output_tokens';
}

export function isAllowedGithubApiUrl(value, apiUrl = process.env.GITHUB_API_URL ?? 'https://api.github.com') {
  try { return new URL(value).origin === new URL(apiUrl).origin && new URL(value).protocol === 'https:'; } catch { return false; }
}

function truncate(value, limit) {
  const text = String(value ?? '');
  const marker = '\n[truncated]';
  return text.length <= limit ? text : `${text.slice(0, Math.max(0, limit - marker.length))}${marker}`;
}

function itemFromComment(comment, kind, changedPaths) {
  const body = String(comment.body ?? '');
  const isReviewAgentBot = comment.user?.login === 'github-actions[bot]' && FINAL_MARKER.test(body);
  const quotedFullDiff = /(?:^|\n)>? ?diff --git |(?:^|\n)```diff/.test(body);
  if (!body || (!isReviewAgentBot && quotedFullDiff)) return null;
  const path = comment.path ?? null;
  const line = comment.line ?? comment.original_line ?? null;
  const trusted = TRUSTED.has(comment.author_association);
  return {
    id: `${kind}:${comment.id ?? ''}`, kind, author: comment.user?.login ?? 'unknown',
    trusted, botFinding: isReviewAgentBot, createdAt: comment.updated_at ?? comment.created_at ?? '', path, line,
    relevant: Boolean(path && changedPaths.has(path)),
    unresolved: comment.resolved === false || comment.state === 'CHANGES_REQUESTED',
    body: truncate(body, 1_500),
  };
}

export function selectReviewHistory({ reviewComments = [], issueComments = [], reviews = [], changedFiles = [] }) {
  const changedPaths = new Set(changedFiles.map((file) => typeof file === 'string' ? file : file.filename));
  const candidates = [
    ...reviewComments.map((comment) => itemFromComment(comment, 'inline', changedPaths)),
    ...issueComments.map((comment) => itemFromComment(comment, 'conversation', changedPaths)),
    ...reviews.map((review) => itemFromComment(review, 'review', changedPaths)),
  ].filter(Boolean);
  const seen = new Set();
  const unique = candidates.filter((item) => {
    const key = `${item.author}\0${item.path ?? ''}\0${item.line ?? ''}\0${item.body}`;
    if (seen.has(key)) return false;
    seen.add(key); return true;
  });
  unique.sort((a, b) => {
    // Trusted human feedback comes first. Bot findings are retained only as
    // lower-priority context for checking whether a previous finding remains.
    const score = (x) => (x.trusted ? 8 : 0) + (x.unresolved ? 4 : 0) + (x.relevant ? 2 : 0) + (x.botFinding ? 1 : 0);
    return score(b) - score(a) || String(b.createdAt).localeCompare(String(a.createdAt));
  });
  const selected = [];
  let chars = 0;
  for (const item of unique) {
    const cost = item.body.length + 220;
    if (selected.length >= HISTORY_MAX_ITEMS || chars + cost > HISTORY_MAX_CHARS) continue;
    selected.push(item); chars += cost;
  }
  return { considered: unique.length, included: selected, chars };
}

function log(event, fields = {}) { console.log(JSON.stringify({ event, ...fields })); }
function githubContext() {
  const eventPath = process.env.GITHUB_EVENT_PATH;
  if (!eventPath) throw new Error('Missing GitHub event context.');
  return fs.readFile(eventPath, 'utf8').then(JSON.parse);
}
function runUrl(repository) { return `https://github.com/${repository.full_name}/actions/runs/${process.env.GITHUB_RUN_ID ?? ''}`; }
async function githubRequest(url, options = {}) {
  if (!isAllowedGithubApiUrl(url)) throw new Error('GitHub API URL was not allowed.');
  const response = await fetch(url, { ...options, headers: {
    Accept: 'application/vnd.github+json', Authorization: `Bearer ${process.env.GITHUB_TOKEN}`,
    'X-GitHub-Api-Version': '2022-11-28', ...options.headers,
  }, signal: options.signal ?? AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error(`GitHub API request failed (${response.status}).`);
  return response;
}
async function githubJson(url, options) { return (await githubRequest(url, options)).json(); }
async function paginate(url) {
  const all = [];
  let next = url.includes('?') ? `${url}&per_page=100` : `${url}?per_page=100`;
  while (next) {
    const response = await githubRequest(next); all.push(...await response.json());
    next = /<([^>]+)>; rel="next"/.exec(response.headers.get('link') ?? '')?.[1] ?? null;
  }
  return all;
}
async function addReaction(api, commentId) {
  await githubJson(`${api}/issues/comments/${commentId}/reactions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content: 'eyes' }) });
}
async function postComment(api, issueNumber, body) {
  await githubJson(`${api}/issues/${issueNumber}/comments`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ body }) });
}
async function exactHeadSha(checkoutPath) {
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  return (await promisify(execFile)('git', ['-C', checkoutPath, 'rev-parse', 'HEAD'])).stdout.trim();
}
export function verifyCheckoutShas({ baseSha, headSha, pr }) {
  if (!baseSha || baseSha !== pr.base?.sha) throw new Error('Checked-out PR base did not match GitHub metadata.');
  if (!headSha || headSha !== pr.head?.sha) throw new Error('Checked-out PR head did not match GitHub metadata.');
}

function isWithin(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}
async function snapshotRoot(snapshots, snapshot) {
  if (snapshot !== 'base' && snapshot !== 'head') throw new Error('Snapshot must be base or head.');
  return fs.realpath(snapshots?.[snapshot]);
}
async function snapshotPath(snapshots, snapshot, requested = '.') {
  const root = await snapshotRoot(snapshots, snapshot);
  const candidate = path.resolve(root, requested ?? '.');
  if (!isWithin(root, candidate)) throw new Error('Path must stay within the selected snapshot.');
  const resolved = await fs.realpath(candidate);
  if (!isWithin(root, resolved)) throw new Error('Path must stay within the selected snapshot.');
  if (path.relative(root, resolved).split(path.sep)[0] === '.git') throw new Error('Git metadata cannot be explored.');
  return { root, resolved };
}
async function snapshotFiles(root, start, limit) {
  const files = [];
  const pending = [start];
  let limitReached = false;
  while (pending.length && files.length < limit) {
    const current = pending.shift();
    const entries = await fs.readdir(current, { withFileTypes: true });
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (files.length >= limit) {
        limitReached = true;
        break;
      }
      const candidate = path.join(current, entry.name);
      if (entry.isSymbolicLink()) continue;
      if (entry.isDirectory() && entry.name !== '.git') pending.push(candidate);
      else if (entry.isFile()) files.push(path.relative(root, candidate));
    }
  }
  return { files, truncated: limitReached || pending.length > 0 };
}
async function readSnapshotFile(resolved, maxChars = EXPLORATION_FILE_MAX_CHARS) {
  const stat = await fs.stat(resolved);
  if (!stat.isFile()) throw new Error('Path must name a regular file.');
  const handle = await fs.open(resolved, 'r');
  try {
    const buffer = Buffer.alloc(Math.min(stat.size, maxChars));
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    const content = buffer.toString('utf8', 0, bytesRead);
    if (content.includes('\0')) throw new Error('Binary files cannot be explored.');
    return { content, truncated: stat.size > bytesRead };
  } finally {
    await handle.close();
  }
}
function explorationError(name, error) {
  return JSON.stringify({ tool: name, error: error instanceof Error ? error.message : 'Tool failed.' });
}
export async function executeExplorationTool(call, snapshots) {
  const name = call?.name;
  let args;
  try { args = typeof call?.arguments === 'string' ? JSON.parse(call.arguments) : call?.arguments ?? {}; } catch { return explorationError(name, new Error('Tool arguments were invalid JSON.')); }
  try {
    if (!EXPLORATION_TOOLS.some((tool) => tool.name === name)) throw new Error('Tool is not available.');
    if (name === 'list_files') {
      const { root, resolved } = await snapshotPath(snapshots, args.snapshot, args.path);
      const stat = await fs.stat(resolved);
      if (!stat.isDirectory()) throw new Error('Path must name a directory.');
      const listed = await snapshotFiles(root, resolved, Math.min(Math.max(args.limit ?? 100, 1), 100));
      return JSON.stringify({ snapshot: args.snapshot, path: args.path ?? '.', ...listed });
    }
    if (name === 'read_file') {
      const { root, resolved } = await snapshotPath(snapshots, args.snapshot, args.path);
      const { content, truncated } = await readSnapshotFile(resolved);
      const lines = content.split(/\r?\n/);
      const start = Math.min(Math.max(args.line_start ?? 1, 1), lines.length || 1);
      const end = Math.min(Math.max(args.line_end ?? start + 199, start), start + 199, lines.length);
      return JSON.stringify({ snapshot: args.snapshot, path: path.relative(root, resolved), line_start: start, line_end: end, content: lines.slice(start - 1, end).join('\n'), truncated: truncated || end < lines.length });
    }
    const { root, resolved } = await snapshotPath(snapshots, args.snapshot, args.path);
    const stat = await fs.stat(resolved);
    if (!stat.isDirectory()) throw new Error('Path must name a directory.');
    const { files, truncated } = await snapshotFiles(root, resolved, EXPLORATION_SEARCH_MAX_FILES);
    const matches = [];
    for (const file of files) {
      const { content } = await readSnapshotFile(path.join(root, file), 64_000).catch(() => ({ content: '' }));
      for (const [index, line] of content.split(/\r?\n/).entries()) {
        if (!line.includes(args.query)) continue;
        matches.push({ path: file, line: index + 1, text: truncate(line, 300) });
        if (matches.length >= EXPLORATION_SEARCH_MAX_MATCHES) break;
      }
      if (matches.length >= EXPLORATION_SEARCH_MAX_MATCHES) break;
    }
    return JSON.stringify({ snapshot: args.snapshot, query: args.query, matches, truncated: truncated || matches.length >= EXPLORATION_SEARCH_MAX_MATCHES });
  } catch (error) {
    return explorationError(name, error);
  }
}

export async function runExplorationLoop(requestReview, input, snapshots) {
  let response = await requestReview(input);
  const turns = [{ role: 'user', content: input }];
  let calls = 0;
  let characters = 0;
  while (true) {
    const toolCalls = (Array.isArray(response.output) ? response.output : []).filter((item) => item.type === 'function_call');
    if (!toolCalls.length) return { response, calls, characters };
    if (calls + toolCalls.length > EXPLORATION_MAX_CALLS) throw new Error('Exploration exceeded its fixed tool-call limit.');
    const outputs = [];
    for (const call of toolCalls) {
      const raw = await executeExplorationTool(call, snapshots);
      const { text } = redactSensitiveText(raw);
      const remaining = EXPLORATION_MAX_CHARS - characters;
      if (remaining <= 0) throw new Error('Exploration exceeded its fixed result budget.');
      const output = truncate(text, remaining);
      characters += output.length;
      outputs.push({ type: 'function_call_output', call_id: call.call_id, output });
    }
    calls += toolCalls.length;
    // Responses requires the complete prior output, including reasoning, on
    // manually managed tool turns.
    turns.push(...response.output, ...outputs);
    response = await requestReview(turns, calls >= EXPLORATION_MAX_CALLS || characters >= EXPLORATION_MAX_CHARS ? { toolChoice: 'none' } : undefined);
  }
}
export async function readFileContext(checkoutPath, files) {
  const root = await fs.realpath(checkoutPath);
  const rootPrefix = `${root}${path.sep}`;
  const sections = [];
  let remaining = FILE_CONTEXT_MAX_CHARS;
  for (const file of files) {
    if (remaining <= 0) break;
    if (file.status === 'removed') continue;
    const candidate = path.resolve(root, file.filename);
    if (!candidate.startsWith(rootPrefix)) continue;
    try {
      const resolved = await fs.realpath(candidate);
      if (!resolved.startsWith(rootPrefix)) continue;
      const header = `--- ${file.filename} ---\n`;
      const marker = '\n[truncated]\n';
      const contentLimit = remaining - header.length - marker.length;
      if (contentLimit <= 0) break;
      const handle = await fs.open(resolved, 'r');
      let buffer;
      let bytesRead;
      let truncated;
      try {
        const { size } = await handle.stat();
        buffer = Buffer.alloc(Math.min(size, contentLimit));
        ({ bytesRead } = await handle.read(buffer, 0, buffer.length, 0));
        truncated = size > bytesRead;
      } finally {
        await handle.close();
      }
      const content = buffer.toString('utf8', 0, bytesRead);
      if (content.includes('\0')) continue;
      const section = `${header}${content}${truncated ? marker : '\n'}`;
      sections.push(section);
      remaining -= section.length;
    } catch {
      // Context is best-effort; unreadable, out-of-tree, or binary files are omitted.
    }
  }
  return sections.join('\n');
}
function formatHistory(items) {
  if (!items.length) return '(No relevant prior review feedback selected.)';
  return items.map((item) => `- [${item.kind}; ${item.botFinding ? 'prior torch-ice-review-agent finding' : item.trusted ? 'trusted maintainer' : 'untrusted'}; ${item.createdAt}] ${item.author}${item.path ? ` on ${item.path}${item.line ? `:${item.line}` : ''}` : ''}:\n${item.body}`).join('\n');
}
export function safeFailureReason(error) {
  const message = error instanceof Error ? error.message : '';
  if (/OpenAI API key is not configured/.test(message)) return 'The OpenAI API key is not configured.';
  if (/Checked-out PR base/.test(message)) return 'The checked-out PR base could not be verified.';
  if (/Checked-out PR head/.test(message)) return 'The checked-out PR head could not be verified.';
  if (/GitHub API request failed/.test(message)) return message;
  if (/OpenAI request failed/.test(message)) return message;
  if (/OpenAI request timed out/.test(message) || error?.name === 'TimeoutError') return 'The OpenAI request timed out.';
  if (/GitHub reported changed files but returned no diff/.test(message)) return message;
  if (/OpenAI response did not complete/.test(message)) return 'OpenAI did not complete the review.';
  if (/OpenAI returned no review text/.test(message)) return message;
  if (/safe output limit/.test(message)) return 'OpenAI returned review text that exceeded the safe output limit.';
  if (/Exploration exceeded its fixed tool-call limit/.test(message)) return 'The review exceeded its fixed exploration tool-call limit.';
  if (/Exploration exceeded its fixed result budget/.test(message)) return 'The review exceeded its fixed exploration result-size limit.';
  if (/GitHub API URL was not allowed/.test(message)) return 'A GitHub API URL was rejected by the review agent.';
  return 'An internal torch-ice-review-agent error occurred.';
}
export function reviewRequestTimeoutMs(deadline, now = Date.now()) {
  const remaining = deadline - now;
  if (remaining <= 0) throw new Error('Review deadline exceeded.');
  return Math.min(OPENAI_REQUEST_TIMEOUT_MS, remaining);
}
export function formatFailureComment({ error, repository, headSha = null, force = false }) {
  const safe = safeFailureReason(error);
  const marker = headSha ? `<!-- torch-ice-review-agent: failure head_sha=${headSha}${force ? ' attempt=force' : ''} -->` : '<!-- torch-ice-review-agent: failure -->';
  return { safe, body: `Review agent could not complete this run: ${safe} See [workflow logs](${runUrl(repository)}).\n\n${marker}` };
}
async function reportFailure({ api, prNumber, repository, error, headSha, force }) {
  const failure = formatFailureComment({ error, repository, headSha, force });
  log('review_failure', { pr_number: prNumber, reason: failure.safe });
  await postComment(api, prNumber, failure.body).catch(() => {});
}
async function main() {
  const event = await githubContext();
  const command = parseReviewCommand(event.comment?.body);
  const valid = Boolean(event.issue?.pull_request && TRUSTED.has(event.comment?.author_association) && command);
  if (process.argv.includes('--validate')) {
    if (process.env.GITHUB_OUTPUT) await fs.appendFile(process.env.GITHUB_OUTPUT, `accepted=${valid}\n`);
    return;
  }
  if (!valid) return;
  const api = event.repository.url;
  const prNumber = event.issue.number;
  if (process.argv.includes('--base-sha')) {
    try {
      const pr = await githubJson(`${api}/pulls/${prNumber}`);
      if (!pr.base?.sha) throw new Error('GitHub did not return a PR base SHA.');
      if (process.env.GITHUB_OUTPUT) await fs.appendFile(process.env.GITHUB_OUTPUT, `base_sha=${pr.base.sha}\n`);
    } catch (error) {
      await reportFailure({ api, prNumber, repository: event.repository, error, headSha: null, force: command.force });
      process.exitCode = 1;
    }
    return;
  }
  let headSha = null;
  try {
    // A reaction is only an acknowledgement. Some repositories or token
    // policies deny reactions even when normal issue comments are allowed;
    // do not let that cosmetic operation prevent the requested review.
    await addReaction(api, event.comment.id).catch((error) => {
      log('review_warning', { pr_number: prNumber, operation: 'acknowledgement_reaction', reason: safeFailureReason(error) });
    });
    if (command.force && event.comment.author_association !== 'OWNER') {
      await postComment(api, prNumber, 'Only repository owners may use `@torch-ice-review-agent --force`; no review was run.\n\n<!-- torch-ice-review-agent: rejected reason=force_requires_owner -->');
      return;
    }
    let pr = await githubJson(`${api}/pulls/${prNumber}`);
    const [baseSha, checkedOutHeadSha] = await Promise.all([
      exactHeadSha(process.env.PR_BASE_CHECKOUT_PATH), exactHeadSha(process.env.PR_CHECKOUT_PATH),
    ]);
    verifyCheckoutShas({ baseSha, headSha: checkedOutHeadSha, pr });
    headSha = checkedOutHeadSha;
    const [files, issueComments, reviewComments, reviews, diffResponse] = await Promise.all([
      paginate(`${api}/pulls/${prNumber}/files`), paginate(`${api}/issues/${prNumber}/comments`),
      paginate(`${api}/pulls/${prNumber}/comments`), paginate(`${api}/pulls/${prNumber}/reviews`),
      githubRequest(`${api}/pulls/${prNumber}`, { headers: { Accept: 'application/vnd.github.v3.diff' } }),
    ]);
    pr = await githubJson(`${api}/pulls/${prNumber}`);
    verifyCheckoutShas({ baseSha, headSha: checkedOutHeadSha, pr });
    const rawDiff = await diffResponse.text();
    if (files.length > 0 && !rawDiff.trim()) throw new Error('GitHub reported changed files but returned no diff.');
    const diff = truncate(rawDiff, DIFF_MAX_CHARS);
    const blockedLabel = (pr.labels ?? []).map((label) => String(label.name ?? '').toLowerCase()).find((name) => BLOCKED_LABELS.has(name));
    if (blockedLabel) {
      log('review_rejected', { pr_number: prNumber, reason: 'blocked_label', label: blockedLabel });
      await postComment(api, prNumber, `Review agent did not run because this PR has the \`${blockedLabel}\` label.\n\n<!-- torch-ice-review-agent: rejected reason=blocked_label -->`);
      return;
    }
    const successfulForcedReviews = issueComments.filter((comment) => isSuccessfulReviewResult(comment, headSha) && String(comment.body).replace(/\r$/, '').endsWith(`${BOT_MARKER}${headSha} attempt=force -->`));
    const latestForcedSuccess = successfulForcedReviews.map((comment) => Date.parse(comment.created_at ?? comment.updated_at ?? '')).filter(Number.isFinite).sort((a, b) => b - a)[0];
    if (command.force && latestForcedSuccess && Date.now() - latestForcedSuccess < FORCE_COOLDOWN_MS) {
      log('review_rejected', { pr_number: prNumber, reason: 'force_cooldown', head_sha: headSha });
      await postComment(api, prNumber, 'A review for this PR head ran recently. Wait 15 minutes before forcing another review.\n\n<!-- torch-ice-review-agent: rejected reason=force_cooldown -->');
      return;
    }
    if (command.force && successfulForcedReviews.length >= FORCE_MAX_PER_HEAD) {
      log('review_rejected', { pr_number: prNumber, reason: 'force_limit', head_sha: headSha });
      await postComment(api, prNumber, 'This PR head has reached its limit of two forced reviews. Push a new commit before requesting another.\n\n<!-- torch-ice-review-agent: rejected reason=force_limit -->');
      return;
    }
    const priorSuccess = issueComments.some((comment) => isSuccessfulReviewResult(comment, headSha));
    log('review_context', { pr_number: prNumber, head_sha: headSha, changed_files: files.length, diff_characters_received: rawDiff.length, diff_characters_sent: diff.length, diff_truncated: diff.length !== rawDiff.length, deduplication_skipped: priorSuccess && !command.force });
    if (priorSuccess && !command.force) {
      await postComment(api, prNumber, formatDeduplicationComment(headSha));
      return;
    }
    const history = selectReviewHistory({ reviewComments, issueComments, reviews, changedFiles: files });
    log('review_history', { comments_considered: history.considered, comments_included: history.included.length, history_characters_sent: history.chars });
    const [instructions, checklist] = await Promise.all([
      fs.readFile(path.join(process.cwd(), '.github/prompts/torch-ice-review-agent.md'), 'utf8'),
      fs.readFile(path.join(process.cwd(), '.github/prompts/architecture-review-checklist.md'), 'utf8'),
    ]);
    if (!process.env.OPENAI_API_KEY) throw new Error('The OpenAI API key is not configured.');
    const reviewMode = selectReviewMode(files);
    const fileContext = await readFileContext(process.env.PR_CHECKOUT_PATH, files);
    const reviewInput = buildReviewInput({ commandPrompt: command.prompt, pr, headSha, files, fileContext, diff, history: history.included, checklist, reviewMode });
    const { text: input, count: redactions } = redactSensitiveText(reviewInput.input);
    log('review_input', { pr_number: prNumber, review_mode: reviewMode, input_characters: input.length, input_budget_characters: INPUT_MAX_CHARS, file_context_characters: fileContext.length, truncated: reviewInput.truncated, redactions });
    const started = Date.now();
    const deadline = started + REVIEW_DEADLINE_MS;
    const requestReview = async (requestInput, maxOutputTokens, { toolChoice = 'auto' } = {}) => {
      try {
        return await fetch('https://api.openai.com/v1/responses', { method: 'POST', signal: AbortSignal.timeout(reviewRequestTimeoutMs(deadline)), headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ model: 'gpt-5.6-terra', text: { verbosity: 'medium' }, max_output_tokens: maxOutputTokens, store: false, instructions, tools: EXPLORATION_TOOLS, tool_choice: toolChoice, parallel_tool_calls: false, input: requestInput }) });
      } catch (error) {
        if (error?.name === 'TimeoutError') throw new Error('OpenAI request timed out.');
        throw error;
      }
    };
    const explore = async (maxOutputTokens) => runExplorationLoop(async (requestInput, options) => {
      const response = await requestReview(requestInput, maxOutputTokens, options);
      if (!response.ok) throw new Error(`OpenAI request failed (${response.status}).`);
      return response.json();
    }, input, { base: process.env.PR_BASE_CHECKOUT_PATH, head: process.env.PR_CHECKOUT_PATH });
    let exploration = await explore(INITIAL_MAX_OUTPUT_TOKENS);
    let result = exploration.response;
    if (shouldRetryForOutputLimit(result)) {
      log('openai_retry', { pr_number: prNumber, reason: result.incomplete_details.reason, max_output_tokens: RETRY_MAX_OUTPUT_TOKENS });
      exploration = await explore(RETRY_MAX_OUTPUT_TOKENS);
      result = exploration.response;
    }
    const latencyMs = Date.now() - started;
    const extracted = extractResponseText(result);
    log('openai_response', {
      latency_ms: latencyMs,
      status: result.status ?? null,
      output_items: Array.isArray(result.output) ? result.output.length : 0,
      exploration_tool_calls: exploration.calls,
      exploration_characters_sent: exploration.characters,
      extracted_text_characters: extracted.length,
      incomplete_details: result.incomplete_details ?? null,
      usage: result.usage ?? null,
    });
    if (result.status !== 'completed') throw new Error('OpenAI response did not complete.');
    if (!extracted) throw new Error('OpenAI returned no review text.');
    const output = sanitizeReviewOutput(extracted);
    if (!output) throw new Error('OpenAI returned no review text.');
    await postComment(api, prNumber, `${output}\n\n${BOT_MARKER}${headSha}${command.force ? ' attempt=force' : ''} -->`);
  } catch (error) {
    await reportFailure({ api, prNumber, repository: event.repository, error, headSha, force: command.force });
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
