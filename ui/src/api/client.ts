import { invoke } from '@tauri-apps/api/core';

export interface SidecarInfo {
  url: string;
  token: string;
}

interface SidecarInvokeResult {
  url: string | null;
  token: string;
}

let cached: SidecarInfo | null = null;
let readyPromise: Promise<SidecarInfo> | null = null;

const READY_TIMEOUT_MS = 120_000;
const POLL_INTERVAL_MS = 200;

async function discoverAndProbe(): Promise<SidecarInfo> {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  let url: string | null = null;
  let token = '';
  while (Date.now() < deadline) {
    if (!url || !token) {
      const info = (await invoke('sidecar_url')) as SidecarInvokeResult;
      url = info.url;
      token = info.token;
    }
    if (url && token) {
      try {
        const r = await fetch(url + '/health', {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (r.ok) return { url, token };
      } catch {}
    }
    await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
  }
  throw new Error('sidecar did not become healthy within 120 seconds');
}

export async function sidecarInfo(): Promise<SidecarInfo> {
  if (cached) return cached;
  if (!readyPromise) {
    readyPromise = discoverAndProbe().then(info => { cached = info; return info; });
  }
  return readyPromise;
}

// Drop the cached URL + token so the next api() call re-discovers the sidecar.
// Toggling hub mode respawns the sidecar on a new loopback port with a new
// bearer token; without this the cache would dial the dead old socket and
// fetch would throw "TypeError: Load failed".
export function resetSidecarInfo(): void {
  cached = null;
  readyPromise = null;
}

export async function baseUrl(): Promise<string> {
  return (await sidecarInfo()).url;
}

const IDEMPOTENT_METHODS = new Set(['GET', 'HEAD', 'PUT', 'DELETE', 'OPTIONS']);

// Only methods that are safe to repeat may be retried on a network error.
// A POST/PATCH can reach the server and have its response lost; resending it
// would duplicate the side effect (e.g. queue a second transcription job).
function isIdempotent(method: string | undefined): boolean {
  return IDEMPOTENT_METHODS.has((method ?? 'GET').toUpperCase());
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const info = await sidecarInfo();
  const headers = new Headers(init?.headers);
  headers.set('Authorization', `Bearer ${info.token}`);
  let lastErr: unknown;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch(info.url + path, { ...init, headers });
      if (!r.ok) throw new Error(`${r.status} ${path}: ${await r.text()}`);
      return r.json() as Promise<T>;
    } catch (e) {
      lastErr = e;
      if (e instanceof TypeError && isIdempotent(init?.method)) {
        await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
        continue;
      }
      throw e;
    }
  }
  throw lastErr;
}
