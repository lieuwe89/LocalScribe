import { invoke } from '@tauri-apps/api/core';

let cached: string | null = null;
let readyPromise: Promise<string> | null = null;

const READY_TIMEOUT_MS = 120_000;
const POLL_INTERVAL_MS = 200;

async function discoverAndProbe(): Promise<string> {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  let url: string | null = null;
  while (Date.now() < deadline) {
    if (!url) {
      url = (await invoke('sidecar_url')) as string | null;
    }
    if (url) {
      try {
        const r = await fetch(url + '/health');
        if (r.ok) return url;
      } catch {}
    }
    await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
  }
  throw new Error('sidecar did not become healthy within 120 seconds');
}

export async function baseUrl(): Promise<string> {
  if (cached) return cached;
  if (!readyPromise) {
    readyPromise = discoverAndProbe().then(u => { cached = u; return u; });
  }
  return readyPromise;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const u = await baseUrl();
  let lastErr: unknown;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch(u + path, init);
      if (!r.ok) throw new Error(`${r.status} ${path}: ${await r.text()}`);
      return r.json() as Promise<T>;
    } catch (e) {
      lastErr = e;
      if (e instanceof TypeError) {
        await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
        continue;
      }
      throw e;
    }
  }
  throw lastErr;
}
