import { create } from 'zustand';
import { baseUrl } from '../api/client';

export type BackendStatus = 'starting' | 'ready' | 'failed';

interface State {
  status: BackendStatus;
  elapsedMs: number;
  error: string | null;
  /** True once start() has been called; reset alongside status via setState in tests. */
  _started: boolean;
  start: () => void;
}

const TICK_MS = 250;

export const useBackend = create<State>((set, get) => {
  let startTs = 0;
  let tickHandle: ReturnType<typeof setInterval> | null = null;

  const stopTicking = () => {
    if (tickHandle !== null) {
      clearInterval(tickHandle);
      tickHandle = null;
    }
  };

  return {
    status: 'starting',
    elapsedMs: 0,
    error: null,
    _started: false,
    start: () => {
      if (get()._started) return;
      set({ _started: true });
      startTs = Date.now();
      // Update elapsedMs every TICK_MS while still starting
      tickHandle = setInterval(() => {
        if (get().status === 'starting') {
          set({ elapsedMs: Date.now() - startTs });
        }
      }, TICK_MS);

      baseUrl()
        .then(async (url) => {
          const r = await fetch(url + '/health');
          if (!r.ok) throw new Error(`/health ${r.status}`);
          set({ status: 'ready', elapsedMs: Date.now() - startTs });
          stopTicking();
        })
        .catch((e) => {
          set({ status: 'failed', error: String(e), elapsedMs: Date.now() - startTs });
          stopTicking();
        });
    },
  };
});
