import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../api/client', () => ({
  baseUrl: vi.fn(),
}));

import { baseUrl } from '../api/client';
import { useBackend } from './backend';

describe('useBackend', () => {
  beforeEach(() => {
    useBackend.setState({ status: 'starting', elapsedMs: 0, error: null, _started: false });
    vi.useFakeTimers();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('starts in "starting" state', () => {
    expect(useBackend.getState().status).toBe('starting');
  });

  it('transitions to "ready" once baseUrl resolves and /health returns ok', async () => {
    (baseUrl as ReturnType<typeof vi.fn>).mockResolvedValue('http://127.0.0.1:1234');
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true });

    useBackend.getState().start();
    await vi.runOnlyPendingTimersAsync();
    await vi.runOnlyPendingTimersAsync();

    expect(useBackend.getState().status).toBe('ready');
  });

  it('tracks elapsedMs while starting', async () => {
    (baseUrl as ReturnType<typeof vi.fn>).mockImplementation(
      () => new Promise(() => {}) // never resolves
    );

    useBackend.getState().start();
    await vi.advanceTimersByTimeAsync(2000);

    expect(useBackend.getState().elapsedMs).toBeGreaterThanOrEqual(2000);
    expect(useBackend.getState().status).toBe('starting');
  });
});
