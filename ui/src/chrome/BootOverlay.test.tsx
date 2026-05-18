import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BootOverlay } from './BootOverlay';
import { useBackend } from '../stores/backend';

function setBackend(partial: Partial<ReturnType<typeof useBackend.getState>>) {
  useBackend.setState({ ...useBackend.getState(), ...partial });
}

describe('BootOverlay', () => {
  beforeEach(() => {
    useBackend.getState()._resetForTests();
    useBackend.setState({ status: 'starting', elapsedMs: 0, error: null });
  });

  it('renders while status is "starting"', () => {
    render(<BootOverlay />);
    expect(screen.getByTestId('boot-overlay')).toBeInTheDocument();
    expect(screen.getByText(/starting/i)).toBeInTheDocument();
  });

  it('does not render when status is "ready"', () => {
    setBackend({ status: 'ready' });
    render(<BootOverlay />);
    expect(screen.queryByTestId('boot-overlay')).not.toBeInTheDocument();
  });

  it('shows extended message after 15 seconds', () => {
    setBackend({ elapsedMs: 16000 });
    render(<BootOverlay />);
    expect(screen.getByText(/taking longer than usual/i)).toBeInTheDocument();
  });

  it('shows error message when status is "failed"', () => {
    setBackend({ status: 'failed', error: 'boom' });
    render(<BootOverlay />);
    expect(screen.getByText(/couldn.?t start/i)).toBeInTheDocument();
  });
});
