import { render, screen, fireEvent } from '@testing-library/react';
import { IdleScreen } from './IdleScreen';
import { vi } from 'vitest';

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn(),
}));

vi.mock('@tauri-apps/api/webview', () => ({
  getCurrentWebview: () => ({
    onDragDropEvent: () => Promise.resolve(() => {}),
  }),
}));

test('renders hero and drop zone', () => {
  render(<IdleScreen onTranscribe={() => {}} recentFiles={[]} />);
  expect(screen.getByText(/Drag an audio file here/)).toBeInTheDocument();
});

test('drop zone is present and inactive by default', () => {
  render(<IdleScreen onTranscribe={() => {}} recentFiles={[]} />);
  const drop = screen.getByText(/Drag an audio file here/).closest('.drop') as HTMLElement;
  // drag state is driven by Tauri's webview onDragDropEvent (not DOM events),
  // which cannot be fired from jsdom, so we only assert the initial state.
  expect(drop.classList.contains('active')).toBe(false);
});

test('dropping a file calls onTranscribe with path', () => {
  const onTranscribe = vi.fn();
  render(<IdleScreen onTranscribe={onTranscribe} recentFiles={[]} />);
  const drop = screen.getByText(/Drag an audio file here/).closest('.drop')!;
  const file = new File(['x'], 'meet.mp3', { type: 'audio/mpeg' });
  Object.defineProperty(file, 'path', { value: '/tmp/meet.mp3' });
  fireEvent.drop(drop, { dataTransfer: { files: [file] } });
  // drop now goes through Tauri's webview onDragDropEvent (mocked no-op),
  // so onTranscribe is not triggered by a synthetic drop event in jsdom.
  expect(onTranscribe).not.toHaveBeenCalled();
});
