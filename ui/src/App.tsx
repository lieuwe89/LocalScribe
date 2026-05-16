import { useState, useEffect } from 'react';
import './styles/global.css';
import { Window } from './chrome/Window';
import { Sidebar } from './chrome/Sidebar';
import { MainHeader } from './chrome/MainHeader';
import { useLibrary } from './stores/library';
import type { Route } from './types/route';

export default function App() {
  const [route, setRoute] = useState<Route>('idle');
  const [tid, setTid] = useState<string | null>(null);
  const refreshLibrary = useLibrary(s => s.refresh);

  useEffect(() => { refreshLibrary().catch(() => {}); }, [refreshLibrary]);

  return (
    <Window screenLabel={route}>
      <Sidebar route={route} setRoute={setRoute} currentTranscriptId={tid} setCurrentTranscriptId={setTid} />
      <div className="main">
        <MainHeader route={route} />
        <div className="main-body"><pre>{route}</pre></div>
      </div>
    </Window>
  );
}
