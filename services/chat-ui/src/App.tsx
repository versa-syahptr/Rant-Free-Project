import './index.css';
import {useState, useEffect, useRef} from 'react';

const MODEL_SERVICE_URL = import.meta.env.VITE_MODEL_SERVICE_URL ?? 'http://localhost:8000';

type ToxicityScores = {
  toxic: number;
  severe_toxic: number;
  obscene: number;
  threat: number;
  insult: number;
  identity_hate: number;
};

type Post = {
  id: string;
  author: string;
  content: string;
  createdAt: string;
  scores: ToxicityScores | null;
  confidence: number | null;
};

type FilterMode = 'off' | 'mild' | 'strict';

// mild: hide if toxic or obscene score > 0.5
// strict: hide if any score > 0.3
function isToxic(scores: ToxicityScores | null, mode: FilterMode): boolean {
  if (mode === 'off' || scores === null) return false;
  if (mode === 'mild') return scores.toxic > 0.5 || scores.obscene > 0.5;
  return Object.values(scores).some(s => s > 0.3);
}

async function fetchPrediction(text: string): Promise<{scores: ToxicityScores; confidence: number} | null> {
  try {
    const res = await fetch(`${MODEL_SERVICE_URL}/predict`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text}),
    });
    if (!res.ok) return null;
    const data = await res.json();
    const scores = Object.fromEntries(
      data.predictions.map((p: {label: string; score: number}) => [p.label, p.score])
    ) as ToxicityScores;
    return {scores, confidence: data.confidence};
  } catch {
    return null;
  }
}

function App() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [filterMode, setFilterMode] = useState<FilterMode>('off');
  const feedEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({behavior: 'smooth'});
  }, [posts]);

  async function handleSubmit(content: string) {
    const prediction = await fetchPrediction(content);
    const newPost: Post = {
      id: crypto.randomUUID(),
      author: 'You',
      content,
      createdAt: new Date().toISOString(),
      scores: prediction?.scores ?? null,
      confidence: prediction?.confidence ?? null,
    };
    setPosts(prev => [...prev, newPost]);
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="brand-name">RANT FREE</div>
          <div className="brand-tag">say it. we'll bleep it.</div>
        </div>
        <FilterToggle mode={filterMode} onChange={setFilterMode} />
      </header>

      <main className="feed">
        {posts.length === 0 ? (
          <div className="feed-empty">no rants yet. be the first.</div>
        ) : (
          posts.map(post => <PostCard key={post.id} post={post} filterMode={filterMode} />)
        )}
        <div ref={feedEndRef} />
      </main>

      <Composer onSubmit={handleSubmit} />
    </div>
  );
}

function FilterToggle({mode, onChange}: {mode: FilterMode; onChange: (m: FilterMode) => void}) {
  const options: {id: FilterMode; label: string}[] = [
    {id:'off', label:'off'},
    {id:'mild', label:'mild'},
    {id:'strict', label:'strict'},
  ];
  return (
    <div className="filter">
      <div className="filter-label">civilized mode</div>
      <div className="filter-seg">
        {options.map(o => (
          <button
            key={o.id}
            className={`filter-opt ${mode === o.id ? 'on' : ''}`}
            onClick={() => onChange(o.id)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function PostCard({post, filterMode}: {post: Post; filterMode: FilterMode}) {
  if (isToxic(post.scores, filterMode)) {
    return (
      <div className="post-card post-card--hidden">
        <div className="post-hidden-label">⚠ filtered by civilized mode</div>
      </div>
    );
  }
  return (
    <div className="post-card">
      <div className="post-meta">
        <span className="post-author">@{post.author}</span>
        <span className="post-date">{new Date(post.createdAt).toLocaleTimeString()}</span>
      </div>
      <div className="post-content">{post.content}</div>
    </div>
  );
}

function Composer({onSubmit}: {onSubmit: (content: string) => Promise<void>}) {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setLoading(true);
    setText('');
    await onSubmit(trimmed);
    setLoading(false);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>){
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }
  return (
    <div className="composer">
      <textarea
        className="composer-textarea"
        placeholder="get it off your chest..."
        rows={2}
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
      />
      <button className="composer-submit" onClick={handleSubmit} disabled={!text.trim() || loading}>
        {loading ? 'checking...' : 'post rant →'}
      </button>
    </div>
  );
}

export default App;
