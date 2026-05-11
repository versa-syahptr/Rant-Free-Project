import './index.css';
import {useState} from 'react';

type Post = {
  id: string;
  author: string;
  content: string;
  createdAt: string;
};

function App() {
  const [posts, setPosts] = useState<Post[]>([]);

  function handleSubmit(content: string) {
    const newPost: Post = {
      id: crypto.randomUUID(),
      author: 'You',
      content,
      createdAt: new Date().toISOString()
    };
    setPosts([newPost, ...posts]);
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="brand-name">RANT FREE</div>
          <div className="brand-tag">say it. we'll bleep it.</div>
        </div>
      </header>

      <main className="feed">
        {posts.length === 0 ? (
          <div className="feed-empty">no rants yet. be the first.</div>
        ) : (
          posts.map(post => <PostCard key={post.id} post={post} />)
        )}
        
      </main>

      <Composer onSubmit={handleSubmit} />
    </div>
  );
}

function PostCard({post}: {post: Post}) {
  return (
    <div className="post-card">
      <div className="post-meta">
        <span className="post-author">{post.author}</span>
        <span className="post-date">{new Date(post.createdAt).toLocaleString()}</span>
      </div>
      <div className="post-content">{post.content}</div>
    </div>
  );
}

function Composer({onSubmit}: {onSubmit: (content: string) => void}) {
  const [text, setText] = useState('');

  function handleSubmit() {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
    setText('');
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
      />
      <button className="composer-submit" onClick={handleSubmit} disabled={!text.trim()}>
        post rant →
      </button>
    </div>
  );
}

export default App;
