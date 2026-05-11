import './index.css';

function App() {
  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="brand-name">RANT FREE</div>
          <div className="brand-tag">say it. we'll bleep it.</div>
        </div>
      </header>

      <main className="feed">
        <div className="feed-empty">no rants yet. be the first.</div>
      </main>

      <Composer />
    </div>
  );
}

function Composer() {
  return (
    <div className="composer">
      <textarea
        className="composer-textarea"
        placeholder="get it off your chest..."
        rows={2}
      />
      <button className="composer-submit">
        post rant →
      </button>
    </div>
  );
}

export default App;
