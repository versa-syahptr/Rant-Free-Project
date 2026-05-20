export type ToxicityResult = {
  score_toxic: number;
};

export type FilterMode = 'off' | 'mild' | 'strict';

export type Post = {
  id: string;
  author: string;
  content: string;
  createdAt: string;
  scores: ToxicityResult | null;
  confidence: number | null;
  requestId: string | null;
};

export type PostSeverity = 'clean' | 'mild' | 'severe';
