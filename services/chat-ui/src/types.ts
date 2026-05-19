export type ToxicityScores = {
  toxic: number;
  severe_toxic: number;
  obscene: number;
  threat: number;
  insult: number;
  identity_hate: number;
};

export type FilterMode = 'off' | 'mild' | 'strict';

export type Post = {
  id: string;
  author: string;
  content: string;
  createdAt: string;
  scores: ToxicityScores | null;
  confidence: number | null;
  requestId: string | null;
};

export type PostSeverity = 'clean' | 'mild' | 'severe';
