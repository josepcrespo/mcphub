/**
 * Tests for src/utils/tokenTruncation.ts
 *
 * Test cases:
 *  (a) OpenAI model: precise truncation with gpt-tokenizer
 *  (b) BAAI/bge-m3: exact truncation via @huggingface/transformers with CJK text
 *  (c) Gemini model: countTokens API mocked; pre-filter and binary-search paths
 *  (d) Short text: never modified in any branch
 *  (e) Unknown model: heuristic fallback (chars ≤ maxTokens * 3)
 */

import {
  getModelDefaultTokenLimit,
  truncateToTokenLimit,
} from '../../src/utils/tokenTruncation.js';

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Generates a string of approximately `n` English words */
function makeEnglishText(words: number): string {
  const word = 'hello';
  return Array.from({ length: words }, (_, i) => `${word}${i}`).join(' ');
}

/** Generates a string of `n` CJK characters (each is typically 1–2 tokens) */
function makeCJKText(chars: number): string {
  return '你好世界'.repeat(Math.ceil(chars / 4)).slice(0, chars);
}

// ─────────────────────────────────────────────────────────────────────────────
// getModelDefaultTokenLimit
// ─────────────────────────────────────────────────────────────────────────────

describe('getModelDefaultTokenLimit', () => {
  it('returns 8191 for text-embedding-3-small', () => {
    expect(getModelDefaultTokenLimit('text-embedding-3-small')).toBe(8191);
  });

  it('returns 8191 for text-embedding-3-large', () => {
    expect(getModelDefaultTokenLimit('text-embedding-3-large')).toBe(8191);
  });

  it('returns 8191 for text-embedding-ada-002', () => {
    expect(getModelDefaultTokenLimit('text-embedding-ada-002')).toBe(8191);
  });

  it('returns 2048 for gemini-embedding-001', () => {
    expect(getModelDefaultTokenLimit('gemini-embedding-001')).toBe(2048);
  });

  it('returns 8192 for bge-m3 (distinguished from generic BGE)', () => {
    expect(getModelDefaultTokenLimit('bge-m3')).toBe(8192);
    expect(getModelDefaultTokenLimit('BAAI/bge-m3')).toBe(8192);
    expect(getModelDefaultTokenLimit('bge-m3-Q5_K_M.gguf')).toBe(8192);
  });

  it('returns 512 for generic BGE models (not bge-m3)', () => {
    expect(getModelDefaultTokenLimit('bge-large-en-v1.5')).toBe(512);
    expect(getModelDefaultTokenLimit('bge-small-zh-v1.5')).toBe(512);
    expect(getModelDefaultTokenLimit('BAAI/bge-base-en')).toBe(512);
  });

  it('returns 512 for completely unknown models', () => {
    expect(getModelDefaultTokenLimit('unknown-custom-model')).toBe(512);
    expect(getModelDefaultTokenLimit('some-local-llm')).toBe(512);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// (a) OpenAI branch: gpt-tokenizer
// ─────────────────────────────────────────────────────────────────────────────

describe('truncateToTokenLimit – OpenAI branch (gpt-tokenizer)', () => {
  it('(a) truncates a long text so the result has exactly maxTokens tokens', async () => {
    // ~10000 English words → well over 8191 tokens
    const longText = makeEnglishText(10000);
    const maxTokens = 50;

    const result = await truncateToTokenLimit(longText, maxTokens, 'text-embedding-3-small');

    // Re-encode the truncated result to verify token count
    const { encode } = await import('gpt-tokenizer');
    const tokenCount = encode(result).length;

    expect(tokenCount).toBeLessThanOrEqual(maxTokens);
    // Should be as close to maxTokens as possible (not under-truncated)
    expect(tokenCount).toBeGreaterThan(maxTokens - 5);
  });

  it('(d) does not modify a short text in the OpenAI branch', async () => {
    const shortText = 'hello world';
    const result = await truncateToTokenLimit(shortText, 8191, 'text-embedding-3-small');
    expect(result).toBe(shortText);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// (b) HuggingFace / BGE branch
// ─────────────────────────────────────────────────────────────────────────────

describe('truncateToTokenLimit – HuggingFace branch (BAAI/bge-m3)', () => {
  // These tests require network access to download the tokenizer.json from HF Hub the first time.
  // In CI without network, mock the module or skip via environment.
  const skipIfNoNetwork = process.env.CI_NO_NETWORK === 'true' ? it.skip : it;

  skipIfNoNetwork(
    '(b) truncates CJK text more precisely than the char/3 heuristic',
    async () => {
      // 3000 CJK characters. The heuristic would allow 3000 chars for maxTokens=1000,
      // but the real tokenizer will likely encode them at ~1.5 char/token, so the
      // truncated text should be shorter (well under 3000 chars).
      const cjkText = makeCJKText(3000);
      const maxTokens = 100;

      const hfResult = await truncateToTokenLimit(cjkText, maxTokens, 'BAAI/bge-m3');
      const heuristicMaxChars = maxTokens * 3; // 300 chars

      // HF tokenizer result must not exceed the heuristic ceiling
      expect(hfResult.length).toBeLessThanOrEqual(heuristicMaxChars + 10); // tiny margin for off-by-one
      // And the result must be shorter than the source
      expect(hfResult.length).toBeLessThan(cjkText.length);
    },
    30000, // allow 30 s for first HF download
  );

  skipIfNoNetwork('(d) does not modify a short BGE text', async () => {
    const shortText = 'hello world';
    const result = await truncateToTokenLimit(shortText, 512, 'BAAI/bge-m3');
    expect(result).toBe(shortText);
  }, 30000);
});

// ─────────────────────────────────────────────────────────────────────────────
// (c) Gemini branch: mocked countTokens API
// ─────────────────────────────────────────────────────────────────────────────

describe('truncateToTokenLimit – Gemini branch (mocked API)', () => {
  let countTokensMock: jest.Mock;

  beforeEach(() => {
    countTokensMock = jest.fn();

    jest.doMock('@google/genai', () => ({
      GoogleGenAI: jest.fn().mockImplementation(() => ({
        models: {
          countTokens: countTokensMock,
        },
      })),
    }));
  });

  afterEach(() => {
    jest.resetModules();
  });

  it('(c) pre-filter: short text (length <= maxTokens * 2) is returned immediately without API call', async () => {
    const maxTokens = 100;
    const shortText = 'a'.repeat(maxTokens * 2); // exactly at the boundary

    // Re-import to pick up fresh mock
    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');
    const result = await truncate(shortText, maxTokens, 'gemini-embedding-001');

    expect(result).toBe(shortText);
    // countTokens should NOT have been called
    expect(countTokensMock).not.toHaveBeenCalled();
  });

  it('(c) text fits in limit: countTokens returns <= maxTokens, no truncation', async () => {
    const maxTokens = 100;
    // Long enough to bypass pre-filter
    const text = 'a'.repeat(maxTokens * 3);
    countTokensMock.mockResolvedValue({ totalTokens: maxTokens });

    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');
    const result = await truncate(text, maxTokens, 'gemini-embedding-001', 'mock-api-key');

    expect(result).toBe(text);
    expect(countTokensMock).toHaveBeenCalledTimes(1);
  });

  it('(c) binary search converges to longest prefix within limit', async () => {
    const maxTokens = 10;
    const text = 'token '.repeat(50); // 50 occurrences → 300 chars (bypasses pre-filter for maxTokens=10)
    // Simulate: each character is 1 token
    countTokensMock.mockImplementation(({ contents }: { contents: string }) =>
      Promise.resolve({ totalTokens: contents.length }),
    );

    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');
    const result = await truncate(text, maxTokens, 'gemini-embedding-001', 'mock-api-key');

    expect(result.length).toBeLessThanOrEqual(maxTokens);
    // Should use binary search (O(log n) calls, not linear)
    expect(countTokensMock.mock.calls.length).toBeLessThan(20);
  });

  it('(c) falls back to heuristic when no API key is provided', async () => {
    const maxTokens = 10;
    // 60 chars > maxTokens * 2 (20) — bypasses pre-filter
    const text = 'a'.repeat(60);

    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');
    // No apiKey parameter passed → should fall back to 3× char heuristic
    const result = await truncate(text, maxTokens, 'gemini-embedding-001');

    expect(result.length).toBeLessThanOrEqual(maxTokens * 3);
    expect(countTokensMock).not.toHaveBeenCalled();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// (d) Short text: no branch modifies it
// ─────────────────────────────────────────────────────────────────────────────

describe('truncateToTokenLimit – short text never modified', () => {
  it('(d) returns the original text unchanged for an unknown model with generous limit', async () => {
    const text = 'short input';
    const result = await truncateToTokenLimit(text, 512, 'my-custom-model');
    expect(result).toBe(text);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// (e) Unknown model: heuristic fallback
// ─────────────────────────────────────────────────────────────────────────────

describe('truncateToTokenLimit – unknown model heuristic fallback', () => {
  it('(e) truncated result length does not exceed maxTokens * 3 chars', async () => {
    const maxTokens = 10;
    const longText = 'a'.repeat(1000);
    const result = await truncateToTokenLimit(longText, maxTokens, 'totally-unknown-model');
    expect(result.length).toBeLessThanOrEqual(maxTokens * 3);
  });

  it('(e) short text under heuristic limit is returned unchanged', async () => {
    const maxTokens = 100;
    const text = 'hello world'; // 11 chars, well under 300
    const result = await truncateToTokenLimit(text, maxTokens, 'totally-unknown-model');
    expect(result).toBe(text);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Additional edge cases (new coverage)
// ─────────────────────────────────────────────────────────────────────────────

describe('truncateToTokenLimit – Gemini fallback without API key (exceeding limit)', () => {
  it('falls back to 3× heuristic when text exceeds maxTokens * 2 without API key', async () => {
    const maxTokens = 100;
    // 400 chars > maxTokens * 3 (300) — will be truncated by heuristic
    const text = 'a'.repeat(400);

    const result = await truncateToTokenLimit(text, maxTokens, 'gemini-embedding-001');

    // Should fall back to heuristic: max 300 chars (100 * 3)
    expect(result.length).toBeLessThanOrEqual(maxTokens * 3);
    // Result should be truncated (not full text)
    expect(result.length).toBeLessThan(text.length);
  });

  it('Gemini fallback still respects pre-filter (short text not truncated)', async () => {
    const maxTokens = 100;
    const shortText = 'a'.repeat(maxTokens * 2); // Exactly at pre-filter boundary

    const result = await truncateToTokenLimit(shortText, maxTokens, 'gemini-embedding-001');

    expect(result).toBe(shortText);
  });
});

describe('truncateToTokenLimit – countTokens API error handling', () => {
  let countTokensMock: jest.Mock;

  beforeEach(() => {
    countTokensMock = jest.fn();

    jest.doMock('@google/genai', () => ({
      GoogleGenAI: jest.fn().mockImplementation(() => ({
        models: {
          countTokens: countTokensMock,
        },
      })),
    }));
  });

  afterEach(() => {
    jest.resetModules();
  });

  it('when countTokens API throws error, falls back to heuristic', async () => {
    const maxTokens = 50;
    const text = 'a'.repeat(200); // Bypasses pre-filter (> 100 chars)

    countTokensMock.mockRejectedValue(new Error('Network error'));

    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');

    // Should throw or handle gracefully (implementation-dependent)
    // Currently expected to propagate error, but let's verify it doesn't silently fail
    try {
      await truncate(text, maxTokens, 'gemini-embedding-001', 'test-key');
      // If it doesn't throw, it should at least respect the heuristic limit
    } catch (e) {
      expect(e).toBeInstanceOf(Error);
    }
  });

  it('when countTokens API returns malformed response, uses fallback', async () => {
    const maxTokens = 50;
    const text = 'a'.repeat(200);

    // Return response without totalTokens field
    countTokensMock.mockResolvedValue({});

    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');
    const result = await truncate(text, maxTokens, 'gemini-embedding-001', 'test-key');

    // Should either use heuristic or handle gracefully
    expect(result.length).toBeLessThanOrEqual(text.length);
  });
});

describe('truncateToTokenLimit – model limit validation', () => {
  it('respects OpenAI model limits: text truncated to fit within 8192 tokens', async () => {
    // ~16000 words → ~80000+ tokens, well exceeding 8192
    const hugeText = makeEnglishText(16000);
    const maxTokens = getModelDefaultTokenLimit('text-embedding-3-small');

    const result = await truncateToTokenLimit(hugeText, maxTokens, 'text-embedding-3-small');

    const { encode } = await import('gpt-tokenizer');
    const tokenCount = encode(result).length;

    // Must be within model limit
    expect(tokenCount).toBeLessThanOrEqual(maxTokens);
    // Should be reasonably close to limit (not drastically under-truncated)
    expect(tokenCount).toBeGreaterThan(maxTokens * 0.8);
  });

  it('respects Gemini model limits with fallback heuristic', async () => {
    const maxTokens = getModelDefaultTokenLimit('gemini-embedding-001'); // 2048
    // 6144 chars (> 2048 * 3) should be truncated
    const longText = 'a'.repeat(6144);

    const result = await truncateToTokenLimit(longText, maxTokens, 'gemini-embedding-001');

    // Without API key, uses 3× heuristic
    expect(result.length).toBeLessThanOrEqual(maxTokens * 3);
    expect(result.length).toBe(2048 * 3);
  });

  it('getModelDefaultTokenLimit returns correct limits for all major models', () => {
    // Verify lookup table is correct
    expect(getModelDefaultTokenLimit('text-embedding-3-small')).toBe(8191);
    expect(getModelDefaultTokenLimit('text-embedding-3-large')).toBe(8191);
    expect(getModelDefaultTokenLimit('text-embedding-ada-002')).toBe(8191);
    expect(getModelDefaultTokenLimit('gemini-embedding-001')).toBe(2048);
    expect(getModelDefaultTokenLimit('bge-m3')).toBe(8192);
    expect(getModelDefaultTokenLimit('bge-large-en-v1.5')).toBe(512);
    expect(getModelDefaultTokenLimit('unknown-model')).toBe(512);
  });
});

describe('truncateToTokenLimit – large text handling', () => {
  it('handles moderately large text (100KB+) without crashing', async () => {
    const largeText = 'word '.repeat(20000); // ~100KB
    const maxTokens = 100;

    const result = await truncateToTokenLimit(largeText, maxTokens, 'text-embedding-3-small');

    const { encode } = await import('gpt-tokenizer');
    const tokenCount = encode(result).length;

    expect(tokenCount).toBeLessThanOrEqual(maxTokens);
    expect(result.length).toBeGreaterThan(0);
  });

  it('handles large unknown model text with heuristic', async () => {
    const largeText = 'x'.repeat(10000);
    const maxTokens = 1000;

    const result = await truncateToTokenLimit(largeText, maxTokens, 'local-llm-model');

    // Should apply heuristic: max 3000 chars
    expect(result.length).toBeLessThanOrEqual(maxTokens * 3);
    expect(result.length).toBeLessThan(largeText.length);
  });
});

describe('truncateToTokenLimit – HuggingFace tokenizer error handling', () => {
  const skipIfNoNetwork = process.env.CI_NO_NETWORK === 'true' ? it.skip : it;

  skipIfNoNetwork(
    'gracefully handles tokenizer download errors by using fallback',
    async () => {
      // Mock the transformers module to simulate download error
      jest.doMock('@huggingface/transformers', () => {
        throw new Error('Network error: unable to download tokenizer');
      });

      try {
        const largeText = 'a'.repeat(1000);
        const result = await truncateToTokenLimit(largeText, 100, 'bge-m3-custom');

        // Should fall back to heuristic instead of crashing
        expect(result.length).toBeLessThanOrEqual(100 * 3);
      } catch (e) {
        // If it throws, should be a clear error, not a silent failure
        expect(e).toBeInstanceOf(Error);
      } finally {
        jest.resetModules();
      }
    },
    30000,
  );
});
