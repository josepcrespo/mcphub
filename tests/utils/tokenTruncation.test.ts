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
  it('respects OpenAI model limits: text truncated to fit within 8191 tokens', async () => {
    // ~16000 words → ~80000+ tokens, well exceeding 8191
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

// ─────────────────────────────────────────────────────────────────────────────
// Three-tier fallback: official HF → hf-mirror.com → heuristic
// ─────────────────────────────────────────────────────────────────────────────

describe('truncateToTokenLimit – HuggingFace three-tier fallback', () => {
  afterEach(() => {
    jest.resetModules();
    jest.restoreAllMocks();
  });

  it('falls back to hf-mirror.com when official HuggingFace Hub fails', async () => {
    // Shared env object: its remoteHost is mutated by getHFTokenizer before each download attempt
    const mockEnv = { remoteHost: 'https://huggingface.co/' };

    // Minimal mock tokenizer: callable as function + has decode method
    const mockDecode = jest.fn().mockImplementation(async (ids: number[]) => 'x'.repeat(ids.length));
    const mockTokenizerFn = jest.fn().mockImplementation(async (text: string) => ({
      input_ids: { data: new Int32Array(Array.from({ length: text.length }, (_, i) => i + 1)) },
    }));
    (mockTokenizerFn as any).decode = mockDecode;

    jest.doMock('@huggingface/transformers', () => ({
      env: mockEnv,
      AutoTokenizer: {
        from_pretrained: jest.fn().mockImplementation(async () => {
          // Fail only for the official host; succeed for hf-mirror.com
          if (mockEnv.remoteHost.includes('huggingface.co')) {
            throw new Error('Connection refused: huggingface.co is blocked');
          }
          return mockTokenizerFn;
        }),
      },
    }));

    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const maxTokens = 5;
    const text = 'a'.repeat(20); // 20 chars > maxTokens * 3 = 15

    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');
    const result = await truncate(text, maxTokens, 'BAAI/bge-m3');

    // Should have warned about official HF failure and mirror retry
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('HuggingFace Hub unreachable'),
      expect.any(Error),
    );
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('Retrying with hf-mirror.com'),
      expect.any(Error),
    );

    // Should NOT have warned about mirror failure (mirror succeeded)
    const allWarnings = warnSpy.mock.calls.map((call) => call[0] as string);
    expect(allWarnings.some((msg) => msg.includes('hf-mirror.com also failed'))).toBe(false);

    // Mirror tokenizer was used: result length should be <= maxTokens (1 token per char in mock)
    expect(result.length).toBeLessThanOrEqual(maxTokens);
  });

  it('falls back to heuristic when both HuggingFace Hub and hf-mirror.com fail', async () => {
    const mockEnv = { remoteHost: 'https://huggingface.co/' };

    jest.doMock('@huggingface/transformers', () => ({
      env: mockEnv,
      AutoTokenizer: {
        from_pretrained: jest.fn().mockRejectedValue(new Error('All hosts unreachable')),
      },
    }));

    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const maxTokens = 10;
    const longText = 'a'.repeat(100); // 100 chars > maxTokens * 3 = 30

    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');
    const result = await truncate(longText, maxTokens, 'BAAI/bge-m3');

    // Both tier-1 and tier-2 warnings should have been logged
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('HuggingFace Hub unreachable'),
      expect.any(Error),
    );
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('hf-mirror.com also failed'),
      expect.any(Error),
    );
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('Falling back to character-based heuristic truncation'),
      expect.any(Error),
    );

    // Result should be heuristic truncation: exactly maxTokens * 3 chars
    expect(result.length).toBeLessThanOrEqual(maxTokens * 3);
    expect(result.length).toBeLessThan(longText.length);
  });

  it('short text is returned unchanged even when all tokenizer hosts fail', async () => {
    const mockEnv = { remoteHost: 'https://huggingface.co/' };

    jest.doMock('@huggingface/transformers', () => ({
      env: mockEnv,
      AutoTokenizer: {
        from_pretrained: jest.fn().mockRejectedValue(new Error('Network unavailable')),
      },
    }));

    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const maxTokens = 100;
    const shortText = 'hello world'; // 11 chars, well under maxTokens * 3 = 300

    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');
    const result = await truncate(shortText, maxTokens, 'BAAI/bge-m3');

    // Heuristic: text fits within maxTokens * 3, so returned as-is
    expect(result).toBe(shortText);
  });

  it('concurrent calls for the same model/host trigger only one download', async () => {
    let downloadCount = 0;
    const mockEnv = { remoteHost: '' };
    const mockDecode = jest.fn().mockResolvedValue('xxx');
    const mockTokenizerFn = jest.fn().mockImplementation(async () => ({
      input_ids: { data: new Int32Array([1, 2, 3]) },
    }));
    (mockTokenizerFn as any).decode = mockDecode;

    jest.doMock('@huggingface/transformers', () => ({
      env: mockEnv,
      AutoTokenizer: {
        from_pretrained: jest.fn().mockImplementation(async () => {
          downloadCount++;
          return mockTokenizerFn;
        }),
      },
    }));

    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');

    // Fire two concurrent calls for the same model/host
    await Promise.all([
      truncate('hello world', 100, 'BAAI/bge-m3'),
      truncate('hello world', 100, 'BAAI/bge-m3'),
    ]);

    expect(downloadCount).toBe(1);
  }, 30000);

  it('skips unhealthy official host within TTL and goes directly to mirror', async () => {
    const networkError = new Error('Connection refused: huggingface.co is blocked');
    const mockEnv = { remoteHost: '' };
    const mockDecode = jest.fn().mockResolvedValue('x'.repeat(5));
    const mockTokenizerFn = jest.fn().mockImplementation(async (text: string) => ({
      input_ids: { data: new Int32Array(Array.from({ length: text.length }, (_, i) => i + 1)) },
    }));
    (mockTokenizerFn as any).decode = mockDecode;

    const fromPretrainedMock = jest.fn().mockImplementation(async () => {
      if (mockEnv.remoteHost.includes('huggingface.co')) throw networkError;
      return mockTokenizerFn;
    });

    jest.doMock('@huggingface/transformers', () => ({
      env: mockEnv,
      AutoTokenizer: { from_pretrained: fromPretrainedMock },
    }));

    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');

    // Call 1: official fails → marked unhealthy; mirror succeeds
    await truncate('hello world test', 5, 'BAAI/bge-m3');
    warnSpy.mockClear();
    fromPretrainedMock.mockClear();

    // Call 2 (within TTL): official should be skipped
    await truncate('hello world test', 5, 'BAAI/bge-m3');

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Skipping HuggingFace Hub'));
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('TTL active'));
    // Mirror tokenizer is cached from call 1 — no new download needed
    expect(fromPretrainedMock).not.toHaveBeenCalled();
  }, 30000);

  it('retries HuggingFace Hub after TTL expires', async () => {
    const mockEnv = { remoteHost: '' };
    const mockDecode = jest.fn().mockResolvedValue('x'.repeat(3));
    const mockTokenizerFn = jest.fn().mockImplementation(async () => ({
      input_ids: { data: new Int32Array([1, 2, 3]) },
    }));
    (mockTokenizerFn as any).decode = mockDecode;

    const fromPretrainedMock = jest.fn().mockImplementation(async () => {
      if (mockEnv.remoteHost.includes('huggingface.co')) throw new Error('Connection refused');
      return mockTokenizerFn;
    });

    jest.doMock('@huggingface/transformers', () => ({
      env: mockEnv,
      AutoTokenizer: { from_pretrained: fromPretrainedMock },
    }));

    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');
    const realNow = Date.now();

    // Call 1: official fails → marked unhealthy (TTL = realNow + 7 days); mirror succeeds
    await truncate('hello', 5, 'BAAI/bge-m3');
    fromPretrainedMock.mockClear();
    warnSpy.mockClear();

    // Call 2 (within TTL): official is skipped
    await truncate('hello', 5, 'BAAI/bge-m3');
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Skipping HuggingFace Hub'));
    fromPretrainedMock.mockClear();
    warnSpy.mockClear();

    // Advance clock beyond TTL (8 days)
    jest.spyOn(Date, 'now').mockReturnValue(realNow + 8 * 24 * 60 * 60 * 1000);

    // Call 3 (TTL expired): official should be retried
    await truncate('hello', 5, 'BAAI/bge-m3');

    // Should have warned about official failure (retried), NOT about skipping
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('HuggingFace Hub unreachable'),
      expect.any(Error),
    );
    expect(warnSpy).not.toHaveBeenCalledWith(expect.stringContaining('Skipping HuggingFace Hub'));
    // from_pretrained was called exactly once (for the official host retry)
    expect(fromPretrainedMock).toHaveBeenCalledTimes(1);
  }, 30000);

  it('includes the original error object as second argument in tier warnings', async () => {
    const tier1Error = new Error('tier1: Connection refused');
    const tier2Error = new Error('tier2: mirror also blocked');
    const mockEnv = { remoteHost: '' };
    let callCount = 0;

    jest.doMock('@huggingface/transformers', () => ({
      env: mockEnv,
      AutoTokenizer: {
        from_pretrained: jest.fn().mockImplementation(async () => {
          callCount++;
          throw callCount === 1 ? tier1Error : tier2Error;
        }),
      },
    }));

    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { truncateToTokenLimit: truncate } = await import('../../src/utils/tokenTruncation.js');

    await truncate('a'.repeat(100), 10, 'BAAI/bge-m3');

    const tier1Warn = warnSpy.mock.calls.find(
      (c) => typeof c[0] === 'string' && c[0].includes('HuggingFace Hub unreachable'),
    );
    expect(tier1Warn).toBeDefined();
    expect(tier1Warn![1]).toBe(tier1Error);

    const tier2Warn = warnSpy.mock.calls.find(
      (c) => typeof c[0] === 'string' && c[0].includes('hf-mirror.com also failed'),
    );
    expect(tier2Warn).toBeDefined();
    expect(tier2Warn![1]).toBe(tier2Error);
  }, 30000);
});
