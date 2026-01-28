/**
 * Integration tests for vector embedding validation and API interaction
 * 
 * These tests verify the complete integration flow between:
 * - External embedding APIs (OpenAI, LM Studio, LocalAI, Ollama)
 * - Embedding generation with validation layers
 * - Vector database operations
 * - Dimension consistency checks
 * 
 * Tests cover:
 * - Embedding API interaction with different providers
 * - All validation layers (array, zero-vector, range, dimensions)
 * - Fallback mechanisms when validation fails
 * - Database initialization and dimension migration
 * - Text preprocessing and API response handling
 */

jest.mock('openai');
jest.mock('../../src/db/index.js');
jest.mock('../../src/db/connection.js');
jest.mock('../../src/utils/smartRouting.js');
jest.mock('../../src/services/mcpService.js');

import {
  saveToolsAsVectorEmbeddings,
  searchToolsByVector,
  getAllVectorizedTools,
  removeServerToolEmbeddings,
} from '../../src/services/vectorSearchService.js';
import { getRepositoryFactory } from '../../src/db/index.js';
import { getAppDataSource, isDatabaseConnected, initializeDatabase } from '../../src/db/connection.js';
import { getSmartRoutingConfig } from '../../src/utils/smartRouting.js';
import OpenAI from 'openai';
import { Tool } from '../../src/types/index.js';

describe('Vector Embedding Integration Tests', () => {
  let mockDataSource: any;
  let mockRepository: any;
  let mockSmartRoutingConfig: any;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup global.fetch mock
    global.fetch = jest.fn() as jest.Mock;

    // Setup mock DataSource
    mockDataSource = {
      query: jest.fn(),
      isInitialized: true,
    };

    // Setup mock Repository
    mockRepository = {
      saveEmbedding: jest.fn().mockResolvedValue(undefined),
      searchByText: jest.fn().mockResolvedValue([]),
      searchSimilar: jest.fn().mockResolvedValue([]),
      deleteByServerName: jest.fn().mockResolvedValue(0),
    };

    // Setup mock SmartRouting Config
    mockSmartRoutingConfig = {
      enabled: true,
      openaiApiKey: 'test-api-key',
      openaiApiBaseUrl: 'http://localhost:8000',
      openaiApiEmbeddingModel: 'text-embedding-3-small',
      dbUrl: 'postgresql://localhost/mcphub',
    };

    // Setup mocks
    (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
    (getRepositoryFactory as jest.Mock).mockReturnValue(() => mockRepository);
    (getAppDataSource as jest.Mock).mockReturnValue(mockDataSource);
    (isDatabaseConnected as jest.Mock).mockReturnValue(true);
    (initializeDatabase as jest.Mock).mockResolvedValue(undefined);

    // Mock OpenAI client
    (OpenAI as unknown as jest.Mock).mockImplementation(() => ({
      embeddings: {
        create: jest.fn().mockResolvedValue({
          data: [
            {
              embedding: new Array(1536).fill(0.5), // Valid embedding
            },
          ],
        }),
      },
    }));

    // Suppress console output during tests
    jest.spyOn(console, 'log').mockImplementation(() => {});
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Embedding API interaction and validation', () => {
    it('should save tools by generating valid embeddings', async () => {
      mockDataSource.query.mockResolvedValue([]);
      
      // Mock fetch response for embedding API
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });

      const tools: Tool[] = [
        {
          name: 'search_tool',
          description: 'Search for documents',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string' },
            },
          },
        },
      ];

      await saveToolsAsVectorEmbeddings('docs_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalledWith(
        'tool',
        'docs_server:search_tool',
        expect.stringContaining('search_tool'),
        expect.any(Array),
        expect.any(Object),
        'text-embedding-3-small'
      );
    });

    it('should handle different API base URLs', async () => {
      mockSmartRoutingConfig.openaiApiBaseUrl = 'http://localhost:11434';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1024).fill(0.3) }],
        }),
      });

      const tools: Tool[] = [
        {
          name: 'test',
          description: 'test',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('server1', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should normalize base URLs with trailing slashes', async () => {
      mockSmartRoutingConfig.openaiApiBaseUrl = 'http://localhost:8000/';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });

      const tools: Tool[] = [
        {
          name: 'test',
          description: 'test',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('server1', tools);

      // Should still save successfully despite trailing slash
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle API errors and use fallback', async () => {
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));

      const tools: Tool[] = [
        {
          name: 'failing_tool',
          description: 'Tool that will use fallback',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('server1', tools);

      // Should still save with fallback embedding
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should validate embedding dimensions match expected', async () => {
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });

      const tools: Tool[] = [
        {
          name: 'dimension_test',
          description: 'Test dimension validation',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('server1', tools);

      // Embedding should have correct dimensions (1536 for text-embedding-3-small)
      const callArgs = mockRepository.saveEmbedding.mock.calls[0];
      const embedding = callArgs[3];
      expect(embedding).toHaveLength(1536);
    });

    it('should handle non-OK HTTP responses from API', async () => {
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 500,
        text: jest.fn().mockResolvedValue('Server error'),
      });

      const tools: Tool[] = [
        {
          name: 'error_tool',
          description: 'Tool for error handling',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('server1', tools);

      // Should use fallback and still save
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle malformed JSON responses', async () => {
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockRejectedValue(new Error('Invalid JSON')),
      });

      const tools: Tool[] = [
        {
          name: 'json_error_tool',
          description: 'Test JSON error',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('server1', tools);

      // Should use fallback
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('Validation functions coverage through embedding generation', () => {
    it('should validate embeddings are non-zero vectors', async () => {
      mockDataSource.query.mockResolvedValue([]);

      // Mock API to return zero vector
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0) }], // All zeros
        }),
      });

      const tools: Tool[] = [
        {
          name: 'zero_vector_test',
          description: 'Test zero vector detection',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('zero_server', tools);

      // Should use fallback instead of zero vector
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
      const embedding = mockRepository.saveEmbedding.mock.calls[0][3];
      // Should not be all zeros (using fallback)
      const hasNonZero = embedding.some((v: number) => v !== 0);
      expect(hasNonZero).toBe(true);
    });

    it('should validate embedding values are in expected range', async () => {
      mockDataSource.query.mockResolvedValue([]);

      // Mock API to return out-of-range values
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(10) }], // Out of range
        }),
      });

      const tools: Tool[] = [
        {
          name: 'range_test',
          description: 'Test value range validation',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('range_server', tools);

      // Should use fallback
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should validate embedding is an array', async () => {
      mockDataSource.query.mockResolvedValue([]);

      // Mock API to return non-array
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: 'not an array' }], // Not an array
        }),
      });

      const tools: Tool[] = [
        {
          name: 'array_test',
          description: 'Test array validation',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('array_validation_server', tools);

      // Should use fallback
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle empty embedding array', async () => {
      mockDataSource.query.mockResolvedValue([]);

      // Mock API to return empty array
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: [] }], // Empty array
        }),
      });

      const tools: Tool[] = [
        {
          name: 'empty_array_test',
          description: 'Test empty array handling',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('empty_array_server', tools);

      // Should use fallback
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('API Response extraction and validation', () => {
    it('should handle API response without data field', async () => {
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          // Missing 'data' field
          result: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });

      const tools: Tool[] = [
        {
          name: 'no_data_field_test',
          description: 'Test missing data field',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('no_data_server', tools);

      // Should use fallback
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle API response with empty data array', async () => {
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [], // Empty data array
        }),
      });

      const tools: Tool[] = [
        {
          name: 'empty_data_test',
          description: 'Test empty data array',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('empty_data_server', tools);

      // Should use fallback
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle API response without embedding in first item', async () => {
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [
            {
              // Missing 'embedding' field
              object: 'embedding',
            },
          ],
        }),
      });

      const tools: Tool[] = [
        {
          name: 'no_embedding_field_test',
          description: 'Test missing embedding field',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('no_embedding_server', tools);

      // Should use fallback
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('OpenAI API vs Fetch API strategy selection', () => {
    it('should use fetch API for non-official OpenAI endpoints', async () => {
      mockSmartRoutingConfig.openaiApiKey = 'key-123';
      mockSmartRoutingConfig.openaiApiBaseUrl = 'http://localhost:8000';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      mockDataSource.query.mockResolvedValue([]);

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'fetch_api_test',
          description: 'Should use fetch for local API',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('local_api_server', tools);

      // Fetch should be called for non-official endpoints
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/embeddings'),
        expect.any(Object)
      );
    });

    it('should handle Nomic Embed Text task_type parameter', async () => {
      mockSmartRoutingConfig.openaiApiEmbeddingModel = 'nomic-embed-text-v1.5';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      mockDataSource.query.mockResolvedValue([]);

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(768).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'nomic_test',
          description: 'Test nomic task_type parameter',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('nomic_server', tools);

      // Check that fetch was called with task_type for Nomic
      expect(fetchMock).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('task_type'),
        })
      );
    });
  });

  describe('Embedding dimension consistency verification', () => {
    it('should detect dimension mismatches from test embedding', async () => {
      mockDataSource.query.mockResolvedValue([]);

      // First call returns 1536, second call returns 3072 (mismatch)
      const fetchMock = jest.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: jest.fn().mockResolvedValue({
            data: [{ embedding: new Array(1536).fill(0.5) }], // Test embedding
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: jest.fn().mockResolvedValue({
            data: [{ embedding: new Array(3072).fill(0.5) }], // Tool embedding with different dimensions
          }),
        });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'mismatch_test',
          description: 'Test dimension mismatch detection',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('mismatch_server', tools);

      // Tool with mismatched dimensions should be skipped
      expect(mockRepository.saveEmbedding).not.toHaveBeenCalled();
    });
  });

  describe('Smart routing configuration handling', () => {
    it('should normalize API base URLs before using them', async () => {
      mockSmartRoutingConfig.openaiApiBaseUrl = 'http://localhost:11434/';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      mockDataSource.query.mockResolvedValue([]);

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'url_normalization_test',
          description: 'Test URL normalization',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('url_test_server', tools);

      // Fetch should be called without double slashes
      expect(fetchMock).toHaveBeenCalledWith(
        'http://localhost:11434/embeddings', // No double slashes
        expect.any(Object)
      );
    });

    it('should handle configuration without trailing slashes', async () => {
      mockSmartRoutingConfig.openaiApiBaseUrl = 'http://localhost:8000';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      mockDataSource.query.mockResolvedValue([]);

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'no_slash_test',
          description: 'Test without trailing slash',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('no_slash_server', tools);

      expect(fetchMock).toHaveBeenCalled();
    });
  });

  describe('Text preprocessing and truncation', () => {
    it('should remove newline characters from text before embedding', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'multiline_tool',
          description: 'Tool with\nmultiple\nlines',
          inputSchema: {
            type: 'object',
            properties: {
              param1: { type: 'string', description: 'Line 1\nLine 2' },
            },
          },
        },
      ];

      await saveToolsAsVectorEmbeddings('multiline_server', tools);

      // Check that the request body has spaces instead of newlines
      const callArgs = fetchMock.mock.calls[0][1];
      const body = JSON.parse(callArgs.body);
      expect(body.input).not.toContain('\n');
    });

    it('should truncate very long text before sending to API', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const longText = 'a'.repeat(10000); // Much longer than 8000 limit
      const tools: Tool[] = [
        {
          name: 'long_tool',
          description: longText,
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('long_text_server', tools);

      // Check that the text was truncated
      const callArgs = fetchMock.mock.calls[0][1];
      const body = JSON.parse(callArgs.body);
      expect(body.input.length).toBeLessThanOrEqual(8000);
    });
  });

  describe('Database vector dimension migration scenarios', () => {
    it('should handle dimension query returning no results', async () => {
      mockDataSource.query
        .mockResolvedValueOnce([]) // Dimension check returns empty
        .mockResolvedValueOnce(undefined); // Should continue with save

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'dim_query_test',
          description: 'Test dimension query handling',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('dimension_query_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle ALTER TABLE query for dimension update', async () => {
      // Simulate a scenario where dimensions need to be updated
      mockDataSource.query
        .mockResolvedValueOnce([{ dimensions: 1024 }]) // Current dimension (different from 1536)
        .mockResolvedValueOnce([]) // Check for record dimensions
        .mockResolvedValueOnce(undefined) // DROP INDEX
        .mockResolvedValueOnce(undefined) // ALTER TABLE
        .mockResolvedValueOnce(undefined) // CREATE INDEX
        .mockResolvedValueOnce(undefined); // ALTER TABLE for dimensions

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }], // 1536 dimensions
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'dimension_migration_test',
          description: 'Test dimension migration',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('migration_server', tools);

      // Should attempt ALTER TABLE for migration
      expect(mockDataSource.query).toHaveBeenCalledWith(
        expect.stringContaining('ALTER TABLE')
      );
    });

    it('should handle DROP INDEX when creating new indices', async () => {
      mockDataSource.query
        .mockResolvedValueOnce([]) // Dimension check
        .mockResolvedValueOnce(undefined); // DROP INDEX

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'drop_index_test',
          description: 'Test DROP INDEX',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('drop_index_server', tools);

      expect(mockDataSource.query).toHaveBeenCalledWith(
        expect.stringContaining('DROP INDEX')
      );
    });

    it('should handle DELETE FROM vector_embeddings for dimension mismatch cleanup', async () => {
      mockDataSource.query
        .mockResolvedValueOnce([{ dimensions: 1024 }]) // Current dimension
        .mockResolvedValueOnce([{ dimensions: 1024, count: 5, model: 'old-model' }]) // Records with different dimensions
        .mockResolvedValueOnce(undefined) // DROP INDEX
        .mockResolvedValueOnce(undefined) // DELETE FROM vector_embeddings (clearMismatchedVectorData)
        .mockResolvedValueOnce(undefined) // ALTER TABLE
        .mockResolvedValueOnce(undefined); // CREATE INDEX

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }], // New dimensions: 1536
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'clear_mismatched_test',
          description: 'Test clearMismatchedVectorData',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('clear_mismatched_server', tools);

      // Should call DELETE for mismatched dimensions
      expect(mockDataSource.query).toHaveBeenCalledWith(
        expect.stringContaining('DELETE FROM vector_embeddings'),
        expect.any(Array)
      );
    });
  });

  describe('Complete integration scenarios', () => {
    it('should handle complete workflow with multiple tools and dimension changes', async () => {
      // Simulate: save tools -> search -> remove -> verify
      mockDataSource.query
        .mockResolvedValueOnce([]) // For saveTools dimension check
        .mockResolvedValueOnce([]) // For getAllVectorizedTools dimension check
        .mockResolvedValueOnce([]); // For after remove

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      // Step 1: Save tools
      const tools: Tool[] = [
        {
          name: 'integration_tool_1',
          description: 'Integration test tool 1',
          inputSchema: { type: 'object' },
        },
        {
          name: 'integration_tool_2',
          description: 'Integration test tool 2',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('integration_server', tools);
      expect(mockRepository.saveEmbedding).toHaveBeenCalledTimes(2);

      // Step 2: Search
      mockRepository.searchByText.mockResolvedValue([
        {
          similarity: 0.95,
          embedding: {
            text_content: 'integration_tool_1',
            metadata: JSON.stringify({
              serverName: 'integration_server',
              toolName: 'integration_tool_1',
              description: 'Integration test tool 1',
              inputSchema: {},
            }),
          },
        },
      ]);

      const searchResults = await searchToolsByVector('integration');
      expect(searchResults).toHaveLength(1);

      // Step 3: Remove
      mockRepository.deleteByServerName.mockResolvedValue(2);
      await removeServerToolEmbeddings('integration_server');
      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('integration_server');
    });
  });

  describe('Embedding dimension validation (validateEmbeddingDimensions)', () => {
    it('should detect and handle incorrect dimensions from API and use fallback', async () => {
      // Setup: Configure model as text-embedding-3-small (expected: 1536 dimensions)
      mockSmartRoutingConfig.openaiApiEmbeddingModel = 'text-embedding-3-small';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      
      mockDataSource.query.mockResolvedValue([]);

      // Mock console.error to verify validation error messages
      const consoleErrorSpy = jest.spyOn(console, 'error');

      // Mock fetch to return WRONG dimensions (768 instead of 1536)
      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [
            {
              embedding: new Array(768).fill(0.5), // Wrong dimensions!
            },
          ],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'dimension_mismatch_tool',
          description: 'Tool to test dimension validation',
          inputSchema: { type: 'object' },
        },
      ];

      // Execute: Save tools with wrong dimensions
      await saveToolsAsVectorEmbeddings('dimension_test_server', tools);

      // Verify: validateEmbeddingDimensions was triggered and logged error
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('CRITICAL: Embedding API returned unexpected dimensions!')
      );
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Expected dimensions: 1536')
      );
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Actual dimensions: 768')
      );

      // Verify: Fallback embedding was used (100 dimensions)
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
      const callArgs = mockRepository.saveEmbedding.mock.calls[0];
      const embeddingArray = callArgs[3]; // 4th argument is the embedding array
      expect(embeddingArray).toHaveLength(100); // Fallback uses 100 dimensions
    });

    it('should detect dimensions mismatch for text-embedding-3-large', async () => {
      mockSmartRoutingConfig.openaiApiEmbeddingModel = 'text-embedding-3-large';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      
      mockDataSource.query.mockResolvedValue([]);

      const consoleErrorSpy = jest.spyOn(console, 'error');

      // Return 1536 dimensions instead of expected 3072
      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'large_model_test',
          description: 'Test large model dimension mismatch',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      // Verify dimension mismatch was detected
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Expected dimensions: 3072')
      );
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Actual dimensions: 1536')
      );

      // Fallback should be used
      const callArgs = mockRepository.saveEmbedding.mock.calls[0];
      expect(callArgs[3]).toHaveLength(100);
    });

    it('should detect dimensions mismatch for bge-m3 model', async () => {
      mockSmartRoutingConfig.openaiApiEmbeddingModel = 'bge-m3';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      
      mockDataSource.query.mockResolvedValue([]);

      const consoleErrorSpy = jest.spyOn(console, 'error');

      // Return wrong dimensions (512 instead of 1024)
      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(512).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'bge_test',
          description: 'Test BGE model dimension mismatch',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('bge_server', tools);

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Expected dimensions: 1024')
      );
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Actual dimensions: 512')
      );
    });

    it('should detect dimensions mismatch for nomic-embed-text model', async () => {
      mockSmartRoutingConfig.openaiApiEmbeddingModel = 'nomic-embed-text-v1.5';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      
      mockDataSource.query.mockResolvedValue([]);

      const consoleErrorSpy = jest.spyOn(console, 'error');

      // Return wrong dimensions (384 instead of 768)
      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(384).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'nomic_test',
          description: 'Test Nomic model dimension mismatch',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('nomic_server', tools);

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Expected dimensions: 768')
      );
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Actual dimensions: 384')
      );
    });

    it('should verify all validation layers are called in correct order', async () => {
      mockSmartRoutingConfig.openaiApiEmbeddingModel = 'text-embedding-3-small';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      
      mockDataSource.query.mockResolvedValue([]);

      const consoleErrorSpy = jest.spyOn(console, 'error');
      const consoleLogSpy = jest.spyOn(console, 'log');

      // Return array with wrong dimensions
      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(768).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'validation_order_test',
          description: 'Test validation order',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('validation_server', tools);

      // Verify logs show validation progression:
      // 1. Received embedding (validateEmbeddingArray passes)
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Received embedding:')
      );

      // 2. Array validation passes (no error about "not an array")
      expect(consoleErrorSpy).not.toHaveBeenCalledWith(
        expect.stringContaining('not an array or empty')
      );

      // 3. Zero-vector validation passes (no error about "zeros")
      expect(consoleErrorSpy).not.toHaveBeenCalledWith(
        expect.stringContaining('vector full of zeros')
      );

      // 4. Range validation passes (no error about "out of expected range")
      expect(consoleErrorSpy).not.toHaveBeenCalledWith(
        expect.stringContaining('out of expected range')
      );

      // 5. Dimension validation FAILS (error logged)
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('CRITICAL: Embedding API returned unexpected dimensions!')
      );

      // 6. Fallback is used (no success message for API embedding)
      expect(consoleLogSpy).not.toHaveBeenCalledWith(
        expect.stringMatching(/✅ Embedding generated: 768 dimensions/)
      );
    });

    it('should include curl command in error message for debugging', async () => {
      mockSmartRoutingConfig.openaiApiEmbeddingModel = 'text-embedding-3-small';
      mockSmartRoutingConfig.openaiApiBaseUrl = 'http://localhost:8000';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      
      mockDataSource.query.mockResolvedValue([]);

      const consoleErrorSpy = jest.spyOn(console, 'error');

      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(768).fill(0.5) }],
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'curl_debug_test',
          description: 'Test curl command in error',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('curl_server', tools);

      // Verify error includes debugging curl command
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('curl -s http://localhost:8000/embeddings')
      );
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('"model":"text-embedding-3-small"')
      );
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining("jq '.data[0].embedding | length'")
      );
    });

    it('should handle zero-vector with wrong dimensions (multiple validation failures)', async () => {
      mockSmartRoutingConfig.openaiApiEmbeddingModel = 'text-embedding-3-small';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);
      
      mockDataSource.query.mockResolvedValue([]);

      const consoleErrorSpy = jest.spyOn(console, 'error');

      // Return wrong dimensions AND all zeros
      const fetchMock = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({
          data: [{ embedding: new Array(768).fill(0) }], // Wrong dimensions + all zeros
        }),
      });
      global.fetch = fetchMock;

      const tools: Tool[] = [
        {
          name: 'multi_fail_test',
          description: 'Test multiple validation failures',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('multi_fail_server', tools);

      // Both validations should fail, but zero-vector check comes first
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('vector full of zeros')
      );
      
      // Dimension check should not be reached because zero-vector check fails first
      // and triggers fallback immediately
    });
  });
});
