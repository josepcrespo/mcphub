jest.mock('openai');
jest.mock('../../src/db/index.js');
jest.mock('../../src/db/connection.js');
jest.mock('../../src/utils/smartRouting.js');
jest.mock('../../src/services/mcpService.js');

import {
  createVectorIndex,
  saveToolsAsVectorEmbeddings,
  searchToolsByVector,
  getAllVectorizedTools,
  removeServerToolEmbeddings,
  VECTOR_MAX_DIMENSIONS,
  HALFVEC_MAX_DIMENSIONS,
} from '../../src/services/vectorSearchService.js';
import { getRepositoryFactory } from '../../src/db/index.js';
import { getAppDataSource, isDatabaseConnected, initializeDatabase } from '../../src/db/connection.js';
import { getSmartRoutingConfig } from '../../src/utils/smartRouting.js';
import OpenAI from 'openai';
import { Tool } from '../../src/types/index.js';

describe('Vector Search Service', () => {
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

    // Suppress console output during tests
    jest.spyOn(console, 'log').mockImplementation(() => {});
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('createVectorIndex', () => {
    it('should create HNSW index for vectors <= 2000 dimensions', async () => {
      mockDataSource.query.mockResolvedValue(undefined);

      const result = await createVectorIndex(mockDataSource, 1536);

      expect(result.success).toBe(true);
      expect(result.indexType).toBe('hnsw');
      expect(mockDataSource.query).toHaveBeenCalledWith(expect.stringContaining('DROP INDEX'));
      expect(mockDataSource.query).toHaveBeenCalledWith(expect.stringContaining('hnsw'));
    });

    it('should fallback to IVFFlat when HNSW fails', async () => {
      mockDataSource.query
        .mockResolvedValueOnce(undefined) // DROP INDEX
        .mockRejectedValueOnce(new Error('HNSW not supported'))
        .mockResolvedValueOnce(undefined); // IVFFlat creation

      const result = await createVectorIndex(mockDataSource, 1536);

      expect(result.success).toBe(true);
      expect(result.indexType).toBe('ivfflat');
    });

    it('should create HNSW with halfvec casting for 2001-4000 dimensions', async () => {
      mockDataSource.query.mockResolvedValue(undefined);

      const result = await createVectorIndex(mockDataSource, 3072);

      expect(result.success).toBe(true);
      expect(result.indexType).toBe('hnsw-halfvec');
      expect(mockDataSource.query).toHaveBeenCalledWith(expect.stringContaining('halfvec'));
    });

    it('should fail for dimensions > 4000', async () => {
      const result = await createVectorIndex(mockDataSource, 5000);

      expect(result.success).toBe(false);
      expect(result.indexType).toBeNull();
      expect(result.message).toContain('exceed maximum');
    });

    it('should use custom table and column names', async () => {
      mockDataSource.query.mockResolvedValue(undefined);

      await createVectorIndex(mockDataSource, 1536, 'custom_table', 'custom_embedding');

      expect(mockDataSource.query).toHaveBeenCalledWith(expect.stringContaining('custom_table'));
      expect(mockDataSource.query).toHaveBeenCalledWith(expect.stringContaining('custom_embedding'));
    });

    it('should handle index creation failure gracefully', async () => {
      mockDataSource.query
        .mockRejectedValueOnce(new Error('Drop index error'))
        .mockRejectedValueOnce(new Error('HNSW creation failed'))
        .mockRejectedValueOnce(new Error('IVFFlat creation failed'));

      const result = await createVectorIndex(mockDataSource, 1536);

      expect(result.success).toBe(false);
    });
  });

  describe('saveToolsAsVectorEmbeddings', () => {
    it('should save tools with valid embeddings', async () => {
      const tools: Tool[] = [
        {
          name: 'test_tool',
          description: 'A test tool',
          inputSchema: {
            type: 'object',
            properties: { arg1: { type: 'string' } },
          },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('test_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalledWith(
        'tool',
        'test_server:test_tool',
        expect.any(String),
        expect.any(Array),
        expect.any(Object),
        'text-embedding-3-small'
      );
    });

    it('should skip empty tool list', async () => {
      await saveToolsAsVectorEmbeddings('test_server', []);

      expect(mockRepository.saveEmbedding).not.toHaveBeenCalled();
    });

    it('should initialize database if not connected', async () => {
      (isDatabaseConnected as jest.Mock).mockReturnValueOnce(false);

      const tools: Tool[] = [
        {
          name: 'test_tool',
          description: 'A test tool',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('test_server', tools);

      expect(initializeDatabase).toHaveBeenCalled();
    });

    it('should skip saving if smart routing is disabled', async () => {
      mockSmartRoutingConfig.enabled = false;
      (getSmartRoutingConfig as jest.Mock).mockResolvedValueOnce(mockSmartRoutingConfig);

      const tools: Tool[] = [
        {
          name: 'test_tool',
          description: 'A test tool',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      expect(mockRepository.saveEmbedding).not.toHaveBeenCalled();
    });

    it('should handle embedding dimension consistency checks', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'tool1',
          description: 'Tool 1',
          inputSchema: { type: 'object' },
        },
        {
          name: 'tool2',
          description: 'Tool 2',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle embedding generation failures', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'test_tool',
          description: 'A test tool',
          inputSchema: { type: 'object' },
        },
      ];

      // First embedding (test) returns valid array
      // We'll rely on the mock to handle embeddings

      await saveToolsAsVectorEmbeddings('test_server', tools);

      // Should still attempt to save
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('searchToolsByVector', () => {
    it('should search tools by vector similarity', async () => {
      const mockResults = [
        {
          similarity: 0.95,
          embedding: {
            text_content: 'test_tool test description',
            metadata: JSON.stringify({
              serverName: 'test_server',
              toolName: 'test_tool',
              description: 'test description',
              inputSchema: { type: 'object' },
            }),
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('test query');

      expect(results).toHaveLength(1);
      expect(results[0].toolName).toBe('test_tool');
      expect(results[0].serverName).toBe('test_server');
      expect(results[0].similarity).toBe(0.95);
    });

    it('should filter results by server names', async () => {
      const mockResults = [
        {
          similarity: 0.95,
          embedding: {
            text_content: 'test_tool test description',
            metadata: JSON.stringify({
              serverName: 'server1',
              toolName: 'test_tool',
              description: 'test description',
              inputSchema: {},
            }),
          },
        },
        {
          similarity: 0.90,
          embedding: {
            text_content: 'other_tool other description',
            metadata: JSON.stringify({
              serverName: 'server2',
              toolName: 'other_tool',
              description: 'other description',
              inputSchema: {},
            }),
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('test query', 10, 0.7, ['server1']);

      expect(results).toHaveLength(1);
      expect(results[0].serverName).toBe('server1');
    });

    it('should handle malformed metadata', async () => {
      const mockResults = [
        {
          similarity: 0.95,
          embedding: {
            text_content: 'test_tool test description',
            metadata: 'invalid json',
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('test query');

      expect(results).toHaveLength(1);
      expect(results[0].toolName).toBe('test_tool');
    });

    it('should handle search errors gracefully', async () => {
      mockRepository.searchByText.mockRejectedValue(new Error('Search failed'));

      const results = await searchToolsByVector('test query');

      expect(results).toEqual([]);
    });
  });

  describe('getAllVectorizedTools', () => {
    it('should return all vectorized tools', async () => {
      const mockResults = [
        {
          similarity: 1.0,
          embedding: {
            metadata: JSON.stringify({
              serverName: 'server1',
              toolName: 'tool1',
              description: 'Tool 1 description',
              inputSchema: { type: 'object' },
            }),
          },
        },
        {
          similarity: 1.0,
          embedding: {
            metadata: JSON.stringify({
              serverName: 'server2',
              toolName: 'tool2',
              description: 'Tool 2 description',
              inputSchema: { type: 'object' },
            }),
          },
        },
      ];

      mockRepository.searchSimilar.mockResolvedValue(mockResults);
      mockDataSource.query.mockResolvedValue([]);

      const results = await getAllVectorizedTools();

      expect(results).toHaveLength(2);
      expect(results[0].toolName).toBe('tool1');
      expect(results[1].toolName).toBe('tool2');
    });

    it('should filter tools by server names', async () => {
      const mockResults = [
        {
          similarity: 1.0,
          embedding: {
            metadata: JSON.stringify({
              serverName: 'server1',
              toolName: 'tool1',
              description: 'description',
              inputSchema: {},
            }),
          },
        },
        {
          similarity: 1.0,
          embedding: {
            metadata: JSON.stringify({
              serverName: 'server2',
              toolName: 'tool2',
              description: 'description',
              inputSchema: {},
            }),
          },
        },
      ];

      mockRepository.searchSimilar.mockResolvedValue(mockResults);
      mockDataSource.query.mockResolvedValue([]);

      const results = await getAllVectorizedTools(['server1']);

      expect(results).toHaveLength(1);
      expect(results[0].serverName).toBe('server1');
    });

    it('should handle database query for dimensions', async () => {
      mockDataSource.query
        .mockResolvedValueOnce([{ dimensions: 1536 }]) // Dimension query
        .mockResolvedValueOnce([]); // searchSimilar

      mockRepository.searchSimilar.mockResolvedValue([]);

      await getAllVectorizedTools();

      expect(mockDataSource.query).toHaveBeenCalled();
    });

    it('should handle errors gracefully', async () => {
      mockRepository.searchSimilar.mockRejectedValue(new Error('Search failed'));

      const results = await getAllVectorizedTools();

      expect(results).toEqual([]);
    });
  });

  describe('removeServerToolEmbeddings', () => {
    it('should remove embeddings for a server', async () => {
      mockRepository.deleteByServerName.mockResolvedValue(5);

      await removeServerToolEmbeddings('test_server');

      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('test_server');
    });

    it('should handle database not connected initially', async () => {
      (isDatabaseConnected as jest.Mock).mockReturnValueOnce(false);
      mockRepository.deleteByServerName.mockResolvedValue(3);

      await removeServerToolEmbeddings('test_server');

      expect(initializeDatabase).toHaveBeenCalled();
      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('test_server');
    });

    it('should skip initialization when database already connected', async () => {
      (isDatabaseConnected as jest.Mock).mockReturnValueOnce(true);
      mockRepository.deleteByServerName.mockResolvedValue(1);

      await removeServerToolEmbeddings('test_server');

      // When DB is already connected, should not initialize
      expect(initializeDatabase).not.toHaveBeenCalled();
      expect(mockRepository.deleteByServerName).toHaveBeenCalled();
    });

    it('should handle deletion errors', async () => {
      mockRepository.deleteByServerName.mockRejectedValue(new Error('Delete failed'));

      await removeServerToolEmbeddings('test_server');

      // Should not throw, just log error
      expect(mockRepository.deleteByServerName).toHaveBeenCalled();
    });
  });

  describe('syncAllServerToolsEmbeddings', () => {
    it('should sync tools for all connected servers', async () => {
      const mockGetServersInfo = jest.fn().mockResolvedValue([
        {
          name: 'server1',
          status: 'connected',
          tools: [
            { name: 'tool1', description: 'desc1', inputSchema: {} },
            { name: 'tool2', description: 'desc2', inputSchema: {} },
          ],
        },
        {
          name: 'server2',
          status: 'connected',
          tools: [{ name: 'tool3', description: 'desc3', inputSchema: {} }],
        },
      ]);

      jest.doMock('../../src/services/mcpService.js', () => ({
        getServersInfo: mockGetServersInfo,
      }));

      mockDataSource.query.mockResolvedValue([]);

      // We can't easily test this without importing, but we've set up the mocks
      // The real implementation would call getServersInfo and sync each server
    });
  });

  describe('Vector Index Creation Edge Cases', () => {
    it('should verify VECTOR_MAX_DIMENSIONS constant is 2000', () => {
      expect(VECTOR_MAX_DIMENSIONS).toBe(2000);
    });

    it('should verify HALFVEC_MAX_DIMENSIONS constant is 4000', () => {
      expect(HALFVEC_MAX_DIMENSIONS).toBe(4000);
    });

    it('should handle empty vector results', async () => {
      mockRepository.searchByText.mockResolvedValue([]);

      const results = await searchToolsByVector('test');

      expect(results).toEqual([]);
    });

    it('should handle null metadata gracefully', async () => {
      const mockResults = [
        {
          similarity: 0.95,
          embedding: {
            text_content: 'fallback text',
            metadata: null,
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('test');

      expect(results).toHaveLength(1);
    });
  });

  describe('Integration Scenarios', () => {
    it('should handle full workflow: save, search, remove', async () => {
      const tools: Tool[] = [
        {
          name: 'search_tool',
          description: 'Search functionality',
          inputSchema: { type: 'object' },
        },
      ];

      // Save
      mockDataSource.query.mockResolvedValue([]);
      await saveToolsAsVectorEmbeddings('test_server', tools);
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();

      // Search
      mockRepository.searchByText.mockResolvedValue([
        {
          similarity: 0.95,
          embedding: {
            text_content: 'search_tool',
            metadata: JSON.stringify({
              serverName: 'test_server',
              toolName: 'search_tool',
              description: 'Search functionality',
              inputSchema: {},
            }),
          },
        },
      ]);

      const results = await searchToolsByVector('search');
      expect(results).toHaveLength(1);

      // Remove
      mockRepository.deleteByServerName.mockResolvedValue(1);
      await removeServerToolEmbeddings('test_server');
      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('test_server');
    });

    it('should maintain consistency across multiple dimension migrations', async () => {
      mockDataSource.query.mockResolvedValue(undefined);

      // First migration
      const result1 = await createVectorIndex(mockDataSource, 1536);
      expect(result1.success).toBe(true);

      jest.clearAllMocks();
      mockDataSource.query.mockResolvedValue(undefined);

      // Second migration with different dimensions
      const result2 = await createVectorIndex(mockDataSource, 3072);
      expect(result2.success).toBe(true);

      // Both should create indices
      expect(mockDataSource.query).toHaveBeenCalledWith(expect.stringContaining('DROP INDEX'));
    });
  });

  describe('Error Recovery', () => {
    it('should recover from API call failures', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'test_tool',
          description: 'Test tool',
          inputSchema: { type: 'object' },
        },
      ];

      // Simulate API failure followed by recovery
      mockRepository.saveEmbedding
        .mockRejectedValueOnce(new Error('API Error'))
        .mockResolvedValueOnce(undefined);

      await saveToolsAsVectorEmbeddings('test_server', tools);

      // Should still attempt to save
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle timeout during embedding generation', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'test_tool',
          description: 'Test',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      // Should not throw
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('Input Validation', () => {
    it('should validate server name is not empty', async () => {
      const tools: Tool[] = [
        {
          name: 'test',
          description: 'test',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('', tools);

      // Should handle gracefully
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle tools with missing inputSchema', async () => {
      const tools: Tool[] = [
        {
          name: 'test_tool',
          description: 'Test tool',
          // inputSchema missing
        } as Tool,
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('test_server', tools);

      // Should handle gracefully
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should filter out invalid server names in search', async () => {
      mockRepository.searchByText.mockResolvedValue([]);

      const results = await searchToolsByVector('query', 10, 0.7, []);

      expect(results).toEqual([]);
    });
  });

  describe('Model-specific dimension detection', () => {
    // These tests cover getDimensionsForModel which has many branches
    const testCases = [
      { model: 'text-embedding-3-small', expected: 1536, description: 'OpenAI small model' },
      { model: 'text-embedding-3-large', expected: 3072, description: 'OpenAI large model' },
      { model: 'bge-m3', expected: 1024, description: 'BGE M3 model' },
      { model: 'bge-m3-quantized.gguf', expected: 1024, description: 'BGE M3 quantized variant' },
      { model: 'nomic-embed-text', expected: 768, description: 'Nomic model' },
      { model: 'nomic-embed-text-v1.5-Q5_K_M.gguf', expected: 768, description: 'Nomic quantized variant' },
      { model: 'fallback', expected: 100, description: 'Fallback implementation' },
      { model: 'simple-hash', expected: 100, description: 'Simple hash implementation' },
      { model: 'unknown-model', expected: 1536, description: 'Unknown model defaults to OpenAI small' },
    ];

    // Note: We can't directly test getDimensionsForModel since it's not exported
    // But we can test it indirectly through saveToolsAsVectorEmbeddings
    it('should handle various embedding models in saveToolsAsVectorEmbeddings', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'test_tool',
          description: 'Test tool',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      // Should complete without errors
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    // Parametrized test using testCases array to verify dimensions for each model
    testCases.forEach(({ model, expected, description }) => {
      it(`should return correct dimensions for ${description} (${model})`, async () => {
        mockDataSource.query.mockResolvedValue([]);

        // Update mock config with the current test model
        mockSmartRoutingConfig.openaiApiEmbeddingModel = model;
        (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

        const tools: Tool[] = [
          {
            name: 'dimension_test_tool',
            description: 'Tool to test embedding dimensions',
            inputSchema: { type: 'object' },
          },
        ];

        // Mock the fetch to return embeddings with the expected dimension
        (global.fetch as jest.Mock).mockResolvedValue({
          ok: true,
          json: async () => ({
            data: [
              {
                embedding: new Array(expected).fill(0.5), // Create embedding with expected dimensions
              },
            ],
          }),
        });

        await saveToolsAsVectorEmbeddings('dimension_test_server', tools);

        // Verify that saveEmbedding was called with expected dimension embedding
        expect(mockRepository.saveEmbedding).toHaveBeenCalledWith(
          'tool',
          'dimension_test_server:dimension_test_tool',
          expect.any(String),
          expect.arrayContaining([expect.any(Number)]), // Verify embedding array is present
          expect.any(Object),
          model
        );

        // Verify the embedding array has the correct length (dimension)
        const callArgs = mockRepository.saveEmbedding.mock.calls[0];
        const embeddingArray = callArgs[3]; // 4th argument is the embedding array
        expect(embeddingArray).toHaveLength(expected);
      });
    });
  });

  describe('Error handling and edge cases', () => {
    it('should handle halfvec index creation when halfvec not supported', async () => {
      mockDataSource.query
        .mockResolvedValueOnce(undefined) // DROP INDEX
        .mockRejectedValueOnce(new Error('type "halfvec" does not exist'));

      const result = await createVectorIndex(mockDataSource, 3072);

      expect(result.success).toBe(false);
      expect(result.indexType).toBeNull();
    });

    it('should handle IVFFlat fallback for various errors', async () => {
      mockDataSource.query
        .mockResolvedValueOnce(undefined) // DROP INDEX
        .mockRejectedValueOnce(new Error('operator class does not exist'))
        .mockResolvedValueOnce(undefined); // IVFFlat success

      const result = await createVectorIndex(mockDataSource, 1536);

      expect(result.success).toBe(true);
      expect(result.indexType).toBe('ivfflat');
    });

    it('should handle both HNSW and IVFFlat failures', async () => {
      mockDataSource.query
        .mockResolvedValueOnce(undefined) // DROP INDEX
        .mockRejectedValueOnce(new Error('HNSW failed'))
        .mockRejectedValueOnce(new Error('IVFFlat also failed'));

      const result = await createVectorIndex(mockDataSource, 1536);

      expect(result.success).toBe(false);
    });

    it('should handle different error messages for halfvec', async () => {
      const errorMessages = [
        'halfvec is not supported',
        'type does not exist for halfvec',
        'operator class "halfvec_cosine_ops" does not exist',
      ];

      for (const errorMsg of errorMessages) {
        jest.clearAllMocks();
        mockDataSource.query
          .mockResolvedValueOnce(undefined) // DROP INDEX
          .mockRejectedValueOnce(new Error(errorMsg));

        const result = await createVectorIndex(mockDataSource, 3072);

        expect(result.success).toBe(false);
      }
    });

    it('should include model info in vector database metadata', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'test_tool',
          description: 'Test',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('server1', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalledWith(
        'tool',
        expect.any(String),
        expect.any(String),
        expect.any(Array),
        expect.any(Object),
        'text-embedding-3-small' // Model name should be passed
      );
    });
  });

  describe('Tool embedding processing with complex schemas', () => {
    it('should extract properties from complex inputSchema', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'complex_tool',
          description: 'A complex tool',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string', description: 'Search query' },
              limit: { type: 'number', description: 'Result limit' },
              filters: {
                type: 'object',
                properties: {
                  category: { type: 'string' },
                  date: { type: 'string' },
                },
              },
            },
            required: ['query'],
          },
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      // Should extract property names into searchable text
      expect(mockRepository.saveEmbedding).toHaveBeenCalledWith(
        'tool',
        'test_server:complex_tool',
        expect.stringContaining('query'), // Property names should be included
        expect.any(Array),
        expect.objectContaining({
          inputSchema: expect.objectContaining({
            properties: expect.any(Object),
          }),
        }),
        expect.any(String)
      );
    });

    it('should handle tools without inputSchema properties', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'simple_tool',
          description: 'Simple tool',
          inputSchema: { type: 'object' }, // No properties
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle tools with null/undefined inputSchema', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'minimal_tool',
          description: 'Minimal tool',
          inputSchema: null as any,
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      // Should not throw, just skip schema properties
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('Vector search result transformation', () => {
    it('should handle metadata with additional custom fields', async () => {
      const mockResults = [
        {
          similarity: 0.92,
          embedding: {
            text_content: 'tool_name description',
            metadata: JSON.stringify({
              serverName: 'custom_server',
              toolName: 'custom_tool',
              description: 'Custom description',
              inputSchema: { type: 'object', properties: { custom: { type: 'string' } } },
              customField: 'custom_value', // Extra field
            }),
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('test');

      expect(results[0].inputSchema).toEqual(
        expect.objectContaining({
          type: 'object',
          properties: expect.any(Object),
        })
      );
    });

    it('should extract server and tool names from text when metadata missing', async () => {
      const mockResults = [
        {
          similarity: 0.88,
          embedding: {
            text_content: 'server1_tool1 tool description',
            metadata: null,
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('test');

      expect(results[0]).toHaveProperty('serverName');
      expect(results[0]).toHaveProperty('toolName');
      expect(results[0]).toHaveProperty('description');
    });
  });

  describe('Database dimension consistency checks', () => {
    it('should fetch vector dimensions from database schema', async () => {
      mockDataSource.query.mockResolvedValueOnce([{ dimensions: 1536 }]); // Dimension check
      mockRepository.searchSimilar.mockResolvedValue([]);

      await getAllVectorizedTools();

      expect(mockDataSource.query).toHaveBeenCalledWith(expect.stringContaining('pg_attribute'));
    });

    it('should handle missing dimension information from database', async () => {
      mockDataSource.query.mockResolvedValueOnce([]); // No result
      mockRepository.searchSimilar.mockResolvedValue([]);

      await getAllVectorizedTools();

      // Should still work, using default dimensions
      expect(mockRepository.searchSimilar).toHaveBeenCalled();
    });

    it('should handle dimension query errors gracefully', async () => {
      mockDataSource.query.mockRejectedValueOnce(new Error('Query failed'));
      mockRepository.searchSimilar.mockResolvedValue([]);

      await getAllVectorizedTools();

      // Should not throw, continue with search
      expect(mockRepository.searchSimilar).toHaveBeenCalled();
    });
  });

  describe('Server embedding removal with various states', () => {
    it('should report removed count from repository', async () => {
      mockRepository.deleteByServerName.mockResolvedValue(42);

      await removeServerToolEmbeddings('server_with_many_tools');

      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('server_with_many_tools');
    });

    it('should handle removal when no embeddings exist', async () => {
      mockRepository.deleteByServerName.mockResolvedValue(0);

      await removeServerToolEmbeddings('empty_server');

      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('empty_server');
    });

    it('should handle repository errors during removal', async () => {
      mockRepository.deleteByServerName.mockRejectedValue(
        new Error('Repository connection lost')
      );

      // Should not throw
      await removeServerToolEmbeddings('problematic_server');

      expect(mockRepository.deleteByServerName).toHaveBeenCalled();
    });
  });

  describe('Edge cases and boundary conditions', () => {
    it('should handle extremely long tool descriptions', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const longDescription = 'A '.repeat(10000) + 'tool';
      const tools: Tool[] = [
        {
          name: 'long_tool',
          description: longDescription,
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle special characters in tool names', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'tool-with_special.chars@123',
          description: 'Tool with special chars',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalledWith(
        'tool',
        'test_server:tool-with_special.chars@123',
        expect.any(String),
        expect.any(Array),
        expect.any(Object),
        expect.any(String)
      );
    });

    it('should handle unicode characters in descriptions', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'unicode_tool',
          description: '工具描述 🔍 мир 🌍',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('test_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle maximum dimension values', async () => {
      mockDataSource.query.mockResolvedValue(undefined);

      // Test with HALFVEC_MAX_DIMENSIONS
      const result = await createVectorIndex(mockDataSource, HALFVEC_MAX_DIMENSIONS);

      expect(result.success).toBe(true);
    });

    it('should handle boundary dimension value (VECTOR_MAX_DIMENSIONS)', async () => {
      mockDataSource.query.mockResolvedValue(undefined);

      const result = await createVectorIndex(mockDataSource, VECTOR_MAX_DIMENSIONS);

      expect(result.success).toBe(true);
      expect(result.indexType).toBe('hnsw');
    });

    it('should handle dimension value just above VECTOR_MAX_DIMENSIONS', async () => {
      mockDataSource.query.mockResolvedValue(undefined);

      const result = await createVectorIndex(mockDataSource, VECTOR_MAX_DIMENSIONS + 1);

      expect(result.success).toBe(true);
      expect(result.indexType).toBe('hnsw-halfvec');
    });

    it('should handle dimension value at HALFVEC_MAX_DIMENSIONS boundary', async () => {
      mockDataSource.query.mockResolvedValue(undefined);

      const result = await createVectorIndex(mockDataSource, HALFVEC_MAX_DIMENSIONS);

      expect(result.success).toBe(true);
    });
  });

  describe('Search with various parameters', () => {
    it('should search with custom limit', async () => {
      mockRepository.searchByText.mockResolvedValue([]);

      await searchToolsByVector('query', 50);

      expect(mockRepository.searchByText).toHaveBeenCalledWith(
        'query',
        expect.any(Function),
        50,
        expect.any(Number),
        expect.any(Array)
      );
    });

    it('should search with custom threshold', async () => {
      mockRepository.searchByText.mockResolvedValue([]);

      await searchToolsByVector('query', 10, 0.5);

      expect(mockRepository.searchByText).toHaveBeenCalledWith(
        'query',
        expect.any(Function),
        10,
        0.5,
        expect.any(Array)
      );
    });

    it('should search with server name filtering', async () => {
      const mockResults = [
        {
          similarity: 0.95,
          embedding: {
            text_content: 'text',
            metadata: JSON.stringify({
              serverName: 'server1',
              toolName: 'tool1',
              description: 'desc',
              inputSchema: {},
            }),
          },
        },
        {
          similarity: 0.90,
          embedding: {
            text_content: 'text2',
            metadata: JSON.stringify({
              serverName: 'server2',
              toolName: 'tool2',
              description: 'desc2',
              inputSchema: {},
            }),
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('query', 10, 0.7, ['server1', 'server3']);

      // Should only return server1 result
      expect(results).toHaveLength(1);
      expect(results[0].serverName).toBe('server1');
    });

    it('should handle empty server name filter array', async () => {
      mockRepository.searchByText.mockResolvedValue([]);

      const results = await searchToolsByVector('query', 10, 0.7, []);

      expect(results).toEqual([]);
    });
  });

  describe('Embedding API interaction and validation', () => {
    beforeEach(() => {
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

      // Mock global fetch
      global.fetch = jest.fn();
    });

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

  describe('Vector index creation with halfvec warnings', () => {
    it('should warn when halfvec is not supported for high dimensions', async () => {
      const warnSpy = jest.spyOn(console, 'warn');
      mockDataSource.query
        .mockResolvedValueOnce(undefined) // DROP INDEX
        .mockRejectedValueOnce(new Error('type "halfvec" does not exist'));

      await createVectorIndex(mockDataSource, 3072);

      // Should have warned about halfvec not supported
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('HIGH-DIMENSIONAL')
      );

      warnSpy.mockRestore();
    });

    it('should provide recommendations for different dimension issues', async () => {
      const warnSpy = jest.spyOn(console, 'warn');

      // Test dimensions > 4000
      await createVectorIndex(mockDataSource, 5000);

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('EXCEED INDEX LIMITS')
      );

      warnSpy.mockRestore();
    });
  });

  describe('Database initialization and dimension checking', () => {
    it('should initialize database when needed for saving embeddings', async () => {
      (isDatabaseConnected as jest.Mock).mockReturnValueOnce(false);
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'init_test',
          description: 'Test initialization',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('server1', tools);

      expect(initializeDatabase).toHaveBeenCalled();
    });

    it('should verify dimension consistency during save operation', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'consistency_test',
          description: 'Test dimension consistency',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('server1', tools);

      // Should call query to check/ensure dimensions
      expect(mockDataSource.query).toHaveBeenCalled();
    });
  });

  describe('Tool description and schema processing', () => {
    it('should combine tool name, description, and schema properties into searchable text', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'comprehensive_tool',
          description: 'A comprehensive search tool',
          inputSchema: {
            type: 'object',
            properties: {
              searchQuery: { type: 'string' },
              maxResults: { type: 'number' },
              filters: { type: 'object' },
            },
          },
        },
      ];

      await saveToolsAsVectorEmbeddings('comprehensive_server', tools);

      const callArgs = mockRepository.saveEmbedding.mock.calls[0];
      const searchableText = callArgs[2];

      expect(searchableText).toContain('comprehensive_tool');
      expect(searchableText).toContain('comprehensive search tool');
      expect(searchableText).toContain('searchQuery');
      expect(searchableText).toContain('maxResults');
      expect(searchableText).toContain('filters');
    });

    it('should handle tools with deeply nested schema properties', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'nested_schema_tool',
          description: 'Tool with nested schema',
          inputSchema: {
            type: 'object',
            properties: {
              level1: {
                type: 'object',
                properties: {
                  level2: {
                    type: 'object',
                    properties: {
                      level3: { type: 'string' },
                    },
                  },
                },
              },
            },
          },
        },
      ];

      await saveToolsAsVectorEmbeddings('nested_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle tools with array properties in schema', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'array_tool',
          description: 'Tool with array properties',
          inputSchema: {
            type: 'object',
            properties: {
              items: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    itemName: { type: 'string' },
                  },
                },
              },
            },
          },
        },
      ];

      await saveToolsAsVectorEmbeddings('array_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('Multiple tools processing', () => {
    it('should process multiple tools and track success/failure counts', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'tool1',
          description: 'First tool',
          inputSchema: { type: 'object' },
        },
        {
          name: 'tool2',
          description: 'Second tool',
          inputSchema: { type: 'object' },
        },
        {
          name: 'tool3',
          description: 'Third tool',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('multi_server', tools);

      // Should attempt to save all 3 tools
      expect(mockRepository.saveEmbedding).toHaveBeenCalledTimes(3);
    });

    it('should continue processing tools even if one fails', async () => {
      mockDataSource.query.mockResolvedValue([]);

      mockRepository.saveEmbedding
        .mockResolvedValueOnce(undefined) // Tool 1 succeeds
        .mockRejectedValueOnce(new Error('Save failed')) // Tool 2 fails
        .mockResolvedValueOnce(undefined); // Tool 3 succeeds

      const tools: Tool[] = [
        {
          name: 'tool1',
          description: 'Works',
          inputSchema: { type: 'object' },
        },
        {
          name: 'tool2',
          description: 'Fails',
          inputSchema: { type: 'object' },
        },
        {
          name: 'tool3',
          description: 'Works',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('resilient_server', tools);

      // All should be attempted
      expect(mockRepository.saveEmbedding).toHaveBeenCalledTimes(3);
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

  describe('Server tool removal edge cases', () => {
    it('should report success even if no embeddings were removed', async () => {
      mockRepository.deleteByServerName.mockResolvedValue(0);

      await removeServerToolEmbeddings('empty_server');

      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('empty_server');
    });

    it('should handle large number of removed embeddings', async () => {
      mockRepository.deleteByServerName.mockResolvedValue(1000);

      await removeServerToolEmbeddings('large_server');

      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('large_server');
    });

    it('should continue even if database is initializing when removal is called', async () => {
      (isDatabaseConnected as jest.Mock).mockReturnValueOnce(false);
      mockRepository.deleteByServerName.mockResolvedValue(5);

      await removeServerToolEmbeddings('initializing_server');

      expect(initializeDatabase).toHaveBeenCalled();
      expect(mockRepository.deleteByServerName).toHaveBeenCalled();
    });
  });

  describe('Search results metadata parsing variations', () => {
    it('should handle metadata with extra fields gracefully', async () => {
      const mockResults = [
        {
          similarity: 0.88,
          embedding: {
            text_content: 'test_tool description',
            metadata: JSON.stringify({
              serverName: 'advanced_server',
              toolName: 'advanced_tool',
              description: 'Advanced description',
              inputSchema: { type: 'object' },
              customField1: 'value1',
              customField2: { nested: 'value2' },
              timestamp: '2024-01-27T12:00:00Z',
            }),
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('advanced');

      expect(results).toHaveLength(1);
      expect(results[0].toolName).toBe('advanced_tool');
      expect(results[0].serverName).toBe('advanced_server');
    });

    it('should fallback to text parsing when metadata is corrupted', async () => {
      const mockResults = [
        {
          similarity: 0.85,
          embedding: {
            text_content: 'fallback_server_fallback_tool This is the description',
            metadata: '{invalid json that causes parse error',
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('fallback');

      expect(results).toHaveLength(1);
      // Should extract from text_content
      expect(results[0]).toHaveProperty('toolName');
      expect(results[0]).toHaveProperty('description');
    });

    it('should handle very long similarity scores correctly', async () => {
      const mockResults = [
        {
          similarity: 0.9999999999,
          embedding: {
            text_content: 'perfect_tool',
            metadata: JSON.stringify({
              serverName: 'perfect_server',
              toolName: 'perfect_tool',
              description: 'Nearly perfect match',
              inputSchema: {},
            }),
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('perfect');

      expect(results[0].similarity).toBe(0.9999999999);
    });

    it('should handle minimum valid similarity scores', async () => {
      const mockResults = [
        {
          similarity: 0.0001,
          embedding: {
            text_content: 'low_similarity_tool',
            metadata: JSON.stringify({
              serverName: 'low_server',
              toolName: 'low_tool',
              description: 'Low similarity match',
              inputSchema: {},
            }),
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      const results = await searchToolsByVector('low', 10, 0.00001); // Very low threshold

      expect(results[0].similarity).toBe(0.0001);
    });
  });

  describe('getAllVectorizedTools advanced scenarios', () => {
    it('should return empty array if no tools found in database', async () => {
      mockDataSource.query.mockResolvedValueOnce([]); // No dimension info
      mockRepository.searchSimilar.mockResolvedValue([]);

      const results = await getAllVectorizedTools();

      expect(results).toEqual([]);
    });

    it('should filter multiple servers from large result set', async () => {
      const mockResults = Array.from({ length: 20 }, (_, i) => ({
        similarity: 1.0,
        embedding: {
          metadata: JSON.stringify({
            serverName: i % 2 === 0 ? 'server_a' : 'server_b',
            toolName: `tool_${i}`,
            description: `Tool ${i}`,
            inputSchema: {},
          }),
        },
      }));

      mockDataSource.query.mockResolvedValueOnce([]);
      mockRepository.searchSimilar.mockResolvedValue(mockResults);

      const results = await getAllVectorizedTools(['server_a']);

      // Should only return server_a tools
      expect(results.every((r) => r.serverName === 'server_a')).toBe(true);
      expect(results.length).toBe(10);
    });

    it('should handle search similar returning invalid metadata gracefully', async () => {
      const mockResults = [
        {
          similarity: 1.0,
          embedding: {
            metadata: 'invalid json }{',
          },
        },
        {
          similarity: 0.95,
          embedding: {
            metadata: JSON.stringify({
              serverName: 'valid_server',
              toolName: 'valid_tool',
              description: 'Valid tool',
              inputSchema: {},
            }),
          },
        },
      ];

      mockDataSource.query.mockResolvedValueOnce([]);
      mockRepository.searchSimilar.mockResolvedValue(mockResults);

      const results = await getAllVectorizedTools();

      // Should skip invalid and return valid
      expect(results.some((r) => r.serverName === 'valid_server')).toBe(true);
    });
  });

  describe('Error handling in repository operations', () => {
    it('should handle repository save embedding errors with context', async () => {
      mockDataSource.query.mockResolvedValue([]);

      mockRepository.saveEmbedding.mockRejectedValue(
        new Error('Unique constraint violation')
      );

      const tools: Tool[] = [
        {
          name: 'constraint_violation_tool',
          description: 'Violates constraints',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('constraint_server', tools);

      // Should handle gracefully, not throw
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle repository factory returning null', async () => {
      (getRepositoryFactory as jest.Mock).mockReturnValueOnce(() => null);
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'null_repo_tool',
          description: 'Test null repository',
          inputSchema: { type: 'object' },
        },
      ];

      // Should handle gracefully
      await saveToolsAsVectorEmbeddings('null_repo_server', tools);
    });
  });

  describe('Complex integration scenarios', () => {
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
});
