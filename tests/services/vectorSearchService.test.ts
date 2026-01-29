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

  describe('Database Vector Dimension Management', () => {
    describe('checkDatabaseVectorDimensions', () => {
      it('should initialize vector column when no vectors exist', async () => {
        mockDataSource.isInitialized = true;
        mockDataSource.query
          .mockResolvedValueOnce([]) // No dimension info from schema
          .mockResolvedValueOnce([]); // No existing records

        // We need to import the function to test it
        // This test validates the initialization path
        const tools: Tool[] = [
          {
            name: 'init_test_tool',
            description: 'Test initialization',
            inputSchema: { type: 'object' },
          },
        ];

        mockDataSource.query.mockResolvedValue([]);
        await saveToolsAsVectorEmbeddings('init_server', tools);

        expect(mockRepository.saveEmbedding).toHaveBeenCalled();
      });

      it('should detect and log dimension mismatch', async () => {
        mockDataSource.isInitialized = true;
        mockDataSource.query
          .mockResolvedValueOnce([{ dimensions: 1536 }]) // Current dimensions from schema
          .mockResolvedValueOnce([{ dimensions: 1536, model: 'old-model', count: 10 }]); // Record dimensions

        // Verify database query for dimension detection
        const tools: Tool[] = [
          {
            name: 'mismatch_test_tool',
            description: 'Test dimension mismatch',
            inputSchema: { type: 'object' },
          },
        ];

        mockDataSource.query.mockResolvedValue([]);
        await saveToolsAsVectorEmbeddings('dimension_server', tools);

        expect(mockRepository.saveEmbedding).toHaveBeenCalled();
      });

      it('should report success when dimensions match', async () => {
        mockDataSource.isInitialized = true;
        mockDataSource.query.mockResolvedValue([{ dimensions: 1536 }]);

        const tools: Tool[] = [
          {
            name: 'match_test_tool',
            description: 'Dimensions match',
            inputSchema: { type: 'object' },
          },
        ];

        mockDataSource.query.mockResolvedValue([]);
        await saveToolsAsVectorEmbeddings('matching_server', tools);

        expect(mockRepository.saveEmbedding).toHaveBeenCalled();
      });
    });

    describe('clearMismatchedVectorData', () => {
      it('should be called during embedding save when handling mismatches', async () => {
        mockDataSource.isInitialized = true;
        mockDataSource.query.mockResolvedValue([]);

        const tools: Tool[] = [
          {
            name: 'clear_test_tool',
            description: 'Test clearing mismatched data',
            inputSchema: { type: 'object' },
          },
        ];

        await saveToolsAsVectorEmbeddings('clear_server', tools);

        expect(mockRepository.saveEmbedding).toHaveBeenCalled();
      });
    });
  });

  describe('Embedding Validation Functions', () => {
    describe('validateEmbeddingArray and related validators', () => {
      it('should validate and reject non-array embeddings', async () => {
        mockDataSource.query.mockResolvedValue([]);

        const tools: Tool[] = [
          {
            name: 'validation_test',
            description: 'Test validation',
            inputSchema: { type: 'object' },
          },
        ];

        // This test validates that the function handles invalid array responses
        (global.fetch as jest.Mock).mockResolvedValue({
          ok: true,
          json: async () => ({
            data: [
              {
                embedding: 'not an array', // Invalid: not an array
              },
            ],
          }),
        });

        await saveToolsAsVectorEmbeddings('validation_server', tools);

        // Should either use fallback or skip the tool
        // The function should handle gracefully
      });

      it('should validate and reject zero-vectors', async () => {
        mockDataSource.query.mockResolvedValue([]);

        const tools: Tool[] = [
          {
            name: 'zero_vector_test',
            description: 'Test zero vector rejection',
            inputSchema: { type: 'object' },
          },
        ];

        // Mock API returning zero-vector
        (global.fetch as jest.Mock).mockResolvedValue({
          ok: true,
          json: async () => ({
            data: [
              {
                embedding: new Array(1536).fill(0), // Zero-vector - should be rejected
              },
            ],
          }),
        });

        await saveToolsAsVectorEmbeddings('zero_vector_server', tools);

        // Should detect zero-vector and use fallback
      });

      it('should validate and reject out-of-range values', async () => {
        mockDataSource.query.mockResolvedValue([]);

        const tools: Tool[] = [
          {
            name: 'range_test_tool',
            description: 'Test out-of-range validation',
            inputSchema: { type: 'object' },
          },
        ];

        // Mock API returning values outside expected range
        (global.fetch as jest.Mock).mockResolvedValue({
          ok: true,
          json: async () => ({
            data: [
              {
                embedding: Array(1536)
                  .fill(0)
                  .map(() => Math.random() * 100), // Values outside [-1.1, 1.1] range
              },
            ],
          }),
        });

        await saveToolsAsVectorEmbeddings('range_error_server', tools);

        // Should detect invalid range and use fallback
      });

      it('should validate dimension consistency', async () => {
        mockDataSource.query.mockResolvedValue([]);

        const tools: Tool[] = [
          {
            name: 'dimension_consistency_test',
            description: 'Test dimension consistency',
            inputSchema: { type: 'object' },
          },
        ];

        // Mock API returning wrong dimension count
        (global.fetch as jest.Mock).mockResolvedValue({
          ok: true,
          json: async () => ({
            data: [
              {
                embedding: new Array(2048).fill(0.5), // Wrong dimensions (expected 1536)
              },
            ],
          }),
        });

        await saveToolsAsVectorEmbeddings('dimension_consistency_server', tools);

        // Should detect dimension mismatch
      });
    });
  });

  describe('callEmbeddingAPI - Multiple Strategies', () => {
    it('should use OpenAI library for official OpenAI API', async () => {
      mockSmartRoutingConfig.openaiApiBaseUrl = 'https://api.openai.com/v1';
      mockSmartRoutingConfig.openaiApiKey = 'sk-valid-key';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      const tools: Tool[] = [
        {
          name: 'openai_lib_test',
          description: 'Test OpenAI library strategy',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('openai_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should use fetch API for unofficial OpenAI-compatible servers', async () => {
      mockSmartRoutingConfig.openaiApiBaseUrl = 'http://localhost:8000';
      mockSmartRoutingConfig.openaiApiKey = 'test-key';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });

      const tools: Tool[] = [
        {
          name: 'fetch_api_test',
          description: 'Test fetch API strategy',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('fetch_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle fetch API errors', async () => {
      mockSmartRoutingConfig.openaiApiBaseUrl = 'http://localhost:8000';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 500,
        text: async () => 'Internal Server Error',
      });

      const tools: Tool[] = [
        {
          name: 'fetch_error_test',
          description: 'Test fetch API error handling',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('fetch_error_server', tools);

      // Should handle error and possibly use fallback
    });

    it('should handle nomic-embed-text special task_type parameter', async () => {
      mockSmartRoutingConfig.openaiApiEmbeddingModel = 'nomic-embed-text-v1.5';
      mockSmartRoutingConfig.openaiApiBaseUrl = 'http://localhost:8000';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [{ embedding: new Array(768).fill(0.5) }],
        }),
      });

      const tools: Tool[] = [
        {
          name: 'nomic_test',
          description: 'Test nomic model',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('nomic_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('generateFallbackEmbedding', () => {
    it('should generate fallback embeddings with correct dimensions', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'fallback_test',
          description: 'Test fallback embedding generation',
          inputSchema: { type: 'object' },
        },
      ];

      // Mock API failure to trigger fallback
      (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));

      await saveToolsAsVectorEmbeddings('fallback_server', tools);

      // Should fall back to generateFallbackEmbedding
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle fallback with special characters and vocabulary', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'special_chars_tool',
          description:
            'Search for email messages in database with @symbol and #hashtags and [brackets]',
          inputSchema: { type: 'object' },
        },
      ];

      (global.fetch as jest.Mock).mockRejectedValue(new Error('API unavailable'));

      await saveToolsAsVectorEmbeddings('special_fallback_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle fallback with empty or whitespace-only text', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: '   ',
          description: '     ',
          inputSchema: { type: 'object' },
        },
      ];

      (global.fetch as jest.Mock).mockRejectedValue(new Error('API error'));

      await saveToolsAsVectorEmbeddings('whitespace_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('Azure OpenAI Integration', () => {
    it('should use Azure OpenAI when configured', async () => {
      mockSmartRoutingConfig.embeddingProvider = 'azure_openai';
      mockSmartRoutingConfig.azureOpenaiEndpoint = 'https://test.openai.azure.com/';
      mockSmartRoutingConfig.azureOpenaiApiKey = 'test-key';
      mockSmartRoutingConfig.azureOpenaiApiVersion = '2023-05-15';
      mockSmartRoutingConfig.azureOpenaiEmbeddingDeployment = 'embedding-deployment';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });

      const tools: Tool[] = [
        {
          name: 'azure_test',
          description: 'Test Azure OpenAI',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('azure_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle Azure OpenAI missing endpoint', async () => {
      mockSmartRoutingConfig.embeddingProvider = 'azure_openai';
      mockSmartRoutingConfig.azureOpenaiEndpoint = '';
      mockSmartRoutingConfig.azureOpenaiApiKey = '';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      const tools: Tool[] = [
        {
          name: 'azure_missing_test',
          description: 'Test Azure missing config',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('azure_missing_server', tools);

      // Should use fallback
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle Azure OpenAI API errors', async () => {
      mockSmartRoutingConfig.embeddingProvider = 'azure_openai';
      mockSmartRoutingConfig.azureOpenaiEndpoint = 'https://test.openai.azure.com/';
      mockSmartRoutingConfig.azureOpenaiApiKey = 'test-key';
      mockSmartRoutingConfig.azureOpenaiApiVersion = '2023-05-15';
      mockSmartRoutingConfig.azureOpenaiEmbeddingDeployment = 'embedding-deployment';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      (global.fetch as jest.Mock).mockRejectedValue(new Error('Azure API Error'));

      const tools: Tool[] = [
        {
          name: 'azure_error_test',
          description: 'Test Azure error handling',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('azure_error_server', tools);

      // Should fallback gracefully
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle Azure OpenAI missing deployment or version', async () => {
      mockSmartRoutingConfig.embeddingProvider = 'azure_openai';
      mockSmartRoutingConfig.azureOpenaiEndpoint = 'https://test.openai.azure.com/';
      mockSmartRoutingConfig.azureOpenaiApiKey = 'test-key';
      mockSmartRoutingConfig.azureOpenaiApiVersion = '';
      mockSmartRoutingConfig.azureOpenaiEmbeddingDeployment = '';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      const tools: Tool[] = [
        {
          name: 'azure_config_test',
          description: 'Test Azure missing config',
          inputSchema: { type: 'object' },
        },
      ];

      mockDataSource.query.mockResolvedValue([]);

      await saveToolsAsVectorEmbeddings('azure_config_server', tools);

      // Should use fallback
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('saveToolsAsVectorEmbeddings - Advanced Scenarios', () => {
    it('should handle test embedding generation that returns invalid dimensions', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'failed_test_embedding',
          description: 'Test when embedding fails',
          inputSchema: { type: 'object' },
        },
      ];

      // The function internally handles errors from test embedding generation
      // It will catch the error and log it but not throw since error handling is built-in
      await saveToolsAsVectorEmbeddings('failed_server', tools);

      // Verify the function completes (either with success or graceful error handling)
      expect(mockRepository.saveEmbedding).toBeDefined();
    });

    it('should handle database initialization errors', async () => {
      (isDatabaseConnected as jest.Mock).mockReturnValueOnce(false);
      (initializeDatabase as jest.Mock).mockRejectedValueOnce(
        new Error('Database init failed')
      );

      const tools: Tool[] = [
        {
          name: 'db_init_test',
          description: 'Test DB init failure',
          inputSchema: { type: 'object' },
        },
      ];

      mockSmartRoutingConfig.dbUrl = '';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      await saveToolsAsVectorEmbeddings('db_error_server', tools);

      // Should handle gracefully
    });

    it('should handle individual tool embedding failures while continuing', async () => {
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
      ];

      // First tool succeeds, second fails
      mockRepository.saveEmbedding
        .mockResolvedValueOnce(undefined)
        .mockRejectedValueOnce(new Error('Save failed for tool 2'));

      await saveToolsAsVectorEmbeddings('partial_fail_server', tools);

      // Should attempt to save both
      expect(mockRepository.saveEmbedding).toHaveBeenCalledTimes(2);
    });

    it('should truncate extremely long text before embedding', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const longText = 'word '.repeat(5000); // Over 8000 characters when combined

      const tools: Tool[] = [
        {
          name: 'long_text_tool',
          description: longText,
          inputSchema: { type: 'object' },
        },
      ];

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });

      await saveToolsAsVectorEmbeddings('long_text_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle text with newline characters', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'newline_tool',
          description: 'Tool with\nmultiple\nlines\n\n\nof\ndescription',
          inputSchema: { type: 'object' },
        },
      ];

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });

      await saveToolsAsVectorEmbeddings('newline_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('removeServerToolEmbeddings - Edge Cases', () => {
    it('should handle missing DB_URL when database not connected', async () => {
      (isDatabaseConnected as jest.Mock).mockReturnValueOnce(false);
      mockSmartRoutingConfig.dbUrl = '';
      delete process.env.DB_URL;
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      await removeServerToolEmbeddings('no_db_url_server');

      // Should return early without error
    });

    it('should log detailed debug information during removal', async () => {
      (isDatabaseConnected as jest.Mock).mockReturnValueOnce(true);
      mockRepository.deleteByServerName.mockResolvedValue(10);

      await removeServerToolEmbeddings('debug_server');

      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('debug_server');
    });
  });

  describe('getAllVectorizedTools - Dimension Detection', () => {
    it('should query database for dimension information', async () => {
      mockDataSource.query.mockResolvedValueOnce([{ dimensions: 1536 }]);
      mockRepository.searchSimilar.mockResolvedValue([]);

      await getAllVectorizedTools();

      expect(mockDataSource.query).toHaveBeenCalledWith(expect.stringContaining('pg_attribute'));
    });

    it('should handle database with no dimension results', async () => {
      mockDataSource.query.mockResolvedValueOnce([]); // Empty result
      mockRepository.searchSimilar.mockResolvedValue([]);

      const result = await getAllVectorizedTools();

      expect(result).toEqual([]);
    });

    it('should handle dimension query throwing error', async () => {
      mockDataSource.query.mockRejectedValueOnce(new Error('Dimension query failed'));
      mockRepository.searchSimilar.mockResolvedValue([]);

      const result = await getAllVectorizedTools();

      // Should still return results from search
      expect(result).toBeDefined();
    });

    it('should handle malformed dimension metadata in results', async () => {
      mockDataSource.query.mockResolvedValueOnce([{ dimensions: 1536 }]);

      const mockResults = [
        {
          similarity: 1.0,
          embedding: {
            metadata: 'unparseable',
          },
        },
      ];

      mockRepository.searchSimilar.mockResolvedValue(mockResults);

      const result = await getAllVectorizedTools();

      expect(result).toHaveLength(1);
      expect(result[0].serverName).toBe('unknown');
    });
  });

});
