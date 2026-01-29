/**
 * Integration tests for complex vector search workflows
 *
 * These tests verify complete end-to-end workflows involving:
 * - Full lifecycle operations: save, search, and remove
 * - Provider switching and configuration changes
 * - Multiple embedding providers (OpenAI, Azure OpenAI)
 * - State transitions across multiple service operations
 *
 * These tests use mocks but verify how multiple functions
 * interact across complete business workflows.
 */

jest.mock('openai');
jest.mock('../../src/db/index.js');
jest.mock('../../src/db/connection.js');
jest.mock('../../src/utils/smartRouting.js');
jest.mock('../../src/services/mcpService.js');

import {
  saveToolsAsVectorEmbeddings,
  searchToolsByVector,
  removeServerToolEmbeddings,
} from '../../src/services/vectorSearchService.js';
import { getRepositoryFactory } from '../../src/db/index.js';
import { getAppDataSource, isDatabaseConnected, initializeDatabase } from '../../src/db/connection.js';
import { getSmartRoutingConfig } from '../../src/utils/smartRouting.js';
import { Tool } from '../../src/types/index.js';

describe('Vector Search Complex Workflows - Integration Tests', () => {
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

  describe('Complete Lifecycle Workflows', () => {
    it('should handle complete lifecycle with Azure OpenAI: save -> search -> remove', async () => {
      // Configure Azure OpenAI
      mockSmartRoutingConfig.embeddingProvider = 'azure_openai';
      mockSmartRoutingConfig.azureOpenaiEndpoint = 'https://test.openai.azure.com/';
      mockSmartRoutingConfig.azureOpenaiApiKey = 'test-key';
      mockSmartRoutingConfig.azureOpenaiApiVersion = '2023-05-15';
      mockSmartRoutingConfig.azureOpenaiEmbeddingDeployment = 'embedding-deployment';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      // Mock successful Azure embedding response
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [
            {
              embedding: new Array(1536).fill(0.5),
            },
          ],
        }),
      });

      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'azure_lifecycle_tool',
          description: 'Tool for complete lifecycle test',
          inputSchema: { type: 'object' },
        },
      ];

      // STEP 1: Save tools with Azure OpenAI embeddings
      await saveToolsAsVectorEmbeddings('azure_lifecycle_server', tools);
      expect(mockRepository.saveEmbedding).toHaveBeenCalledWith(
        'tool',
        'azure_lifecycle_server:azure_lifecycle_tool',
        expect.any(String),
        expect.arrayContaining([expect.any(Number)]),
        expect.any(Object),
        'text-embedding-3-small'
      );

      // Verify exactly one embedding was saved (for the single tool)
      const saveCount = mockRepository.saveEmbedding.mock.calls.length;
      expect(saveCount).toBe(1);

      // STEP 2: Search for tools using vector similarity
      mockRepository.searchByText.mockResolvedValue([
        {
          similarity: 0.95,
          embedding: {
            text_content: 'azure_lifecycle_tool',
            metadata: JSON.stringify({
              serverName: 'azure_lifecycle_server',
              toolName: 'azure_lifecycle_tool',
              description: 'Tool for complete lifecycle test',
              inputSchema: {},
            }),
          },
        },
      ]);

      const searchResults = await searchToolsByVector('lifecycle test');
      expect(searchResults).toHaveLength(1);
      expect(searchResults[0].toolName).toBe('azure_lifecycle_tool');
      expect(searchResults[0].serverName).toBe('azure_lifecycle_server');
      expect(searchResults[0].similarity).toBe(0.95);

      // STEP 3: Remove embeddings for the server
      mockRepository.deleteByServerName.mockResolvedValue(1);
      await removeServerToolEmbeddings('azure_lifecycle_server');

      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('azure_lifecycle_server');
    });

    it('should handle provider switching workflow: OpenAI -> Azure OpenAI', async () => {
      // PHASE 1: Start with OpenAI provider
      mockSmartRoutingConfig.embeddingProvider = 'openai';
      mockSmartRoutingConfig.openaiApiBaseUrl = 'https://api.openai.com/v1';
      mockSmartRoutingConfig.openaiApiKey = 'sk-openai-key';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'provider_test_tool',
          description: 'Tool for provider switching test',
          inputSchema: { type: 'object' },
        },
      ];

      // Save with OpenAI
      await saveToolsAsVectorEmbeddings('provider_switch_server', tools);
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();

      const openaiSaveCount = mockRepository.saveEmbedding.mock.calls.length;

      // PHASE 2: Switch to Azure OpenAI provider
      jest.clearAllMocks();
      mockSmartRoutingConfig.embeddingProvider = 'azure_openai';
      mockSmartRoutingConfig.azureOpenaiEndpoint = 'https://test.openai.azure.com/';
      mockSmartRoutingConfig.azureOpenaiApiKey = 'azure-key';
      mockSmartRoutingConfig.azureOpenaiApiVersion = '2023-05-15';
      mockSmartRoutingConfig.azureOpenaiEmbeddingDeployment = 'embedding-deployment';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      // Mock Azure response
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [
            {
              embedding: new Array(1536).fill(0.5),
            },
          ],
        }),
      });

      mockDataSource.query.mockResolvedValue([]);

      // Save again with Azure (simulating provider switch in same server)
      await saveToolsAsVectorEmbeddings('provider_switch_server', tools);
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();

      const azureSaveCount = mockRepository.saveEmbedding.mock.calls.length;

      // Both phases should have saved embeddings
      expect(openaiSaveCount).toBeGreaterThan(0);
      expect(azureSaveCount).toBeGreaterThan(0);
    });

    it('should handle search with multiple result filtering', async () => {
      mockDataSource.query.mockResolvedValue([]);

      // Setup multiple results from different servers
      const mockResults = [
        {
          similarity: 0.98,
          embedding: {
            text_content: 'high_relevance_tool',
            metadata: JSON.stringify({
              serverName: 'server_a',
              toolName: 'high_relevance_tool',
              description: 'Very relevant tool',
              inputSchema: { type: 'object' },
            }),
          },
        },
        {
          similarity: 0.92,
          embedding: {
            text_content: 'medium_relevance_tool',
            metadata: JSON.stringify({
              serverName: 'server_b',
              toolName: 'medium_relevance_tool',
              description: 'Medium relevance',
              inputSchema: { type: 'object' },
            }),
          },
        },
        {
          similarity: 0.75,
          embedding: {
            text_content: 'low_relevance_tool',
            metadata: JSON.stringify({
              serverName: 'server_c',
              toolName: 'low_relevance_tool',
              description: 'Low relevance',
              inputSchema: { type: 'object' },
            }),
          },
        },
      ];

      mockRepository.searchByText.mockResolvedValue(mockResults);

      // Search with specific server filter
      const results = await searchToolsByVector('test query', 10, 0.7, ['server_a', 'server_b']);

      expect(results).toHaveLength(2);
      expect(results[0].serverName).toBe('server_a');
      expect(results[1].serverName).toBe('server_b');

      // Verify high similarity result is first
      expect(results[0].similarity).toBeGreaterThan(results[1].similarity);
    });

    it('should handle concurrent save operations for multiple servers', async () => {
      mockDataSource.query.mockResolvedValue([]);

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [{ embedding: new Array(1536).fill(0.5) }],
        }),
      });

      const servers = [
        { name: 'server_1', tools: 2 },
        { name: 'server_2', tools: 3 },
        { name: 'server_3', tools: 1 },
      ];

      // Simulate saving tools for multiple servers
      for (const server of servers) {
        const tools: Tool[] = Array.from({ length: server.tools }, (_, i) => ({
          name: `tool_${i}`,
          description: `Tool ${i} for ${server.name}`,
          inputSchema: { type: 'object' },
        }));

        await saveToolsAsVectorEmbeddings(server.name, tools);
      }

      // Verify all embeddings were saved
      const totalExpectedSaves = servers.reduce((sum, s) => sum + s.tools, 0);
      expect(mockRepository.saveEmbedding).toHaveBeenCalledTimes(totalExpectedSaves);
    });

    it('should handle large-scale search result processing', async () => {
      mockDataSource.query.mockResolvedValue([]);

      // Simulate 100 search results
      const largeResultSet = Array.from({ length: 100 }, (_, i) => ({
        similarity: 1.0 - i * 0.001, // Decreasing similarity
        embedding: {
          text_content: `tool_${i}`,
          metadata: JSON.stringify({
            serverName: `server_${Math.floor(i / 10)}`,
            toolName: `tool_${i}`,
            description: `Tool ${i} description`,
            inputSchema: { type: 'object' },
          }),
        },
      }));

      mockRepository.searchByText.mockResolvedValue(largeResultSet);

      // Search without limit - should return all results
      const allResults = await searchToolsByVector('query');

      expect(allResults.length).toBeGreaterThan(0);
      // Results should be ordered by similarity (highest first)
      for (let i = 1; i < allResults.length; i++) {
        expect(allResults[i].similarity).toBeLessThanOrEqual(allResults[i - 1].similarity);
      }

      // Verify that results are properly transformed from metadata
      expect(allResults[0]).toHaveProperty('serverName');
      expect(allResults[0]).toHaveProperty('toolName');
      expect(allResults[0]).toHaveProperty('similarity');
    });
  });

  describe('Error Recovery and Resilience', () => {
    it('should recover from partial embedding failures in batch operations', async () => {
      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'tool_1',
          description: 'First tool',
          inputSchema: { type: 'object' },
        },
        {
          name: 'tool_2',
          description: 'Second tool',
          inputSchema: { type: 'object' },
        },
        {
          name: 'tool_3',
          description: 'Third tool',
          inputSchema: { type: 'object' },
        },
      ];

      // Simulate: success, failure, success
      mockRepository.saveEmbedding
        .mockResolvedValueOnce(undefined)
        .mockRejectedValueOnce(new Error('Save failed'))
        .mockResolvedValueOnce(undefined);

      await saveToolsAsVectorEmbeddings('partial_fail_server', tools);

      // Should attempt all saves despite failures
      expect(mockRepository.saveEmbedding).toHaveBeenCalledTimes(3);
    });

    it('should handle provider configuration errors gracefully', async () => {
      mockSmartRoutingConfig.embeddingProvider = 'azure_openai';
      mockSmartRoutingConfig.azureOpenaiEndpoint = '';
      mockSmartRoutingConfig.azureOpenaiApiKey = '';
      (getSmartRoutingConfig as jest.Mock).mockResolvedValue(mockSmartRoutingConfig);

      mockDataSource.query.mockResolvedValue([]);

      const tools: Tool[] = [
        {
          name: 'error_handling_tool',
          description: 'Test error handling',
          inputSchema: { type: 'object' },
        },
      ];

      // Should fallback gracefully without throwing
      await saveToolsAsVectorEmbeddings('error_server', tools);

      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });
  });

  describe('State Consistency Across Operations', () => {
    it('should maintain consistency when searching before saving', async () => {
      mockDataSource.query.mockResolvedValue([]);

      // Search without any saved embeddings
      mockRepository.searchByText.mockResolvedValue([]);

      const results = await searchToolsByVector('early search');
      expect(results).toEqual([]);

      // Then save embeddings
      const tools: Tool[] = [
        {
          name: 'consistency_tool',
          description: 'Test consistency',
          inputSchema: { type: 'object' },
        },
      ];

      await saveToolsAsVectorEmbeddings('consistency_server', tools);
      expect(mockRepository.saveEmbedding).toHaveBeenCalled();
    });

    it('should handle removal of non-existent server gracefully', async () => {
      mockRepository.deleteByServerName.mockResolvedValue(0);

      // Remove from server that may not have embeddings
      await removeServerToolEmbeddings('non_existent_server');

      expect(mockRepository.deleteByServerName).toHaveBeenCalledWith('non_existent_server');
    });
  });
});
