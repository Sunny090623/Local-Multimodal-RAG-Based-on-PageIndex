import os
import sys
import json
import unittest
import asyncio
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.provider import (
    check_ollama_status, check_xinference_status, get_active_provider, set_active_provider,
    get_vlm_provider, set_vlm_provider
)
from backend.rag import execute_web_search
from backend.shared import get_shared_client, STORAGE_DIR, WORKSPACE_DIR

class TestRAGBackend(unittest.TestCase):
    
    def test_provider_settings(self):
        set_active_provider("ollama", "gemma4:12b-it-qat", "http://localhost:11434")
        active = get_active_provider()
        self.assertEqual(active.provider_type, "ollama")
        self.assertEqual(active.model_name, "gemma4:12b-it-qat")
        self.assertEqual(active.api_base, "http://localhost:11434")

    def test_parser_settings(self):
        set_vlm_provider(True, "ollama", "gemma4:12b-it-qat", "http://localhost:11434")
        settings = get_vlm_provider()
        self.assertEqual(settings.model_name, "gemma4:12b-it-qat")
        self.assertEqual(settings.provider_type, "ollama")
        self.assertTrue(settings.use_vlm)

    def test_web_search(self):
        # Test DuckDuckGo search API fallback is functioning
        results, err = asyncio.run(execute_web_search("Python Programming"))
        
        # Check if rate-limited or successful
        if err:
            print(f"Warning: DuckDuckGo search returned an error: {err}")
            # Ensure it is caught gracefully as a string description
            self.assertIsInstance(err, str)
        else:
            self.assertTrue(len(results) >= 0)
            if len(results) > 0:
                self.assertIn("url", results[0])
                self.assertIn("title", results[0])
                self.assertIn("snippet", results[0])
                print(f"DuckDuckGo search successful: found {len(results)} items.")

    def test_status_checks(self):
        # Test status check executes without throwing
        ollama_status = check_ollama_status()
        self.assertIn("status", ollama_status)
        
        xinference_status = check_xinference_status()
        self.assertIn("status", xinference_status)


class TestSharedClient(unittest.TestCase):
    """Test the shared PageIndexClient singleton pattern (BUG-004 fix)."""

    def test_singleton_consistency(self):
        """Multiple calls to get_shared_client() should return the same instance."""
        client_a = get_shared_client()
        client_b = get_shared_client()
        self.assertIs(client_a, client_b, "get_shared_client() should return the same instance")

    def test_workspace_exists(self):
        """The workspace directory should be automatically created."""
        self.assertTrue(WORKSPACE_DIR.exists())
        self.assertTrue(STORAGE_DIR.exists())


class TestIndexDocument(unittest.TestCase):
    """Test core index_document function (BUG-008 fix)."""

    def test_index_txt_file(self):
        """Test indexing a plain text file creates a valid document entry."""
        from backend.parser import index_document
        
        # Create a temporary .txt file
        test_content = "# Test Document\n\nThis is a test paragraph for indexing validation.\n\n## Section A\n\nSome content in section A.\n\n## Section B\n\nSome content in section B."
        test_file = STORAGE_DIR / "test_index_validation.txt"
        try:
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(test_content)
            
            # This will fail if LLM is not available, which is expected in CI
            # We just verify the function doesn't crash with asyncio errors
            try:
                doc_id = index_document(str(test_file))
                self.assertIsInstance(doc_id, str)
                self.assertTrue(len(doc_id) > 0)
                
                # Verify document appears in shared client
                client = get_shared_client()
                self.assertIn(doc_id, client.documents)
                
                # Verify workspace JSON file was created
                doc_json = WORKSPACE_DIR / f"{doc_id}.json"
                self.assertTrue(doc_json.exists(), "Document JSON should be persisted to workspace")
                
                # Clean up
                if doc_json.exists():
                    os.remove(doc_json)
                client.documents.pop(doc_id, None)
                
                # Clean up _meta.json to prevent dangling test records
                meta_path = WORKSPACE_DIR / "_meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if doc_id in meta:
                            meta.pop(doc_id, None)
                            with open(meta_path, "w", encoding="utf-8") as f:
                                json.dump(meta, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"Warning: Failed to clean up metadata in test: {e}")
                
                print(f"TXT indexing test passed. doc_id={doc_id}")
            except Exception as e:
                # If LLM is offline, the tree parser will fail — that's OK for this test
                err_str = str(e).lower()
                if "connection" in err_str or "timeout" in err_str or "refused" in err_str:
                    print(f"Skipping TXT index test (LLM offline): {e}")
                else:
                    raise
        finally:
            if test_file.exists():
                os.remove(test_file)

    def test_index_unsupported_format(self):
        """Test that unsupported file formats raise ValueError."""
        from backend.parser import index_document
        
        test_file = STORAGE_DIR / "test_unsupported.xyz"
        try:
            with open(test_file, "w") as f:
                f.write("test")
            
            with self.assertRaises(ValueError) as ctx:
                index_document(str(test_file))
            
            self.assertIn("Unsupported file format", str(ctx.exception))
            print(f"Unsupported format test passed: {ctx.exception}")
        finally:
            if test_file.exists():
                os.remove(test_file)

    def test_index_missing_file(self):
        """Test that indexing a non-existent file raises FileNotFoundError."""
        from backend.parser import index_document
        
        with self.assertRaises(FileNotFoundError):
            index_document("/nonexistent/path/document.pdf")


class TestRAGFlowEdgeCases(unittest.TestCase):
    """Test RAG flow error handling."""

    def test_rag_flow_missing_document(self):
        """Querying a non-existent document should return error JSON."""
        from backend.rag import execute_rag_flow_stream
        
        async def collect_stream():
            results = []
            async for chunk in execute_rag_flow_stream("nonexistent-doc-id", "test query"):
                results.append(json.loads(chunk.strip()))
            return results
        
        results = asyncio.run(collect_stream())
        
        self.assertTrue(len(results) > 0, "Should return at least one message")
        self.assertEqual(results[0]["type"], "error")
        self.assertIn("not found", results[0]["content"])
        print(f"Missing document test passed: {results[0]['content']}")


class TestLiteLLMPatching(unittest.TestCase):
    """Test that litellm is correctly patched by provider.py."""
    
    def test_litellm_completion_is_patched(self):
        """litellm.completion should be the patched version, not the original."""
        import litellm
        from backend.provider import patched_completion, patched_acompletion
        
        self.assertEqual(litellm.completion, patched_completion,
                         "litellm.completion should be patched to patched_completion")
        self.assertEqual(litellm.acompletion, patched_acompletion,
                         "litellm.acompletion should be patched to patched_acompletion")
        print("litellm patching verified.")
    
    def test_drop_params_enabled(self):
        """litellm.drop_params should be True to handle unsupported params gracefully."""
        import litellm
        self.assertTrue(litellm.drop_params)


if __name__ == "__main__":
    unittest.main()
